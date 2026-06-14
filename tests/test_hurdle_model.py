"""Behavioral tests for HurdleModel (issue #75 / risk D-13).

These replace the former mock-brittle `test_fit`/`test_predict` (which asserted call-wiring
and failed env/start-method-dependently) with real fit -> predict CHARACTERIZATION tests on a
tiny panel. They pin the *mechanism* of the `binary x positive` point estimator — which is robust
— rather than what a tiny-data learner happens to predict (which would be flaky). Pattern follows
`tests/test_target_transforms.py`. The current behaviour they document (a hard {0,1} gate) is the
finding of Story A (#78); a deliberate change to a probability gate should break
`test_binary_stage_emits_class_labels` on purpose.
"""
import pytest
import pandas as pd
import numpy as np
from views_stepshifter.models.hurdle_model import HurdleModel


@pytest.fixture
def sample_config():
    return {
        "steps": [1, 2, 3],
        "targets": ["target"],
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor",
        "parameters": {"clf": {"n_estimators": 100, "max_depth": 10}, "reg": {}},
        "sweep": False,
        "target_transform": "identity",
        "metrics": ["test_metric"],
    }


@pytest.fixture
def sample_partitioner_dict():
    return {"train": [0, 10], "test": [11, 20]}


@pytest.fixture
def sample_dataframe():
    index = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    data = {
        "feature1": np.arange(63, dtype=np.float64),
        "feature2": np.arange(63, 126, dtype=np.float64),
        # country 0 -> all 0 (no-event series); countries 1,2 -> all positive
        "target": np.array([0, 1, 2] * 21, dtype=np.float64),
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture(scope="module")
def fitted_hurdle():
    """A real HurdleModel fitted once on a tiny identity-space panel (reused read-only).

    The fit/predict device path (serial vs ProcessPoolExecutor) is decided by the environment
    (`get_device_params`), exactly as in production: no GPU -> the CPU/pool path CI runs;
    a visible GPU -> the serial path. Both compute the same `binary x positive` product, so these
    behavioural assertions hold on either. (Run `CUDA_VISIBLE_DEVICES="" pytest` to exercise the
    CI/CPU path locally — forcing CPU in-process while CUDA is initialized deadlocks on fork.)"""
    cfg = {
        "steps": [1, 2],
        "targets": ["target"],
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor",
        "parameters": {"clf": {"n_estimators": 5}, "reg": {"n_estimators": 5}},
        "sweep": False,
        "target_transform": "identity",
    }
    idx = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    df = pd.DataFrame(
        {
            "feature1": np.arange(63, dtype=np.float64),
            "target": np.array([0, 1, 2] * 21, dtype=np.float64),
        },
        index=idx,
    )
    model = HurdleModel(cfg, {"train": [0, 10], "test": [11, 20]})
    model.fit(df)
    return model


def _forecast_stage_frames(model):
    """Reconstruct the (binary, positive) per-step forecast frames the product is built from.

    Mirrors HurdleModel.predict('forecasting') exactly: concat of `_predict_by_step` over steps
    for sequence 0, sorted. Deterministic given a fitted model (independent of pool ordering)."""
    binary = pd.concat(
        [model._predict_by_step(model._models[s][0], s, 0) for s in model._steps]
    ).sort_index()
    positive = pd.concat(
        [model._predict_by_step(model._models[s][1], s, 0) for s in model._steps]
    ).sort_index()
    return binary, positive


def test_initialization(sample_config, sample_partitioner_dict):
    """The model initializes with the correct attributes."""
    model = HurdleModel(sample_config, sample_partitioner_dict)
    assert model._steps == sample_config["steps"]
    assert model._targets == sample_config["targets"][0]
    assert model._clf_params == sample_config["parameters"]["clf"]
    assert model._reg_params == sample_config["parameters"]["reg"]


def test_fit_trains_a_two_stage_model_per_step(fitted_hurdle):
    """fit() produces, for every step, a (binary, positive) model pair."""
    assert fitted_hurdle.is_fitted_ is True
    assert set(fitted_hurdle._models) == set(fitted_hurdle._steps)
    for step, pair in fitted_hurdle._models.items():
        assert isinstance(pair, tuple) and len(pair) == 2, f"step {step} is not a 2-tuple"


def test_binary_stage_emits_class_labels(fitted_hurdle):
    """D-33 characterization: the binary stage outputs hard {0,1} class LABELS, not probabilities.

    This pins current behaviour. A deliberate switch to a probability gate must break this test."""
    binary, _ = _forecast_stage_frames(fitted_hurdle)
    vals = binary.iloc[:, 0].to_numpy()
    assert np.array_equal(vals, np.round(vals)), "binary stage is not integral (probabilities leaked?)"
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_point_forecast_is_binary_times_positive(fitted_hurdle):
    """The forecasting point prediction equals the element-wise product of the two stages."""
    out = fitted_hurdle.predict("forecasting")
    binary, positive = _forecast_stage_frames(fitted_hurdle)
    expected = binary.iloc[:, 0] * positive.iloc[:, 0]
    assert out.index.equals(expected.index)
    assert np.allclose(out["pred_target"].to_numpy(), expected.to_numpy())


def test_point_forecast_is_finite_nonnegative_raw_scale(fitted_hurdle):
    """Predictions are finite, non-negative, and on a sane raw scale (not exploded/log-space)."""
    v = fitted_hurdle.predict("forecasting")["pred_target"].to_numpy()
    assert v.size > 0
    assert np.isfinite(v).all()
    assert (v >= 0).all()
    assert np.nanmax(v) < 1e6


def test_save(sample_config, sample_partitioner_dict, tmp_path, sample_dataframe):
    """The model is saved correctly to disk."""
    model = HurdleModel(sample_config, sample_partitioner_dict)
    model.fit(sample_dataframe)
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    assert save_path.exists()
