"""
Tests for the declared target-transform registry
(`views_stepshifter.infrastructure.transforms`).

Critical infrastructure (issue #52, ADR-003): proves the transforms are correctly
registered, exactly reversible (round-trip), strictly monotonic, and zero-preserving
— the mathematical guarantees the gate and the model rely on. Green-team section first;
gate-integration and model-wiring tests are added in later TDD slices.
"""
import numpy as np
import pytest

from views_stepshifter.infrastructure.transforms import TRANSFORMS

ALL = ["identity", "log1p", "asinh"]
# Non-negative, zero-inclusive, heavy-tailed sample (the conflict-count domain).
X = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1.0e4])


# -----------------------------------------------------------------------
# Green Team — registry structure
# -----------------------------------------------------------------------

def test_registry_has_exactly_the_expected_keys():
    assert set(TRANSFORMS.keys()) == set(ALL)


def test_registry_values_are_callable_forward_inverse_pairs():
    for name, pair in TRANSFORMS.items():
        assert isinstance(pair, tuple) and len(pair) == 2, f"{name} is not a 2-tuple"
        forward, inverse = pair
        assert callable(forward), f"{name} forward is not callable"
        assert callable(inverse), f"{name} inverse is not callable"


# -----------------------------------------------------------------------
# Green Team — mathematical guarantees (the contract the model relies on)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_round_trip_inverse_of_forward_recovers_input(name):
    forward, inverse = TRANSFORMS[name]
    np.testing.assert_allclose(inverse(forward(X)), X, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", ALL)
def test_zero_preserving(name):
    """forward(0)=0 and inverse(0)=0 — required so the missing-combo zero-fill and
    HurdleModel's (x>0) binarization stay consistent under the transform."""
    forward, inverse = TRANSFORMS[name]
    assert forward(0.0) == 0.0
    assert inverse(0.0) == 0.0


@pytest.mark.parametrize("name", ALL)
def test_strictly_monotonic_increasing(name):
    """Monotonic so forward(x) > 0 <=> x > 0 (preserves ordering and binarization)."""
    forward, _ = TRANSFORMS[name]
    assert np.all(np.diff(forward(X)) > 0)


def test_identity_is_a_true_noop():
    forward, inverse = TRANSFORMS["identity"]
    np.testing.assert_array_equal(forward(X), X)
    np.testing.assert_array_equal(inverse(X), X)


# -----------------------------------------------------------------------
# Beige Team — ReproducibilityGate integration (target_transform is a
# required CORE_GENOME key, validated against the registry)
# -----------------------------------------------------------------------

from views_stepshifter.infrastructure.exceptions import (  # noqa: E402
    MissingHyperparameterError,
)
from views_stepshifter.infrastructure.reproducibility_gate import (  # noqa: E402
    ReproducibilityGate,
)

_audit = ReproducibilityGate.Config.audit_manifest


def _xgb_config(**over):
    cfg = {
        "algorithm": "XGBRegressor",
        "steps": [1, 2, 3],
        "time_steps": 3,
        "parameters": {"n_estimators": 100, "n_jobs": 4},
        "target_transform": "log1p",
    }
    cfg.update(over)
    return cfg


def _hurdle_config(**over):
    cfg = {
        "algorithm": "HurdleModel",
        "steps": [1, 2, 3],
        "time_steps": 3,
        "parameters": {"clf": {"n_estimators": 2}, "reg": {"n_estimators": 2}},
        "target_transform": "identity",
    }
    cfg.update(over)
    return cfg


def _shurf_config(**over):
    cfg = {
        "algorithm": "ShurfModel",
        "steps": [1, 2, 3],
        "time_steps": 3,
        "submodels_to_train": 10,
        "pred_samples": 10,
        "log_target": False,
        "draw_dist": "Lognormal",
        "draw_sigma": 0.6,
        "parameters": {"clf": {"n_estimators": 2}, "reg": {"n_estimators": 2}},
        "target_transform": "identity",
    }
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("tname", ALL)
def test_gate_accepts_each_registered_transform(tname):
    _audit(_xgb_config(target_transform=tname))


def test_gate_rejects_unknown_transform():
    with pytest.raises(MissingHyperparameterError, match="target_transform"):
        _audit(_xgb_config(target_transform="frobnicate"))


def test_gate_rejects_none_transform():
    with pytest.raises(MissingHyperparameterError, match="None"):
        _audit(_xgb_config(target_transform=None))


def test_gate_rejects_missing_transform():
    cfg = _xgb_config()
    del cfg["target_transform"]
    with pytest.raises(MissingHyperparameterError, match="target_transform"):
        _audit(cfg)


# Deferred-models rule (D-26): HurdleModel/ShurfModel must be identity only.

def test_gate_accepts_hurdle_with_identity():
    _audit(_hurdle_config())


def test_gate_accepts_shurf_with_identity():
    _audit(_shurf_config())


@pytest.mark.parametrize("tname", ["log1p", "asinh"])
def test_gate_rejects_hurdle_with_nonidentity(tname):
    with pytest.raises(MissingHyperparameterError, match="identity"):
        _audit(_hurdle_config(target_transform=tname))


@pytest.mark.parametrize("tname", ["log1p", "asinh"])
def test_gate_rejects_shurf_with_nonidentity(tname):
    with pytest.raises(MissingHyperparameterError, match="identity"):
        _audit(_shurf_config(target_transform=tname))


# -----------------------------------------------------------------------
# Beige Team — StepshifterModel wiring (forward in _process_data once,
# inverse at the predict() output boundary once; name serialized; pickle-safe)
# -----------------------------------------------------------------------

import pickle  # noqa: E402

import pandas as pd  # noqa: E402

from views_stepshifter.models.stepshifter import StepshifterModel  # noqa: E402

_PARTS = {"train": [0, 1], "test": [2, 3]}


def _model(transform="log1p"):
    cfg = {
        "steps": [1, 2],
        "targets": ["t"],
        "parameters": {"n_estimators": 5},
        "sweep": False,
        "target_transform": transform,
    }
    return StepshifterModel(cfg, _PARTS)


def _df():
    # Every country present in every month -> _process_data does no filtering/zero-fill,
    # so we can read the forward transform off the target directly.
    idx = pd.MultiIndex.from_product([[0, 1, 2], [0, 1]], names=["month_id", "country_id"])
    return pd.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "f": np.arange(6, dtype=float)}, index=idx
    )


def test_model_init_stores_transform_name():
    assert _model("log1p")._target_transform_name == "log1p"


def test_model_forward_inverse_resolve_from_registry():
    m = _model("log1p")
    x = np.array([0.0, 1.0, 10.0])
    np.testing.assert_allclose(m._forward(x), np.log1p(x))
    np.testing.assert_allclose(m._inverse(x), np.expm1(x))


def test_model_init_unknown_transform_raises():
    with pytest.raises(ValueError, match="target_transform"):
        _model("frobnicate")


def test_model_init_missing_transform_raises():
    cfg = {"steps": [1, 2], "targets": ["t"], "parameters": {}, "sweep": False}
    with pytest.raises(KeyError):
        StepshifterModel(cfg, _PARTS)


def test_process_data_applies_forward_to_target_once():
    m = _model("log1p")
    out = m._process_data(_df())
    np.testing.assert_allclose(
        out["t"].to_numpy(), np.log1p(np.array([0.0, 1, 2, 3, 4, 5]))
    )
    # features untouched
    np.testing.assert_array_equal(out["f"].to_numpy(), np.arange(6, dtype=float))


def test_process_data_identity_leaves_target_unchanged():
    m = _model("identity")
    out = m._process_data(_df())
    np.testing.assert_array_equal(out["t"].to_numpy(), np.array([0.0, 1, 2, 3, 4, 5]))


def test_inverse_transform_predictions_single_frame():
    m = _model("log1p")
    preds = pd.DataFrame({"pred_t": np.log1p(np.array([0.0, 5.0, 100.0]))})
    out = m._inverse_transform_predictions(preds)
    np.testing.assert_allclose(out["pred_t"].to_numpy(), [0.0, 5.0, 100.0])


def test_inverse_transform_predictions_list_of_frames():
    m = _model("log1p")
    frames = [
        pd.DataFrame({"pred_t": np.log1p(np.array([0.0, 5.0]))}),
        pd.DataFrame({"pred_t": np.log1p(np.array([100.0]))}),
    ]
    out = m._inverse_transform_predictions(frames)
    np.testing.assert_allclose(out[0]["pred_t"].to_numpy(), [0.0, 5.0])
    np.testing.assert_allclose(out[1]["pred_t"].to_numpy(), [100.0])


def test_inverse_transform_predictions_identity_is_noop():
    m = _model("identity")
    preds = pd.DataFrame({"pred_t": np.array([0.0, 5.0, 100.0])})
    out = m._inverse_transform_predictions(preds)
    np.testing.assert_array_equal(out["pred_t"].to_numpy(), [0.0, 5.0, 100.0])


def test_model_is_picklable_and_preserves_transform_name():
    m = _model("log1p")
    restored = pickle.loads(pickle.dumps(m))
    assert restored._target_transform_name == "log1p"
    np.testing.assert_allclose(restored._inverse(np.array([0.0, 1.0])), np.expm1([0.0, 1.0]))


def test_predict_routes_through_inverse_exactly_once_end_to_end():
    """A real fit -> predict('forecasting') applies the inverse exactly once at the
    output boundary (proves the inverse is wired into predict(), not only the helper)."""
    idx = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    df = pd.DataFrame(
        {"f1": np.arange(63, dtype=float), "t": np.array([0.0, 1.0, 2.0] * 21)},
        index=idx,
    )
    cfg = {
        "steps": [1, 2],
        "targets": ["t"],
        "parameters": {"n_estimators": 5, "n_jobs": 1},
        "sweep": False,
        "target_transform": "log1p",
        "model_reg": "LGBMRegressor",
    }
    m = StepshifterModel(cfg, {"train": [0, 10], "test": [11, 20]})
    m.fit(df)

    seen = {"n": 0}
    real = m._inverse_transform_predictions

    def _spy(preds):
        seen["n"] += 1
        return real(preds)

    m._inverse_transform_predictions = _spy
    out = m.predict("forecasting")
    assert seen["n"] == 1
    assert np.isfinite(out["pred_t"].to_numpy()).all()


# -----------------------------------------------------------------------
# Red Team — defense-in-depth: the deferred two-stage models must reject a
# non-identity transform at construction (not only at the gate), because their
# overridden predict() applies no inverse -> a non-identity transform would
# silently emit transformed-space predictions.
# -----------------------------------------------------------------------

from views_stepshifter.models.hurdle_model import HurdleModel  # noqa: E402
from views_stepshifter.models.shurf_model import ShurfModel  # noqa: E402


def _hurdle_model_cfg(transform="identity"):
    return {
        "steps": [1, 2],
        "targets": ["t"],
        "parameters": {"clf": {}, "reg": {}},
        "sweep": False,
        "target_transform": transform,
    }


def _shurf_model_cfg(transform="identity"):
    cfg = _hurdle_model_cfg(transform)
    cfg.update(
        submodels_to_train=2,
        pred_samples=2,
        log_target=False,
        draw_dist="Lognormal",
        draw_sigma=0.6,
    )
    return cfg


def test_hurdle_model_accepts_identity():
    HurdleModel(_hurdle_model_cfg("identity"), _PARTS)


def test_shurf_model_accepts_identity():
    ShurfModel(_shurf_model_cfg("identity"), _PARTS)


@pytest.mark.parametrize("t", ["log1p", "asinh"])
def test_hurdle_model_rejects_nonidentity_transform(t):
    with pytest.raises(ValueError, match="identity"):
        HurdleModel(_hurdle_model_cfg(t), _PARTS)


@pytest.mark.parametrize("t", ["log1p", "asinh"])
def test_shurf_model_rejects_nonidentity_transform(t):
    with pytest.raises(ValueError, match="identity"):
        ShurfModel(_shurf_model_cfg(t), _PARTS)


def test_hurdle_identity_fit_predict_produces_sane_raw_output():
    """Real (non-mocked) end-to-end robustness lock for the deferred Hurdle path:
    a HurdleModel with identity fits and predicts, and the binary x positive
    composition yields finite, sanely-scaled raw output. Complements the existing
    mocked fit/predict tests (which are GPU-branch brittle, D-13)."""
    idx = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    df = pd.DataFrame(
        {"f1": np.arange(63, dtype=float), "t": np.array([0.0, 1.0, 5.0] * 21)},
        index=idx,
    )
    cfg = {
        "steps": [1, 2],
        "targets": ["t"],
        "parameters": {"clf": {"n_estimators": 5}, "reg": {"n_estimators": 5}},
        "sweep": False,
        "target_transform": "identity",
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor",
    }
    m = HurdleModel(cfg, {"train": [0, 10], "test": [11, 20]})
    m.fit(df)
    out = m.predict("forecasting")
    col = "pred_t"
    vals = out[col].to_numpy()
    assert len(vals) > 0
    assert np.isfinite(vals).all()
    assert np.nanmax(np.abs(vals)) < 1e6  # sane raw scale, not exploded/log-space


# -----------------------------------------------------------------------
# Red Team — ShurfModel log_target=True is broken (D-27): it is identity-pinned
# (raw target), so the log_target=True sampler applies expm1 to a raw prediction
# -> float64 overflow -> inf. The gate must reject it (use log_target=False;
# re-enabling a safe log-space path is deferred, #71).
# -----------------------------------------------------------------------

def test_gate_rejects_shurf_with_log_target_true():
    with pytest.raises(MissingHyperparameterError, match="log_target"):
        _audit(_shurf_config(log_target=True))


def test_gate_accepts_shurf_with_log_target_false():
    _audit(_shurf_config(log_target=False))


def test_shurf_log_target_false_predict_is_finite_no_overflow():
    """End-to-end: a log_target=False Shurf produces FINITE raw-space samples (no
    expm1 overflow) and guards the dead-code removal in predict_sequence. The broken
    log_target=True path is gate-rejected (tested separately)."""
    idx = pd.MultiIndex.from_product(
        [range(21), range(3)], names=["month_id", "country_id"]
    )
    df = pd.DataFrame(
        {"f1": np.arange(63, dtype=float), "t": np.array([0.0, 1.0, 5.0] * 21)},
        index=idx,
    )
    cfg = {
        "steps": [1, 2],
        "targets": ["t"],
        "parameters": {"clf": {"n_estimators": 2}, "reg": {"n_estimators": 2}},
        "sweep": False,
        "target_transform": "identity",
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor",
        "submodels_to_train": 1,
        "pred_samples": 2,
        "log_target": False,
        "draw_dist": "Lognormal",
        "draw_sigma": 0.5,
    }
    m = ShurfModel(cfg, {"train": [0, 10], "test": [11, 20]})
    m.fit(df)
    out = m.predict("forecasting")
    vals = np.concatenate(
        [np.asarray(v, dtype=float).ravel() for v in out["pred_t"]]
    )
    assert vals.size > 0
    assert np.isfinite(vals).all()   # the point: no inf from expm1 overflow
    assert np.nanmax(vals) < 1e6     # sane raw scale, not exploded
