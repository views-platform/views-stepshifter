import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
from views_stepshifter.models.hurdle_model import HurdleModel


@pytest.fixture
def sample_config():
    return {
        "steps": [1, 2, 3],
        "depvar": ["target"],
        "model_clf": "RandomForestClassifier",
        "model_reg": "RandomForestRegressor",
        "parameters": {"clf": {"n_estimators": 100, "max_depth": 10}, "reg": {}},
        "sweep": False,
        "metrics": ["test_metric"]
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
        "target": np.array(
            [0, 1, 2] * 21, dtype=np.float64
        ),  # Ensure some values are above the threshold
    }
    return pd.DataFrame(data, index=index)


def test_initialization(sample_config, sample_partitioner_dict):
    """
    Test the initialization of the HurdleModel.
    Ensure that the model initializes with the correct attributes.
    """
    model = HurdleModel(sample_config, sample_partitioner_dict)
    assert model._steps == sample_config["steps"]
    assert model._depvar == sample_config["depvar"][0]
    assert model._clf_params == sample_config["parameters"]["clf"]
    assert model._reg_params == sample_config["parameters"]["reg"]


def test_fit(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the fit method of the HurdleModel.
    Ensure that the data is processed correctly and the models are fitted.
    """
    with patch("views_stepshifter.models.hurdle_model.HurdleModel._resolve_clf_model") as mock_resolve_clf_model, \
        patch("views_stepshifter.models.hurdle_model.HurdleModel._resolve_reg_model") as mock_resolve_reg_model, \
        patch("views_stepshifter.models.hurdle_model.RegressionModel") as mock_RegressionModel:
                
        model = HurdleModel(sample_config, sample_partitioner_dict)
        model.fit(sample_dataframe)
        assert model._clf == mock_resolve_clf_model(model._config["model_clf"])
        assert model._reg == mock_resolve_reg_model(model._config["model_reg"])
        assert mock_RegressionModel.call_count == len(model._steps) * 2
        assert mock_RegressionModel(lags_past_covariates=[-1], model=model._clf).fit.call_count == len(model._steps) * 2
        
        target_binary = [
            s.map(lambda x: (x > 0).astype(float)) for s in model._target_train
        ]
        target_pos, past_cov_pos = zip(
            *[
                (t, p)
                for t, p in zip(model._target_train, model._past_cov)
                if (t.values() > 0).any()
            ]
        )
        for step in model._steps:
            mock_RegressionModel.assert_has_calls([
                call(lags_past_covariates=[-step], model=model._clf),
                call(lags_past_covariates=[-step], model=model._reg),
            ], any_order=True)

            mock_RegressionModel(lags_past_covariates=[-step], model=model._clf).fit.assert_any_call(target_binary, past_covariates=model._past_cov)
            mock_RegressionModel(lags_past_covariates=[-step], model=model._reg).fit.assert_any_call(target_pos, past_covariates=past_cov_pos)
    
def test_predict(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the predict method of the HurdleModel.
    Ensure that predictions are made correctly for both stages.
    """
    model = HurdleModel(sample_config, sample_partitioner_dict)
    model._resolve_estimator = MagicMock(return_value=MagicMock())
    model.fit(sample_dataframe)
    predictions = model.predict(run_type="forecasting")
    assert not predictions.empty


# Idk about this one

# def test_threshold(sample_config, sample_partitioner_dict, sample_dataframe):
#     """
#     Test the threshold handling in the HurdleModel.
#     Ensure that the threshold is applied correctly in the binary stage.
#     """
#     sample_config["threshold"] = 0.5
#     model = HurdleModel(sample_config, sample_partitioner_dict)
#     model._resolve_estimator = MagicMock(return_value=MagicMock())
#     model.fit(sample_dataframe)
#     predictions = model.predict(sample_dataframe, run_type="forecasting")
    
#     # Ensure that the threshold is applied correctly
#     print(predictions)
#     binary_predictions = predictions["step_combined"] > 0.5
#     print(binary_predictions)
#     assert binary_predictions.all(), f"Some predictions are not above the threshold: {binary_predictions}"


def test_save(sample_config, sample_partitioner_dict, tmp_path, sample_dataframe):
    """
    Test the save method of the HurdleModel.
    Ensure that the model is saved correctly and can be loaded.
    """
    model = HurdleModel(sample_config, sample_partitioner_dict)
    model.fit(sample_dataframe)
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    assert save_path.exists()
