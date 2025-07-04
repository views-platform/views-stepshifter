import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call, Mock
from views_stepshifter.models.hurdle_model import HurdleModel


@pytest.fixture
def sample_config():
    return {
        "steps": [1, 2, 3],
        "targets": ["target"],
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
    assert model._targets == sample_config["targets"][0]
    assert model._clf_params == sample_config["parameters"]["clf"]
    assert model._reg_params == sample_config["parameters"]["reg"]


def test_fit(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the fit method of the HurdleModel.
    Ensure that the data is processed correctly and the models are fitted.
    """
    with patch("views_stepshifter.models.hurdle_model.HurdleModel._resolve_clf_model") as mock_resolve_clf_model, \
        patch("views_stepshifter.models.hurdle_model.HurdleModel._resolve_reg_model") as mock_resolve_reg_model, \
        patch("views_stepshifter.models.hurdle_model.as_completed") as mock_as_completed, \
        patch("views_stepshifter.models.hurdle_model.tqdm.tqdm") as mock_tqdm, \
        patch("views_stepshifter.models.hurdle_model.ProcessPoolExecutor") as mock_ProcessPoolExecutor:

        mock_futures = {
            MagicMock(): "mock_value"
            for i in sample_config["steps"]
        }

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = mock_futures
        mock_ProcessPoolExecutor.return_value.__enter__.return_value = mock_executor
        
        mock_as_completed.return_value = mock_futures.keys()

        mock_tqdm.side_effect = lambda x, **kwargs: x


        model = HurdleModel(sample_config, sample_partitioner_dict)
        model.fit(sample_dataframe)
        assert model._clf == mock_resolve_clf_model(model._config["model_clf"])
        assert model._reg == mock_resolve_reg_model(model._config["model_reg"])
        mock_ProcessPoolExecutor.assert_called_once()
        mock_tqdm.assert_called_once_with(mock_futures.keys(), desc="Fitting models for steps", total=len(mock_futures))
        mock_as_completed.assert_called_once_with(mock_futures.keys())
        models = {
            sample_config["steps"][i]: list(mock_futures.keys())[i].result() for i in range(len(sample_config["steps"]))
        }
        assert model._models == models
        assert model.is_fitted_ == True

        
    
        # target_binary = [
        #     s.map(lambda x: (x > 0).astype(float)) for s in model._target_train
        # ]
        # target_pos, past_cov_pos = zip(
        #     *[
        #         (t, p)
        #         for t, p in zip(model._target_train, model._past_cov)
        #         if (t.values() > 0).any()
        #     ]
        # )
        # for step in model._steps:
        #     # mock_RegressionModel.assert_has_calls([
        #     #     call(lags_past_covariates=[-step], model=model._clf),
        #     #     call(lags_past_covariates=[-step], model=model._reg),
        #     # ], any_order=True)
        #     mock_RandomForestClassifierModel.assert_has_calls([
        #         call(lags_past_covariates=[-step], model=model._clf),
        #     ], any_order=True)
        #     mock_RandomForest.assert_has_calls([
        #         call(lags_past_covariates=[-step], model=model._reg),
        #     ], any_order=True)
        #     # mock_RegressionModel(lags_past_covariates=[-step], model=model._clf).fit.assert_any_call(target_binary, past_covariates=model._past_cov)
        #     # mock_RegressionModel(lags_past_covariates=[-step], model=model._reg).fit.assert_any_call(target_pos, past_covariates=past_cov_pos)
        #     mock_RandomForestClassifierModel(lags_past_covariates=[-step], model=model._clf).fit.assert_any_call(target_binary, past_covariates=model._past_cov)
        #     mock_RandomForest(lags_past_covariates=[-step], model=model._reg).fit.assert_any_call(target_pos, past_covariates=past_cov_pos)
    
def test_predict(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the predict method of the HurdleModel.
    Ensure that predictions are made correctly for both stages.
    """
    with patch("views_stepshifter.models.hurdle_model.check_is_fitted") as mock_check_is_fitted, \
        patch("views_stepshifter.models.hurdle_model.as_completed") as mock_as_completed, \
        patch("views_stepshifter.models.hurdle_model.tqdm.tqdm") as mock_tqdm, \
        patch("views_stepshifter.models.hurdle_model.ProcessPoolExecutor") as mock_ProcessPoolExecutor, \
        patch("views_stepshifter.models.hurdle_model.ForecastingModelManager._resolve_evaluation_sequence_number") as mock_sequence_number:

        
        # the else branch
        future = MagicMock()
        future.result.return_value = pd.DataFrame(
            np.random.rand(5, 1),  
            index=pd.Index(range(5, 10)) 
        )

        mock_futures = {
            MagicMock(): future
            for i in range(len(sample_config["steps"])*2) 
        }

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = mock_futures 
        mock_ProcessPoolExecutor.return_value.__enter__.return_value = mock_executor

        mock_as_completed.return_value = mock_futures.values()

        mock_tqdm.side_effect = lambda x, **kwargs: x


        model = HurdleModel(sample_config, sample_partitioner_dict)
        model._models = {
            sample_config["steps"][i]: list(mock_futures.keys())[i].result() for i in range(len(sample_config["steps"]))
        }
        predictions = model.predict(run_type="forecasting") 

        assert mock_check_is_fitted.call_count == 1
        assert mock_ProcessPoolExecutor.call_count == 1
        mock_tqdm_expected_calls = [call(mock_futures.values(), desc='Predicting binary outcomes', total=len(sample_config['steps'])), call(mock_futures.values(), desc='Predicting positive outcomes', total=len(sample_config['steps']))]
        for i in range(2):
            assert repr(mock_tqdm.call_args_list[i]) == repr(mock_tqdm_expected_calls[i])
        assert mock_as_completed.call_count == 2
        assert not predictions.empty


        # reset mocks
        mock_check_is_fitted.reset_mock()
        mock_ProcessPoolExecutor.reset_mock()
        mock_tqdm.reset_mock()
        mock_as_completed.reset_mock()


        # the if branch
        mock_sequence_number.return_value = 12
        mock_futures2 = {
            MagicMock(): "mock_value"
            for i in range(mock_sequence_number.return_value)
        }
        mock_executor = MagicMock()
        mock_executor.submit.side_effect = mock_futures2 
        mock_ProcessPoolExecutor.return_value.__enter__.return_value = mock_executor

        mock_as_completed.return_value = mock_futures2.keys()

        
        predictions2 = model.predict(run_type="test_run") 

        assert mock_check_is_fitted.call_count == 1
        assert mock_sequence_number.call_count == 2
        assert mock_ProcessPoolExecutor.call_count == 1
        mock_tqdm.assert_called_once_with(mock_futures2.keys(), desc="Predicting for sequence number", total=len(mock_futures2))
        mock_as_completed.assert_called_once_with(mock_futures2.keys())
        assert len(predictions2) != 0



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
