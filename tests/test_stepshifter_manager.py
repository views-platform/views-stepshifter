import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from views_stepshifter.manager.stepshifter_manager import StepshifterManager
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.cli.args import ForecastingModelArgs


@pytest.fixture
def mock_model_path():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_dir = "/path/to/models/test_model"
    mock_path.target = "model"
    mock_path.artifacts = Path("/path/to/artifacts")
    mock_path.get_latest_model_artifact_path.return_value = Path("test_model_202401011_200000")
    mock_path.logging = MagicMock()
    mock_path.models = Path("/path/to/models_root") 
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/path/to/data_raw")
    return mock_path

@pytest.fixture
def mock_config_meta():
    return {
        "name": "test_model",
        "algorithm": "LGBMRegressor",
        "targets": "test_target",
        "metrics": ["test_metric"]
    }

@pytest.fixture
def mock_config_meta_hurdle():
    return {
        "name": "test_model",
        "algorithm": "HurdleModel",
        "targets": "test_target",
        "metrics": ["test_metric"],
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor"
    }

@pytest.fixture
def mock_config_deployment():
    return {
        "deployment_status": "status"
    }

@pytest.fixture
def mock_config_hyperparameters():
    return {
        "steps": [1, 2, 3],
        "time_steps": 3,
        "parameters": {
            "n_estimators": 100,
            "n_jobs": 4,
        }
    }

@pytest.fixture
def mock_config_hyperparameters_hurdle():
    return {
        "steps": [1, 2, 3],
        "time_steps": 3,
        "parameters": {
            "clf": {
                "n_estimators": 100,
            },
            "reg": {
                "n_estimators": 100,
            }
        }
    }

@pytest.fixture
def mock_config_sweep():
    return {
        "parameters": {
            "param": {"value": [1, 2, 3]},
        }
    }

@pytest.fixture
def mock_partitioner_dict():
    return {
        "train": [0, 10],
        "test": [11, 20]
    }

@pytest.fixture
def stepshifter_manager(mock_model_path, mock_config_meta, mock_config_deployment, mock_config_hyperparameters, mock_config_sweep, mock_partitioner_dict):
    """
    Provides a StepshifterManager instance for a non-hurdle model.

    It patches _ModelManager__load_config to inject mock config dictionaries.
    """
    with patch.object(StepshifterManager, '_ModelManager__load_config', side_effect=lambda file, func: {
        "config_meta.py": mock_config_meta,
        "config_deployment.py": mock_config_deployment,
        "config_hyperparameters.py": mock_config_hyperparameters,
        "config_sweep.py": mock_config_sweep
    }.get(file, None)):

        manager = StepshifterManager(mock_model_path, use_prediction_store=False)

        manager._data_loader = MagicMock()
        manager._data_loader.partition_dict = mock_partitioner_dict
        manager._cached_data_path = Path("dummy_cached_df.parquet")

        yield manager

@pytest.fixture
def stepshifter_manager_hurdle(mock_model_path, mock_config_meta_hurdle, mock_config_deployment, mock_config_hyperparameters_hurdle, mock_config_sweep, mock_partitioner_dict):
    """
    Provides a StepshifterManager instance for a hurdle model.

    It patches _ModelManager__load_config to inject mock config dictionaries.
    """
    with patch.object(StepshifterManager, '_ModelManager__load_config', side_effect=lambda file, func: {
        "config_meta.py": mock_config_meta_hurdle,
        "config_deployment.py": mock_config_deployment,
        "config_hyperparameters.py": mock_config_hyperparameters_hurdle,
        "config_sweep.py": mock_config_sweep
    }.get(file, None)):

        manager = StepshifterManager(mock_model_path, use_prediction_store=False)

        manager._data_loader = MagicMock()
        manager._data_loader.partition_dict = mock_partitioner_dict
        manager._cached_data_path = Path("dummy_cached_df.parquet")

        yield manager

def test_stepshifter_manager_init_hurdle(stepshifter_manager_hurdle):
    """
    Test initialization of StepshifterManager with HurdleModel.
    """
    assert stepshifter_manager_hurdle._is_hurdle is True

def test_stepshifter_manager_init_non_hurdle(stepshifter_manager):
    """
    Test initialization of StepshifterManager with non-HurdleModel.
    """
    assert stepshifter_manager._is_hurdle is False

def test_get_standardized_df():
    """
    Test the _get_standardized_df method to ensure it correctly standardizes the DataFrame.
    """
    df1 = pd.DataFrame({
        "a": [1.0, -1.0, np.inf, -np.inf, 3.0],
        "b": [4.0, 5.0, -6.0, 7.0, -8.0]
    })
    expected_df1 = pd.DataFrame({
        "a": [1.0, 0.0, 0.0, 0.0, 3.0],
        "b": [4.0, 5.0, 0.0, 7.0, 0.0]
    })
    df2 = pd.DataFrame({
        "a": [[1.0, -1.0, np.inf],
               [-np.inf, 3.0, 4.0]],
        "b": [[4.0, 5.0, -6.0],
              [7.0, -8.0, 9.0]],
    })
    expected_df2 = pd.DataFrame({
        "a": [[1.0, 0.0, 0.0],
               [0.0, 3.0, 4.0]],
        "b": [[4.0, 5.0, 0.0],
              [7.0, 0.0, 9.0]],
    })
    result_df1 = StepshifterManager._get_standardized_df(df1)
    result_df2 = StepshifterManager._get_standardized_df(df2)
    pd.testing.assert_frame_equal(result_df1, expected_df1)
    pd.testing.assert_frame_equal(result_df2, expected_df2)

def test_split_hurdle_parameters(stepshifter_manager_hurdle):
    """
    Test the _split_hurdle_parameters method to ensure it correctly splits the parameters for HurdleModel.
    """
    stepshifter_manager_hurdle.configs = {
        "algorithm": "HurdleModel",
        "clf_param1": "value1",
        "clf_param2": "value2",
        "reg_param1": "value3",
        "reg_param2": "value4"
    }
    split_config = stepshifter_manager_hurdle._split_hurdle_parameters()
    assert split_config["clf"] == {"param1": "value1", "param2": "value2"}
    assert split_config["reg"] == {"param1": "value3", "param2": "value4"}

def test_get_model(stepshifter_manager, stepshifter_manager_hurdle, mock_partitioner_dict):
    """
    Test the _get_model method to ensure it returns the correct model based on the algorithm specified in the config.
    """
    with patch("views_stepshifter.manager.stepshifter_manager.HurdleModel") as mock_hurdle_model, \
        patch("views_stepshifter.manager.stepshifter_manager.StepshifterModel") as mock_stepshifter_model:

        # --- Test Hurdle ---
        args = ForecastingModelArgs(run_type="test_run_type", saved=True)
        hurdle_args = vars(args)
        hurdle_args["algorithm"] = "HurdleModel"
        stepshifter_manager_hurdle.configs = hurdle_args

        stepshifter_manager_hurdle._get_model(mock_partitioner_dict)
        mock_hurdle_model.assert_called_once_with(stepshifter_manager_hurdle.configs, mock_partitioner_dict)
        mock_stepshifter_model.assert_not_called()

        mock_hurdle_model.reset_mock()
        mock_stepshifter_model.reset_mock()

        # --- Test Non-Hurdle ---
        # Documents configs-setter merge semantics: assigning a single key
        # must add model_reg without removing steps/time_steps/parameters.
        args = ForecastingModelArgs(run_type="test_run_type", saved=True)
        non_hurdle_args = vars(args)
        non_hurdle_args["algorithm"] = "LGBMRegressor"
        non_hurdle_args["steps"] = [1, 2, 3]
        non_hurdle_args["time_steps"] = 3
        non_hurdle_args["parameters"] = {"n_estimators": 100, "n_jobs": 4}
        stepshifter_manager.configs = non_hurdle_args

        stepshifter_manager._get_model(mock_partitioner_dict)

        assert "steps" in stepshifter_manager.configs
        assert "time_steps" in stepshifter_manager.configs
        assert "parameters" in stepshifter_manager.configs
        assert stepshifter_manager.configs["model_reg"] == "LGBMRegressor"
        mock_stepshifter_model.assert_called_once()
        mock_hurdle_model.assert_not_called()

def test_train_model_artifact(stepshifter_manager, stepshifter_manager_hurdle):
    """
    Test the _train_model_artifact method to ensure it correctly trains and saves the model artifact.
    """
    with patch("views_stepshifter.manager.stepshifter_manager.StepshifterManager._split_hurdle_parameters") as mock_split_hurdle, \
        patch("views_stepshifter.manager.stepshifter_manager.read_dataframe") as mock_read_dataframe, \
        patch("views_stepshifter.manager.stepshifter_manager.StepshifterManager._get_model") as mock_get_model:

        # --- Test Non-Hurdle ---
        args = ForecastingModelArgs(run_type="test_run_type", train=True)
        non_hurdle_args = vars(args)
        non_hurdle_args["algorithm"] = "LGBMRegressor"
        non_hurdle_args["sweep"] = False
        non_hurdle_args["steps"] = [1, 2, 3]
        non_hurdle_args["time_steps"] = 3
        non_hurdle_args["parameters"] = {"n_estimators": 100, "n_jobs": 4}
        stepshifter_manager.configs = non_hurdle_args

        stepshifter_manager._train_model_artifact()

        mock_split_hurdle.assert_not_called()
        assert stepshifter_manager.configs["run_type"] == "test_run_type"
        mock_read_dataframe.assert_called_once_with(Path("dummy_cached_df.parquet"))
        mock_get_model.assert_called_once_with(stepshifter_manager._data_loader.partition_dict)
        mock_get_model.return_value.fit.assert_called_once()
        mock_get_model.return_value.save.assert_called_once()

        mock_read_dataframe.reset_mock()
        mock_get_model.reset_mock()

        mock_split_hurdle.reset_mock()

        # --- Test Hurdle ---
        args = ForecastingModelArgs(run_type="test_run_type", train=True)
        hurdle_args = vars(args)
        hurdle_args["algorithm"] = "HurdleModel"
        hurdle_args["steps"] = [1, 2, 3]
        hurdle_args["time_steps"] = 3
        hurdle_args["parameters"] = {"clf": {"n_estimators": 100}, "reg": {"n_estimators": 100}}
        stepshifter_manager_hurdle.configs = hurdle_args
        stepshifter_manager_hurdle._is_hurdle = True

        stepshifter_manager_hurdle._train_model_artifact()

        mock_read_dataframe.assert_called_once_with(Path("dummy_cached_df.parquet"))
        mock_get_model.assert_called_once_with(stepshifter_manager_hurdle._data_loader.partition_dict)

def test_evaluate_model_artifact(stepshifter_manager):
    """
    Test the _evaluate_model_artifact method to ensure it correctly evaluates the model artifact.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ["mock_df"]
    with patch("builtins.open", mock_open(read_data=b"mocked_data")), \
        patch("pickle.load", MagicMock(return_value=mock_model)), \
        patch("views_stepshifter.manager.stepshifter_manager.logger") as mock_logger, \
        patch.object(StepshifterManager, "_get_standardized_df", return_value="standardized_df") as mock_get_standardized_df:

        
        # --- Test default artifact branch (else) ---
        args = ForecastingModelArgs(run_type="test_run_type", evaluate=True, saved=True)
        stepshifter_manager.configs = vars(args)

        eval_type = "test_eval_type"
        artifact_name = None
        stepshifter_manager._evaluate_model_artifact(eval_type, artifact_name)

        assert stepshifter_manager.configs["run_type"] == "test_run_type"
        mock_get_standardized_df.assert_called_once()
     
        mock_logger.reset_mock()

        # --- Test specific artifact branch (if) ---
        artifact_name = "non_default_artifact.pkl"
        expected_path = stepshifter_manager._model_path.artifacts / artifact_name
        
        stepshifter_manager._evaluate_model_artifact(eval_type, artifact_name)
        
        mock_logger.info.assert_called_once_with(f"Using (non-default) artifact: {artifact_name}")
        assert expected_path == Path("/path/to/artifacts/non_default_artifact.pkl")


def test_forecast_model_artifact(stepshifter_manager):
    """
    Test the _forecast_model_artifact method to ensure it correctly forecasts the model artifact.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ["mock_df"]
    with patch("builtins.open", mock_open(read_data=b"mocked_data")) as mock_builtins_open, \
        patch("pickle.load", MagicMock(return_value=mock_model)), \
        patch("views_stepshifter.manager.stepshifter_manager.logger") as mock_logger, \
        patch.object(StepshifterManager, "_get_standardized_df", return_value="standardized_df") as mock_get_standardized_df:

        
        # --- Test default artifact branch (else) ---
        args = ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True)
        stepshifter_manager.configs = vars(args)
        
        artifact_name = None
        stepshifter_manager._forecast_model_artifact(artifact_name)

        assert stepshifter_manager.configs["run_type"] == "forecasting"
        mock_model.predict.assert_called_once_with("forecasting")
        mock_get_standardized_df.assert_called_once()
     
        mock_logger.reset_mock()
        mock_model.predict.reset_mock()
        mock_get_standardized_df.reset_mock()

        # --- Test specific artifact branch (if) with FileNotFoundError ---
        mock_builtins_open.side_effect = FileNotFoundError("Test error")
        artifact_name = "non_default_artifact.pkl"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            stepshifter_manager._forecast_model_artifact(artifact_name)
        
        assert str(exc_info.value) == "Test error"
        
        mock_logger.info.assert_called_once_with(f"Using (non-default) artifact: {artifact_name}")
        path_artifact = stepshifter_manager._model_path.artifacts / artifact_name
        assert path_artifact == Path("/path/to/artifacts/non_default_artifact.pkl")
        mock_logger.exception.assert_called_once_with(f"Model artifact not found at {path_artifact}")

def test_evaluate_sweep(stepshifter_manager):
    """
    Test the _evaluate_sweep method.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = ["mock_df"]
    with patch("views_stepshifter.manager.stepshifter_manager.read_dataframe"), \
        patch.object(StepshifterManager, "_get_standardized_df", return_value="standardized_df") as mock_get_standardized_df:
        
        args = ForecastingModelArgs(run_type="test_run_type", evaluate=True, saved=True)
        stepshifter_manager.configs = vars(args)

        eval_type = "test_eval_type"
        stepshifter_manager._evaluate_sweep(eval_type, mock_model)

        assert stepshifter_manager.configs["run_type"] == "test_run_type"
        mock_model.predict.assert_called_once_with("test_run_type", eval_type)
        mock_get_standardized_df.assert_called_once()
