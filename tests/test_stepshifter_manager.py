import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from views_stepshifter.manager.stepshifter_manager import StepshifterManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_pipeline_core.files.utils import create_log_file

@pytest.fixture
def mock_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/test_model"
    mock_path.target = "model"
    return mock_path

@pytest.fixture
def mock_config_meta():
    return {
        "name": "test_model",
        "algorithm": "LightGBMModel",
        "depvar": "test_depvar"
    }

@pytest.fixture
def mock_config_meta_hurdle():
    return {
        "name": "test_model",
        "algorithm": "HurdleModel",
        "depvar": "test_depvar"
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
        "parameters": {
            "param1": "value1",
            "param2": "value2"
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
def stepshifter_manager(mock_model_path, mock_config_meta, mock_config_deployment, mock_config_hyperparameters, mock_config_sweep):
    with patch.object(StepshifterManager, '_ModelManager__load_config', side_effect=lambda file, func: {
        "config_meta.py": mock_config_meta,
        "config_deployment.py": mock_config_deployment,
        "config_hyperparameters.py": mock_config_hyperparameters,
        "config_sweep.py": mock_config_sweep
    }.get(file, None)):  
        manager = StepshifterManager(mock_model_path)
        return manager

@pytest.fixture
def stepshifter_manager_hurdle(mock_model_path, mock_config_meta_hurdle, mock_config_deployment, mock_config_hyperparameters, mock_config_sweep):
    with patch.object(StepshifterManager, '_ModelManager__load_config', side_effect=lambda file, func: {
        "config_meta.py": mock_config_meta_hurdle,
        "config_deployment.py": mock_config_deployment,
        "config_hyperparameters.py": mock_config_hyperparameters,
        "config_sweep.py": mock_config_sweep
    }.get(file, None)):  
        manager = StepshifterManager(mock_model_path)
        return manager

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
    df = pd.DataFrame({
        "a": [1.0, -1.0, np.inf, -np.inf, 3.0],
        "b": [4.0, 5.0, -6.0, 7.0, -8.0]
    })
    expected_df = pd.DataFrame({
        "a": [1.0, 0.0, 0.0, 0.0, 3.0],
        "b": [4.0, 5.0, 0.0, 7.0, 0.0]
    })
    result_df = StepshifterManager._get_standardized_df(df)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_update_sweep_config(stepshifter_manager):
    """
    Test the _update_sweep_config method to ensure it correctly updates the configuration for a sweep run.
    """
    args = MagicMock()
    args.run_type = "test_run_type"
    updated_config = stepshifter_manager._update_sweep_config(args)
    assert updated_config["parameters"]["run_type"]["value"] == "test_run_type"
    assert updated_config["parameters"]["sweep"]["value"] is True
    assert updated_config["parameters"]["name"]["value"] == "test_model"
    assert updated_config["parameters"]["depvar"]["value"] == "test_depvar"
    assert updated_config["parameters"]["algorithm"]["value"] == "LightGBMModel"

def test_split_hurdle_parameters(stepshifter_manager_hurdle):
    """
    Test the _split_hurdle_parameters method to ensure it correctly splits the parameters for HurdleModel.
    """
    stepshifter_manager_hurdle.config = {
        "clf_param1": "value1",
        "clf_param2": "value2",
        "reg_param1": "value3",
        "reg_param2": "value4"
    }
    split_config = stepshifter_manager_hurdle._split_hurdle_parameters()
    assert split_config["clf"] == {"param1": "value1", "param2": "value2"}
    assert split_config["reg"] == {"param1": "value3", "param2": "value4"}

def test_get_model(stepshifter_manager, mock_partitioner_dict):
    """
    Test the _get_model method to ensure it returns the correct model based on the algorithm specified in the config.
    """
    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))
    model = stepshifter_manager._get_model(mock_partitioner_dict)
    assert isinstance(model, StepshifterModel)

@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.create_log_file")
@patch("views_pipeline_core.files.utils.read_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_train_model_artifact(mock_open, mock_makedirs, mock_datetime, mock_read_log_file, mock_create_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager, mock_partitioner_dict):
    """
    Test the _train_model_artifact method to ensure it correctly trains and saves the model artifact.
    """
    mock_read_pickle.return_value = pd.DataFrame({"a": [1, 2, 3]})
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_log_file.return_value = {"Data Fetch Timestamp": "20230101_000000"}
    mock_read_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3]})

    stepshifter_manager._data_loader = MagicMock()
    stepshifter_manager._data_loader.partition_dict = mock_partitioner_dict
    
    stepshifter_manager._get_model = MagicMock()
    stepshifter_manager._get_model.return_value = MagicMock()
    stepshifter_manager._get_model.return_value.fit = MagicMock()
    stepshifter_manager._get_model.return_value.save = MagicMock()

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = "/mock/path/to/generated"
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")

    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))

    stepshifter_manager._train_model_artifact()

    mock_makedirs.return_value = None
    mock_open.return_value.__enter__.return_value = MagicMock()

    stepshifter_manager._get_model.return_value.fit.assert_called_once()
    stepshifter_manager._get_model.return_value.save.assert_called_once()

@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.create_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_train_model_artifact_sweep(mock_open, mock_makedirs, mock_datetime, mock_create_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager, mock_partitioner_dict):
    """
    Test the _train_model_artifact method to ensure it correctly trains the model artifact during a sweep run.
    """
    mock_read_pickle.return_value = pd.DataFrame({"a": [1, 2, 3]})
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3]})

    stepshifter_manager._data_loader = MagicMock()
    stepshifter_manager._data_loader.partition_dict = mock_partitioner_dict
    
    stepshifter_manager._get_model = MagicMock()
    stepshifter_manager._get_model.return_value = MagicMock()
    stepshifter_manager._get_model.return_value.fit = MagicMock()

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = Path("/mock/path/to/generated")
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")

    stepshifter_manager.config = {
        "run_type": "test_run_type",
        "sweep": True
    }

    stepshifter_manager._train_model_artifact()

    stepshifter_manager._get_model.return_value.fit.assert_called_once()
    stepshifter_manager._get_model.return_value.save.assert_not_called()
    mock_create_log_file.assert_not_called()


@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.create_log_file")
@patch("views_pipeline_core.files.utils.read_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_model_artifact_with_artifact_name(mock_open, mock_makedirs, mock_datetime, mock_read_log_file, mock_create_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager):
    """
    Test the _evaluate_model_artifact method to ensure it correctly evaluates the model artifact with a specific artifact name.
    """
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}),
        StepshifterModel(config={"steps": [1, 2, 3], "depvar": "test_depvar", "model_reg": "LightGBMModel", "parameters": {}, "sweep": False}, partitioner_dict={"train": [0, 10], "test": [11, 20]})
    ]
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_log_file.return_value = {"Data Fetch Timestamp": "20230101_000000"}
    mock_read_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3]})

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = Path("/mock/path/to/generated")
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")

    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))

    mock_makedirs.return_value = None
    mock_open.return_value.__enter__.return_value = MagicMock()

    artifact_name = "test_artifact"
    eval_type = "test_eval_type"

    mock_stepshift_model = StepshifterModel(config={"steps": [1, 2, 3], "depvar": "test_depvar", "model_reg": "LightGBMModel", "parameters": {}, "sweep": False}, partitioner_dict={"train": [0, 10], "test": [11, 20]})
    mock_stepshift_model.predict = MagicMock(return_value=pd.DataFrame({"predictions": [0.1, 0.2, 0.3]}))
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}),
        mock_stepshift_model
    ]

    stepshifter_manager._evaluate_model_artifact(eval_type, artifact_name)

    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.data_raw / "test_run_type_viewser_df.pkl")
    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.artifacts / "test_artifact.pkl")
    mock_stepshift_model.predict.assert_called_once_with(mock_read_dataframe.return_value, "test_run_type", "test_eval_type")

@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.read_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("views_stepshifter.manager.stepshifter_manager.StepshifterManager._get_latest_model_artifact")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_model_artifact_without_artifact_name(mock_open, mock_makedirs, mock_get_latest_model_artifact, mock_datetime, mock_read_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager):
    """
    Test the _evaluate_model_artifact method to ensure it correctly evaluates the model artifact without a specific artifact name.
    """
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}),  
        StepshifterModel(config={"steps": [1, 2, 3], "depvar": "test_depvar", "model_reg": "LightGBMModel", "parameters": {}, "sweep": False}, partitioner_dict={"train": [0, 10], "test": [11, 20]})
    ]
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_log_file.return_value = {"Data Fetch Timestamp": "20230101_000000"}
    mock_get_latest_model_artifact.return_value = Path("/mock/path/to/artifact.pkl")
    mock_read_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3]})

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = Path("/mock/path/to/generated")
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")
    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))

    mock_makedirs.return_value = None
    mock_open.return_value.__enter__.return_value = MagicMock()

    mock_stepshift_model = MagicMock()
    mock_stepshift_model.predict.return_value = pd.DataFrame(
        {"predictions": [0.1, 0.2, 0.3]}
    )
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}),  
        mock_stepshift_model
    ]

    stepshifter_manager._evaluate_model_artifact("test_eval_type", None)

    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.data_raw / "test_run_type_viewser_df.pkl")
    mock_read_pickle.assert_any_call(mock_get_latest_model_artifact.return_value)
    mock_stepshift_model.predict.assert_called_once_with(mock_read_dataframe.return_value, "test_run_type", "test_eval_type")

@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.create_log_file")
@patch("views_pipeline_core.files.utils.read_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("views_stepshifter.manager.stepshifter_manager.StepshifterManager._get_latest_model_artifact")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_forecast_model_artifact_with_artifact_name(mock_open, mock_makedirs, mock_get_latest_model_artifact, mock_datetime, mock_read_log_file, mock_create_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager):
    """
    Test the _forecast_model_artifact method to ensure it correctly forecasts using a specific model artifact.
    """
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}), 
        StepshifterModel(config={"steps": [1, 2, 3], "depvar": "test_depvar", "model_reg": "LightGBMModel", "parameters": {}, "sweep": False}, partitioner_dict={"train": [0, 10], "test": [11, 20]})
    ]
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_log_file.return_value = {"Data Fetch Timestamp": "20230101_000000"}
    mock_read_dataframe.return_value = pd.DataFrame({"a": [1, 2, 3]})

    mock_stepshift_model = MagicMock()
    mock_stepshift_model.predict.return_value = pd.DataFrame(
        {"predictions": [0.1, 0.2, 0.3]}
    )
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}), 
        mock_stepshift_model  
    ]

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = Path("/mock/path/to/generated")
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")

    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))

    mock_makedirs.return_value = None
    mock_open.return_value.__enter__.return_value = MagicMock()

    artifact_name = "test_artifact"

    stepshifter_manager._forecast_model_artifact(artifact_name)

    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.data_raw / "test_run_type_viewser_df.pkl")
    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.artifacts / "test_artifact.pkl")
    mock_stepshift_model.predict.assert_called_once_with(mock_read_dataframe.return_value, "test_run_type")

@patch("views_stepshifter.manager.stepshifter_manager.read_dataframe")
@patch("views_stepshifter.manager.stepshifter_manager.pd.read_pickle")
@patch("views_pipeline_core.files.utils.create_log_file")
@patch("views_pipeline_core.files.utils.read_log_file")
@patch("views_stepshifter.manager.stepshifter_manager.datetime")
@patch("views_stepshifter.manager.stepshifter_manager.StepshifterManager._get_latest_model_artifact")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_forecast_model_artifact_without_artifact_name(mock_open, mock_makedirs, mock_get_latest_model_artifact, mock_datetime, mock_read_log_file, mock_read_pickle, mock_read_dataframe, stepshifter_manager):
    """
    Test the _forecast_model_artifact method to ensure it correctly forecasts using the latest model artifact when no specific artifact name is provided.
    """
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}), 
        StepshifterModel(config={"steps": [1, 2, 3], "depvar": "test_depvar", "model_reg": "LightGBMModel", "parameters": {}, "sweep": False}, partitioner_dict={"train": [0, 10], "test": [11, 20]})
    ]
    mock_datetime.now.return_value.strftime.return_value = "20230101_000000"
    mock_read_log_file.return_value = {"Data Fetch Timestamp": "20230101_000000"}
    mock_get_latest_model_artifact.return_value = Path("/mock/path/to/artifact.pkl")

    stepshifter_manager._model_path.data_raw = Path("/mock/path/to/raw")
    stepshifter_manager._model_path.data_generated = Path("/mock/path/to/generated")
    stepshifter_manager._model_path.artifacts = Path("/mock/path/to/artifacts")
    stepshifter_manager.config = stepshifter_manager._update_single_config(MagicMock(run_type="test_run_type"))

    mock_stepshift_model = MagicMock()
    mock_stepshift_model.predict.return_value = pd.DataFrame(
        {"predictions": [0.1, 0.2, 0.3]}
    )
    mock_read_pickle.side_effect = [
        pd.DataFrame({"a": [1, 2, 3]}), 
        mock_stepshift_model  
    ]

    mock_makedirs.return_value = None
    mock_open.return_value.__enter__.return_value = MagicMock()

    stepshifter_manager._forecast_model_artifact(None)

    mock_read_pickle.assert_any_call(stepshifter_manager._model_path.data_raw / "test_run_type_viewser_df.pkl")
    mock_read_pickle.assert_any_call(mock_get_latest_model_artifact.return_value)
    mock_stepshift_model.predict.assert_called_once_with(mock_read_dataframe.return_value, "test_run_type")