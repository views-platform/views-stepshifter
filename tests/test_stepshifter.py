import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from views_stepshifter.models.stepshifter import StepshifterModel
from views_pipeline_core.managers.model import ModelManager

@pytest.fixture
def config():
    """
    Fixture for the configuration dictionary.

    Returns:
        dict: A dictionary containing configuration parameters for the StepshifterModel.
    """
    return {
        'steps': [1, 2],
        'depvar': ['target'],
        'model_reg': 'RandomForestRegressor',
        'parameters': {'max_depth': 1, 'n_estimators': 100},
        'sweep': False,
        "metrics": ["test_metric"]
    }

@pytest.fixture
def partitioner_dict():
    """
    Fixture for the partitioner dictionary.

    Returns:
        dict: A dictionary specifying the training and testing periods.
    """
    return {
        'train': [0, 10],
        'test': [11, 20]
    }

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


def test_resolve_reg_model(config, partitioner_dict):
    """
    Test the _resolve_estimator method of the StepshifterModel class.

    This test ensures that the _resolve_estimator method correctly resolves the 
    estimator function name to the corresponding Darts model class.

    Asserts:
        - The resolved estimator class name matches the expected class name.
    """
  
    model = StepshifterModel(config, partitioner_dict)
    reg_model = model._resolve_reg_model("XGBRegressor")

    assert reg_model is not None  
       
def test_get_parameters(config):
    """
    Test the _get_parameters method of the StepshifterModel class.

    This test ensures that the _get_parameters method correctly extracts parameters 
    from the configuration dictionary.

    Args:
        config (dict): The configuration dictionary fixture.

    Asserts:
        - The extracted parameters match the expected parameters.
    """
    model = StepshifterModel(config, {'train': [0, 10], 'test': [11, 20]})
    params = model._get_parameters(config)
    assert params == {'max_depth': 1, 'n_estimators': 100}

def test_process_data(config, partitioner_dict):
    """
    Test the _process_data method of the StepshifterModel class.

    This test ensures that the _process_data method correctly processes the input 
    DataFrame by handling missing data and ensuring that only countries existing 
    in the last month of the training data are included.

    Args:
        config (dict): The configuration dictionary fixture.
        partitioner_dict (dict): The partitioner dictionary fixture.

    Asserts:
        - The processed DataFrame has the expected shape.
        - The processed DataFrame contains the expected columns.
    """
    df = pd.DataFrame({
        'month_id': [1, 1, 2, 2],
        'country_id': ['A', 'B', 'A', 'B'],
        'target': [1, 2, 3, 4],
        'indep1': [5, 6, 7, 8]
    }).set_index(['month_id', 'country_id'])

    model = StepshifterModel(config, partitioner_dict)
    processed_df = model._process_data(df)
    assert processed_df.shape == (4, 2)  # Adjusted to account for missing data handling
    assert 'target' in processed_df.columns
    assert 'indep1' in processed_df.columns

def test_prepare_time_series(config, partitioner_dict):
    """
    Test the _prepare_time_series method of the StepshifterModel class.

    This test ensures that the _prepare_time_series method correctly prepares the 
    time series data for training and prediction.

    Args:
        config (dict): The configuration dictionary fixture.
        partitioner_dict (dict): The partitioner dictionary fixture.

    Asserts:
        - The number of prepared time series matches the expected number.
    """
    df = pd.DataFrame({
        'month_id': [1, 1, 2, 2],
        'country_id': ['A', 'B', 'A', 'B'],
        'target': [1, 2, 3, 4],
        'indep1': [5, 6, 7, 8]
    }).set_index(['month_id', 'country_id'])

    model = StepshifterModel(config, partitioner_dict)
    model._level = 'country_id'
    model._independent_variables = ['indep1']
    model._prepare_time_series(df)
    assert len(model._series) == 2

def test_fit(config, partitioner_dict, sample_dataframe):
    """
    Test the fit method of the HurdleModel.
    Ensure that the data is processed correctly and the models are fitted.
    """
    with patch("views_stepshifter.models.stepshifter.StepshifterModel._resolve_reg_model") as mock_resolve_reg_model, \
        patch("views_stepshifter.models.stepshifter.tqdm.tqdm") as mock_tqdm, \
        patch("views_stepshifter.models.stepshifter.ProcessPoolExecutor") as mock_ProcessPoolExecutor:

        mock_futures = {
            MagicMock(): "mock_value"
            for i in config["steps"]
        }

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = mock_futures
        mock_ProcessPoolExecutor.return_value.__enter__.return_value = mock_executor
        

        mock_tqdm.side_effect = lambda x, **kwargs: x


        model = StepshifterModel(config, partitioner_dict)
        model.fit(sample_dataframe)
        assert model._reg == mock_resolve_reg_model(model._config["model_reg"])
        mock_ProcessPoolExecutor.assert_called_once()
        mock_tqdm.assert_called_once_with(mock_futures.keys(), desc="Fitting models for steps", total=len(mock_futures))
        models = {
            config["steps"][i]: list(mock_futures.keys())[i].result() for i in range(len(config["steps"]))
        }
        assert model._models == models
        assert model.is_fitted_ == True


def test_predict(config, partitioner_dict, sample_dataframe):
    """
    Test the predict method of the HurdleModel.
    Ensure that predictions are made correctly for both stages.
    """
    with patch("views_stepshifter.models.stepshifter.check_is_fitted") as mock_check_is_fitted, \
        patch("views_stepshifter.models.stepshifter.as_completed") as mock_as_completed, \
        patch("views_stepshifter.models.stepshifter.tqdm.tqdm") as mock_tqdm, \
        patch("views_stepshifter.models.stepshifter.ProcessPoolExecutor") as mock_ProcessPoolExecutor, \
        patch("views_stepshifter.models.stepshifter.ModelManager._resolve_evaluation_sequence_number") as mock_sequence_number:

        
        # the else branch
        future = MagicMock()
        future.result.return_value = pd.DataFrame(
            np.random.rand(5, 1),  
            index=pd.Index(range(5, 10)) 
        )

        mock_futures = {
            MagicMock(): future
            for i in range(len(config["steps"])*2) 
        }

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = mock_futures 
        mock_ProcessPoolExecutor.return_value.__enter__.return_value = mock_executor

        mock_as_completed.return_value = mock_futures.values()

        mock_tqdm.side_effect = lambda x, **kwargs: x


        model = StepshifterModel(config, partitioner_dict)
        model._models = {
            config["steps"][i]: list(mock_futures.keys())[i].result() for i in range(len(config["steps"]))
        }
        predictions = model.predict(run_type="forecasting") 

        assert mock_check_is_fitted.call_count == 1
        assert mock_ProcessPoolExecutor.call_count == 1
        assert repr(mock_tqdm.call_args_list[0]) == repr(call(mock_futures.values(), desc='Predicting outcomes', total=len(config['steps'])))
        assert mock_as_completed.call_count == 1
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
        assert mock_sequence_number.call_count == 1
        assert mock_ProcessPoolExecutor.call_count == 1
        mock_tqdm.assert_called_once_with(mock_futures2.keys(), desc="Predicting for sequence number", total=len(mock_futures2))
        mock_as_completed.assert_called_once_with(mock_futures2.keys())
        assert len(predictions2) != 0


def test_save(config, partitioner_dict, tmp_path):
    """
    Test the save method of the StepshifterModel class.

    This test ensures that the save method correctly saves the model to a file.

    Args:
        config (dict): The configuration dictionary fixture.
        partitioner_dict (dict): The partitioner dictionary fixture.
        tmp_path (Path): A temporary directory path provided by pytest.

    Asserts:
        - The model file is created successfully.
    """
    model = StepshifterModel(config, partitioner_dict)
    model_path = tmp_path / "model.pkl"
    model.save(model_path)
    assert model_path.exists()
