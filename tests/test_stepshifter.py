import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from views_stepshifter.models.stepshifter import StepshifterModel

@pytest.fixture
def config():
    """
    Fixture for the configuration dictionary.

    Returns:
        dict: A dictionary containing configuration parameters for the StepshifterModel.
    """
    return {
        'steps': [1, 2],
        'depvar': 'target',
        'model_reg': 'LinearRegressionModel',
        'parameters': {'param1': 1, 'param2': 2},
        'sweep': False
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

def test_resolve_estimator():
    """
    Test the _resolve_estimator method of the StepshifterModel class.

    This test ensures that the _resolve_estimator method correctly resolves the 
    estimator function name to the corresponding Darts model class.

    Asserts:
        - The resolved estimator class name matches the expected class name.
    """
    assert StepshifterModel._resolve_estimator('LinearRegressionModel').__name__ == 'LinearRegressionModel'
    assert StepshifterModel._resolve_estimator('RandomForestModel').__name__ == 'RandomForest'
    assert StepshifterModel._resolve_estimator('LightGBMModel').__name__ == 'LightGBMModel'
    assert StepshifterModel._resolve_estimator('XGBModel').__name__ == 'XGBModel'
    with pytest.raises(ValueError):
        StepshifterModel._resolve_estimator('InvalidModel')

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
    assert params == {'param1': 1, 'param2': 2}

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

def test_fit(config, partitioner_dict):
    """
    Test the fit method of the StepshifterModel class.

    This test ensures that the fit method correctly fits the model to the input 
    DataFrame.

    Args:
        config (dict): The configuration dictionary fixture.
        partitioner_dict (dict): The partitioner dictionary fixture.

    Asserts:
        - The model is fitted successfully.
    """
    df = pd.DataFrame({
        'month_id': [1, 1, 2, 2],
        'country_id': ['A', 'B', 'A', 'B'],
        'target': [1, 2, 3, 4],
        'indep1': [5, 6, 7, 8]
    }).set_index(['month_id', 'country_id'])

    # Convert columns to np.float64
    df = df.astype(np.float64)

    model = StepshifterModel(config, partitioner_dict)
    with patch.object(model, '_reg', return_value=MagicMock()) as mock_reg:
        model.fit(df)
        assert model.is_fitted_

def test_predict(config, partitioner_dict):
    """
    Test the predict method of the StepshifterModel class.

    This test ensures that the predict method correctly generates predictions
    based on the input DataFrame and run type.

    Asserts:
        - The predictions are not empty.
    """
    df = pd.DataFrame({
        'month_id': [1, 1, 2, 2],
        'country_id': ['A', 'B', 'A', 'B'],
        'target': [1, 2, 3, 4],
        'indep1': [5, 6, 7, 8]
    }).set_index(['month_id', 'country_id'])

    # Convert columns to np.float64
    df = df.astype(np.float64)

    model = StepshifterModel(config, partitioner_dict)
    with patch.object(model, '_reg', return_value=MagicMock()) as mock_reg:
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [
            MagicMock(pd_dataframe=MagicMock(return_value=pd.DataFrame({
                'step_combined': [0.1, 0.2]
            }, index=[11, 12])))
        ]
        mock_reg.return_value = mock_model_instance

        model.fit(df)
        pred = model.predict(df, 'forecasting')
        assert not pred.empty, "Predictions should not be empty"

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