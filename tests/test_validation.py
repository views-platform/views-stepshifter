import pytest
import pandas as pd
import numpy as np
from views_stepshifter.models.validation import dataframe_is_right_format, views_validate

# FILE: views-stepshifter/views_stepshifter/models/test_validation.py


@pytest.fixture
def valid_dataframe():
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'country_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_index_dataframe():
    index = pd.Index([1, 2], name='month_id')
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_first_level_index_dataframe():
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['wrong_id', 'country_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_second_level_index_dataframe():
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'wrong_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_dtype_dataframe():
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'country_id'])
    data = pd.DataFrame({'value': [1, 2]}, index=index)
    return data

def test_dataframe_is_right_format_valid(valid_dataframe):
    dataframe_is_right_format(valid_dataframe)

def test_dataframe_is_right_format_invalid_index(invalid_index_dataframe):
    with pytest.raises(AssertionError, match="Dataframe must have a two-level index"):
        dataframe_is_right_format(invalid_index_dataframe)

def test_dataframe_is_right_format_invalid_first_level_index(invalid_first_level_index_dataframe):
    with pytest.raises(AssertionError, match="The first level of the index must be 'month_id'"):
        dataframe_is_right_format(invalid_first_level_index_dataframe)

def test_dataframe_is_right_format_invalid_second_level_index(invalid_second_level_index_dataframe):
    with pytest.raises(AssertionError, match="The second level of the index must be 'country_id' or 'priogrid_gid'"):
        dataframe_is_right_format(invalid_second_level_index_dataframe)

def test_dataframe_is_right_format_invalid_dtype(invalid_dtype_dataframe):
    with pytest.raises(AssertionError, match="The dataframe must contain only np.float64 floats"):
        dataframe_is_right_format(invalid_dtype_dataframe)

@views_validate
def dummy_function(dataframe):
    return "Function executed"

def test_views_validate_decorator(valid_dataframe):
    result = dummy_function(valid_dataframe)
    assert result == "Function executed"

def test_views_validate_decorator_invalid_dataframe(invalid_index_dataframe):
    with pytest.raises(AssertionError, match="Dataframe must have a two-level index"):
        dummy_function(invalid_index_dataframe)