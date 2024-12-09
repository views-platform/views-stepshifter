import pytest
import pandas as pd
import numpy as np
from views_stepshifter.models.validation import dataframe_is_right_format, views_validate

@pytest.fixture
def valid_dataframe():
    """
    Fixture for a valid dataframe with a two-level index.
    
    Returns:
        pd.DataFrame: A dataframe with 'month_id' and 'country_id' as index levels.
    """
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'country_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_index_dataframe():
    """
    Fixture for an invalid dataframe with a single-level index.
    
    Returns:
        pd.DataFrame: A dataframe with a single-level index.
    """
    index = pd.Index([1, 2], name='month_id')
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_first_level_index_dataframe():
    """
    Fixture for an invalid dataframe with an incorrect first-level index.
    
    Returns:
        pd.DataFrame: A dataframe with 'wrong_id' and 'country_id' as index levels.
    """
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['wrong_id', 'country_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_second_level_index_dataframe():
    """
    Fixture for an invalid dataframe with an incorrect second-level index.
    
    Returns:
        pd.DataFrame: A dataframe with 'month_id' and 'wrong_id' as index levels.
    """
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'wrong_id'])
    data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)
    return data

@pytest.fixture
def invalid_dtype_dataframe():
    """
    Fixture for an invalid dataframe with incorrect data types.
    
    Returns:
        pd.DataFrame: A dataframe with integer values instead of floats.
    """
    index = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['month_id', 'country_id'])
    data = pd.DataFrame({'value': [1, 2]}, index=index)
    return data

def test_dataframe_is_right_format_valid(valid_dataframe):
    """
    Test that a valid dataframe passes the format validation.
    
    Args:
        valid_dataframe (pd.DataFrame): A valid dataframe fixture.
    """
    dataframe_is_right_format(valid_dataframe)

def test_dataframe_is_right_format_invalid_index(invalid_index_dataframe):
    """
    Test that a dataframe with a single-level index raises an AssertionError.
    
    Args:
        invalid_index_dataframe (pd.DataFrame): An invalid dataframe fixture.
    """
    with pytest.raises(AssertionError, match="Dataframe must have a two-level index"):
        dataframe_is_right_format(invalid_index_dataframe)

def test_dataframe_is_right_format_invalid_first_level_index(invalid_first_level_index_dataframe):
    """
    Test that a dataframe with an incorrect first-level index raises an AssertionError.
    
    Args:
        invalid_first_level_index_dataframe (pd.DataFrame): An invalid dataframe fixture.
    """
    with pytest.raises(AssertionError, match="The first level of the index must be 'month_id'"):
        dataframe_is_right_format(invalid_first_level_index_dataframe)

def test_dataframe_is_right_format_invalid_second_level_index(invalid_second_level_index_dataframe):
    """
    Test that a dataframe with an incorrect second-level index raises an AssertionError.
    
    Args:
        invalid_second_level_index_dataframe (pd.DataFrame): An invalid dataframe fixture.
    """
    with pytest.raises(AssertionError, match="The second level of the index must be 'country_id' or 'priogrid_gid'"):
        dataframe_is_right_format(invalid_second_level_index_dataframe)

def test_dataframe_is_right_format_invalid_dtype(invalid_dtype_dataframe):
    """
    Test that a dataframe with incorrect data types raises an AssertionError.
    
    Args:
        invalid_dtype_dataframe (pd.DataFrame): An invalid dataframe fixture.
    """
    with pytest.raises(AssertionError, match="The dataframe must contain only np.float64 floats"):
        dataframe_is_right_format(invalid_dtype_dataframe)

@views_validate
def dummy_function(dataframe):
    """
    Dummy function to test the views_validate decorator.
    
    Args:
        dataframe (pd.DataFrame): A dataframe to be validated.
    
    Returns:
        str: A success message if validation passes.
    """
    return "Function executed"

def test_views_validate_decorator(valid_dataframe):
    """
    Test that the views_validate decorator allows a valid dataframe.
    
    Args:
        valid_dataframe (pd.DataFrame): A valid dataframe fixture.
    """
    result = dummy_function(valid_dataframe)
    assert result == "Function executed"

def test_views_validate_decorator_invalid_dataframe(invalid_index_dataframe):
    """
    Test that the views_validate decorator raises an AssertionError for an invalid dataframe.
    
    Args:
        invalid_index_dataframe (pd.DataFrame): An invalid dataframe fixture.
    """
    with pytest.raises(AssertionError, match="Dataframe must have a two-level index"):
        dummy_function(invalid_index_dataframe)