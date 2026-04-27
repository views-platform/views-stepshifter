import functools
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def dataframe_is_right_format(dataframe: pd.DataFrame):
    second_level_indices = ["priogrid_gid", "country_id", "priogrid_id"]
    try:
        assert isinstance(dataframe.index, pd.MultiIndex) and len(dataframe.index.levels) == 2
    except AssertionError:
        logger.exception("Dataframe must have a two-level index")
        raise AssertionError("Dataframe must have a two-level index")

    try:
        assert dataframe.index.names[0] == "month_id"
    except AssertionError:
        logger.exception("The first level of the index must be 'month_id'")
        raise AssertionError("The first level of the index must be 'month_id'")

    try:
        assert dataframe.index.names[1] in second_level_indices
    except AssertionError:
        logger.exception("The second level of the index must be 'country_id' or 'priogrid_gid'")
        raise AssertionError("The second level of the index must be 'country_id' or 'priogrid_gid'")

    try:
        if "country_id" in dataframe.columns:
            assert set(dataframe.drop(columns=["country_id"]).dtypes) == {np.dtype(float)}
        else:
            assert set(dataframe.dtypes) == {np.dtype(float)}
    except AssertionError:
        logger.exception("The dataframe must contain only np.float64 floats")
        raise AssertionError("The dataframe must contain only np.float64 floats")
    
# Needs update
def views_validate(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        # When fit()
        #args[0] will be the instance of the StepshifterModel class (self).
        #args[1] will be the DataFrame (df).

        # When predict()
        # args[0] will be the instance of the StepshifterModel class (self).
        # args[1] will be the DataFrame (df).
        # args[2] will be the string (run_type).
        # args[3] will be the optional string (eval_type), if provided.
        dataframe = args[0] if isinstance(args[0], pd.DataFrame) else args[1] # Hardcoding this makes me uncomfortable
        dataframe_is_right_format(dataframe)
        # dataframe_is_right_format(args[1]) # is also uncomfortable
        return fn(*args, **kwargs)
    return inner