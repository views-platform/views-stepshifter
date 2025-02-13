import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from views_stepshifter.models.hurdle_model import HurdleModel


@pytest.fixture
def sample_config():
    return {
        "steps": [1, 2, 3],
        "depvar": "target",
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
    assert model._depvar == sample_config["depvar"]
    assert model._clf_params == sample_config["parameters"]["clf"]
    assert model._reg_params == sample_config["parameters"]["reg"]


def test_fit(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the fit method of the HurdleModel.
    Ensure that the data is processed correctly and the models are fitted.
    """
    model = HurdleModel(sample_config, sample_partitioner_dict)
    model._clf = MagicMock()
    model._reg = MagicMock()
    model.fit(sample_dataframe)

    assert model._clf().fit.called
    assert model._reg().fit.called


def test_predict(sample_config, sample_partitioner_dict, sample_dataframe):
    """
    Test the predict method of the HurdleModel.
    Ensure that predictions are made correctly for both stages.
    """
    model = HurdleModel(sample_config, sample_partitioner_dict)
    model._resolve_estimator = MagicMock(return_value=MagicMock())
    model.fit(sample_dataframe)
    predictions = model.predict(sample_dataframe, run_type="forecasting")
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
    model._resolve_estimator = MagicMock(return_value=MagicMock())
    model.fit(sample_dataframe)
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    assert save_path.exists()
