from views_pipeline_core.managers.model import ModelManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
from darts.models import RegressionModel
import pandas as pd
from typing import List, Dict
import logging
import tqdm

logger = logging.getLogger(__name__)


class HurdleModel(StepshifterModel):
    """
    Hurdle model for time series forecasting. The model consists of two stages:
    1. Binary stage: Predicts whether the target variable is 0 or > 0.
    2. Positive stage: Predicts the value of the target variable when it is > 0.

    Note:
    This algorithm uses a two-step approach.

    **Step 1: Classification Stage**
    In the first step, a classification model is used with a binary target (0 or 1),
    indicating the absence or presence of violence.

    **Step 2: Regression Stage**
    In the second step, we use a regression model to predict a continuous or count value
    (e.g., the expected number of conflict fatalities) for the selected time series.
    We include the entire time series for countries or PRIO grids where the
    classification stage yielded at least one "positive" prediction,
    rather than limiting the regression to just the predicted positive values.
    """

    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]]):
        super().__init__(config, partitioner_dict)
        self._clf_params = self._get_parameters(config)["clf"]
        self._reg_params = self._get_parameters(config)["reg"]
        self._clf = self._resolve_clf_model(config["model_clf"])
        self._reg = self._resolve_reg_model(config["model_reg"])

    def _resolve_clf_model(self, func_name: str):
        """Lookup table for supported classification models"""

        match func_name:
            case "XGBClassifier":
                from xgboost import XGBClassifier

                return XGBClassifier(**self._clf_params)
            case "XGBRFClassifier":
                from xgboost import XGBRFClassifier

                return XGBRFClassifier(**self._clf_params)
            case "LGBMClassifier":
                from lightgbm import LGBMClassifier

                return LGBMClassifier(**self._clf_params)
            case "RandomForestClassifier":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**self._clf_params)
            case _:
                raise ValueError(
                    f"Model {func_name} is not a valid Darts forecasting model or is not supported now. "
                    f"Change the model in the config file."
                )

    @views_validate
    def fit(self, df: pd.DataFrame):
        df = self._process_data(df)
        self._prepare_time_series(df)

        # Binary outcome (event/no-event)
        # According to the DARTS doc, if timeseries uses a numeric type different from np.float32 or np.float64, not all functionalities may work properly.
        # So use astype(float) instead of astype(int) (we should have binary outputs 0,1 though)
        target_binary = [
            s.map(lambda x: (x > 0).astype(float)) for s in self._target_train
        ]

        # Positive outcome (for cases where target > threshold)
        target_pos, past_cov_pos = zip(
            *[
                (t, p)
                for t, p in zip(self._target_train, self._past_cov)
                if (t.values() > 0).any()
            ]
        )

        for step in tqdm.tqdm(self._steps, desc="Fitting model for step", leave=True):
            # Fit binary-like stage using a classification model, but the target is binary (0 or 1)
            # binary_model = self._clf(lags_past_covariates=[-step], **self._clf_params)
            binary_model = RegressionModel(
                lags_past_covariates=[-step], model=self._clf
            )
            binary_model.fit(target_binary, past_covariates=self._past_cov)

            # Fit positive stage using the regression model
            # positive_model = self._reg(lags_past_covariates=[-step], **self._reg_params)
            positive_model = RegressionModel(
                lags_past_covariates=[-step], model=self._reg
            )
            positive_model.fit(target_pos, past_covariates=past_cov_pos)
            self._models[step] = (binary_model, positive_model)
        self.is_fitted_ = True

    @views_validate
    def predict(
        self, df: pd.DataFrame, run_type: str, eval_type: str = "standard"
    ) -> pd.DataFrame:
        df = self._process_data(df)
        check_is_fitted(self, "is_fitted_")

        if run_type != "forecasting":
            final_preds = []
            if eval_type == "standard":
                for sequence_number in tqdm.tqdm(
                    range(
                        ModelManager._resolve_evaluation_sequence_number(eval_type),
                        desc="Predicting for sequence number",
                    )
                ):
                    pred_by_step_binary = [
                        self._predict_by_step(
                            self._models[step][0], step, sequence_number
                        )
                        for step in self._steps
                    ]
                    pred_by_step_positive = [
                        self._predict_by_step(
                            self._models[step][1], step, sequence_number
                        )
                        for step in self._steps
                    ]
                    final_pred = pd.concat(pred_by_step_binary, axis=0) * pd.concat(
                        pred_by_step_positive, axis=0
                    )
                    final_preds.append(final_pred)

        else:
            pred_by_step_binary = [
                self._predict_by_step(self._models[step][0], step, 0)
                for step in self._steps
            ]
            pred_by_step_positive = [
                self._predict_by_step(self._models[step][1], step, 0)
                for step in self._steps
            ]
            final_preds = pd.concat(pred_by_step_binary, axis=0) * pd.concat(
                pred_by_step_positive, axis=0
            )

        return final_preds
