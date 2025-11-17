from views_pipeline_core.managers.model import ForecastingModelManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from typing import List, Dict
import logging
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
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

    def _resolve_clf_model(self, func_name: str):
        """Lookup table for supported classification models"""

        match func_name:
            case "XGBClassifier":
                from darts.models import XGBClassifierModel
                # if self.get_device_params().get("device") == "cuda":
                #     logger.info("\033[92mUsing CUDA for XGBClassifierModel\033[0m")
                #     cuda_params = {"tree_method": "hist", "device": "cuda"}
                #     return partial(XGBClassifierModel, **cuda_params)
                return XGBClassifierModel
            case "XGBRFClassifier":
                from views_stepshifter.models.darts_model import XGBRFClassifierModel
                # if self.get_device_params().get("device") == "cuda":
                #     logger.info("\033[92mUsing CUDA for XGBRFClassifierModel\033[0m")
                #     cuda_params = {"tree_method": "hist", "device": "cuda"}
                #     return partial(XGBRFClassifierModel, **cuda_params)
                return XGBRFClassifierModel
            case "LGBMClassifier":
                from darts.models import LightGBMClassifierModel
                return LightGBMClassifierModel
            case _:
                raise ValueError(
                    f"Model {func_name} is not a valid Darts forecasting model or is not supported now. "
                    f"Change the model in the config file."
                )

    def _fit_by_step(self, step, target_binary, target_pos, past_cov_pos):
        # Fit binary-like stage using a classification model
        binary_model = self._clf(lags_past_covariates=[-step], **self._clf_params)
        binary_model.fit(target_binary, past_covariates=self._past_cov)

        # Fit positive stage using the regression model
        positive_model = self._reg(lags_past_covariates=[-step], **self._reg_params)
        positive_model.fit(target_pos, past_covariates=past_cov_pos)

        return (binary_model, positive_model)
    
    def _predict_by_sequence(self, sequence_number):
        pred_by_step_binary = []
        pred_by_step_positive = []
        
        for step in self._steps:
            # Predict for binary model
            pred_binary = self._predict_by_step(self._models[step][0], step, sequence_number)
            pred_by_step_binary.append(pred_binary)
            
            # Predict for positive model
            pred_positive = self._predict_by_step(self._models[step][1], step, sequence_number)
            pred_by_step_positive.append(pred_positive)
        
        final_pred = (
            pd.concat(pred_by_step_binary, axis=0).sort_index() *
            pd.concat(pred_by_step_positive, axis=0).sort_index()
        )
    
        return final_pred

    @views_validate
    def fit(self, df: pd.DataFrame):
        df = self._process_data(df)
        self._prepare_time_series(df)
        self._clf = self._resolve_clf_model(self._config["model_clf"])
        self._reg = self._resolve_reg_model(self._config["model_reg"])

        # Binary outcome (event/no-event)
        # According to the DARTS doc, if timeseries uses a numeric type different from np.float32 or np.float64, not all functionalities may work properly.
        # So use astype(float) instead of astype(int) (we should have binary outputs 0,1 though)
        target_binary = [
            s.map(lambda x: (x > 0).astype(float)) for s in self._target_train
        ]

        # Positive outcome (for cases where target > 0)
        target_pos, past_cov_pos = zip(
            *[
                (t, p)
                for t, p in zip(self._target_train, self._past_cov)
                if (t.values() > 0).any()
            ]
        )

        if self.get_device_params().get("device") == "cuda":
            for step in tqdm.tqdm(self._steps, desc="Fitting model for step", leave=True):
                # Fit binary-like stage using a classification model, but the target is binary (0 or 1)
                binary_model = self._clf(lags_past_covariates=[-step], **self._clf_params)
                binary_model.fit(target_binary, past_covariates=self._past_cov)

                # Fit positive stage using the regression model
                positive_model = self._reg(lags_past_covariates=[-step], **self._reg_params)
                positive_model.fit(target_pos, past_covariates=past_cov_pos)
                self._models[step] = (binary_model, positive_model)
        else:
            models = {}
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self._fit_by_step, step, target_binary, target_pos, past_cov_pos): step
                    for step in self._steps
                }
                for future in tqdm.tqdm(as_completed(futures.keys()), desc="Fitting models for steps", total=len(futures)):
                    step = futures[future]
                    models[step] = future.result()
            self._models = models
        self.is_fitted_ = True

    def predict(self, run_type: str, eval_type: str = "standard") -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")

        if run_type != "forecasting":

            if eval_type == "standard":
                total_sequence_number = (
                    ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
                )
                if self.get_device_params().get("device") == "cuda":
                    preds = []
                    for sequence_number in tqdm.tqdm(
                        range(ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)),
                        desc="Predicting for sequence number",
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
                        pred = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)
                        preds.append(pred)

                else:
                    preds = [None] * total_sequence_number
                    with ProcessPoolExecutor() as executor:
                        futures = {
                            executor.submit(self._predict_by_sequence, sequence_number): sequence_number
                            for sequence_number in range(ForecastingModelManager._resolve_evaluation_sequence_number(eval_type))
                        }
                        for future in tqdm.tqdm(
                            as_completed(futures.keys()),
                            desc="Predicting for sequence number",
                            total=len(futures),
                        ):
                            sequence_number = futures[future]
                            preds[sequence_number] = future.result()
            else:
                raise ValueError(
                    f"{eval_type} is not supported now. Please use 'standard' evaluation type."
                )

        else:
            if self.get_device_params().get("device") == "cuda":
                pred_by_step_binary = []
                pred_by_step_positive = []
                for step in tqdm.tqdm(self._steps, desc="Predicting for step", total=len(self._steps)):
                    pred_by_step_binary.append(
                        self._predict_by_step(self._models[step][0], step, 0)
                    )
                    pred_by_step_positive.append(
                        self._predict_by_step(self._models[step][1], step, 0)
                    )
                
                preds = pd.concat(pred_by_step_binary, axis=0) * pd.concat(
                    pred_by_step_positive, axis=0
                )
  
            else:
                with ProcessPoolExecutor() as executor:
                    futures_binary = {
                        step: executor.submit(
                            self._predict_by_step, self._models[step][0], step, 0
                        )
                        for step in self._steps
                    }
                    futures_positive = {
                        step: executor.submit(
                            self._predict_by_step, self._models[step][1], step, 0
                        )
                        for step in self._steps
                    }

                    pred_by_step_binary = [
                        future.result()
                        for future in tqdm.tqdm(
                            as_completed(futures_binary.values()),
                            desc="Predicting binary outcomes",
                            total=len(futures_binary),
                        )
                    ]
                    pred_by_step_positive = [
                        future.result()
                        for future in tqdm.tqdm(
                            as_completed(futures_positive.values()),
                            desc="Predicting positive outcomes",
                            total=len(futures_positive),
                        )
                    ]

                    preds = (
                        pd.concat(pred_by_step_binary, axis=0).sort_index()
                        * pd.concat(pred_by_step_positive, axis=0).sort_index()
                    )
        return preds
