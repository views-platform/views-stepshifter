from views_pipeline_core.managers.model import ForecastingModelManager
from views_stepshifter.models.hurdle_model import HurdleModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from typing import List, Dict
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ShurfModel(HurdleModel):
    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]]):
        super().__init__(config, partitioner_dict)
        self._clf_params = self._get_parameters(config)["clf"]
        self._reg_params = self._get_parameters(config)["reg"]

        self._submodel_list = []
        self._submodels_to_train = config["submodels_to_train"]
        self._log_target = config["log_target"]
        self._pred_samples = config["pred_samples"]
        self._draw_dist = config["draw_dist"]
        self._draw_sigma = config["draw_sigma"]

        # self._n_estimators = config['parameters']['n_estimators']
        # self._max_features = config['max_features']
        # self._max_depth = config['max_depth']
        # self._max_samples = config['max_samples']
        # self._geo_unit_samples = config['geo_unit_samples']
        # self._n_jobs = config['n_jobs']

    @views_validate
    def fit(self, df: pd.DataFrame):
        """
        Generate predictions using the trained submodels.
        This method performs the following steps:
        1. Prepares the data for classification and regression stages.
        2. Iterates over each submodel to generate predictions:
            - Predicts probabilities using the classification model.
            - Predicts target values using the regression model.
            - Handles infinite values in predictions.
        3. Draws samples from the distributions:
            - For each prediction sample, combines classification and regression predictions.
            - Applies binomial, Poisson, or lognormal distributions to generate final predictions.
        4. Aggregates the predictions from all submodels into a final DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the final set of predictions with indices set to 'draw'.
        """
        df = self._process_data(df)
        self._prepare_time_series(df)
        self._clf = self._resolve_clf_model(self._config["model_clf"])
        self._reg = self._resolve_reg_model(self._config["model_reg"])

        target_binary = [
            s.map(lambda x: (x > 0).astype(float)) for s in self._target_train
        ]

        target_pos, past_cov_pos = zip(
            *[
                (t, p)
                for t, p in zip(self._target_train, self._past_cov)
                if (t.values() > 0).any()
            ]
        )

        for i in tqdm(range(self._submodels_to_train), desc="Training submodel"):

            for step in tqdm(self._steps, desc=f"Steps for submodel {i+1}"):
                # Fit binary-like stage using a regression model, but the target is binary (0 or 1)
                binary_model = self._clf(
                    lags_past_covariates=[-step], **self._clf_params
                )
                binary_model.fit(target_binary, past_covariates=self._past_cov)

                # Fit positive stage using the regression model
                positive_model = self._reg(
                    lags_past_covariates=[-step], **self._reg_params
                )
                positive_model.fit(target_pos, past_covariates=past_cov_pos)
                self._models[step] = (binary_model, positive_model)

            self._submodel_list.append(self._models)

        self.is_fitted_ = True

    def predict_sequence(self, sequence_number) -> pd.DataFrame:
        """
        Predicts n draws of outcomes based on the provided DataFrame .

        Parameters:
        -----------
        self: StepShiftedHurdleUncertainRF
            The model object.

        run_type : str
            The type of run to perform. Currently it is unlikely to affect the behaviour of the function.

        eval_type : str
            The type of evaluation to perform. Currently it is unlikely to affect the behaviour of the function.

        sequence_number : int
            The sequence number to predict outcomes for.


        Returns:
        --------
        pd.DataFrame
            The final predictions as a DataFrame.
        """

        final_preds = []  
        submodel_number = 0

        for submodel in tqdm(
            self._submodel_list, desc=f"Predicting submodel number", leave=True
        ):
            pred_by_step_binary = [
                self._predict_by_step(submodel[step][0], step, sequence_number)
                for step in self._steps
            ]
            pred_by_step_positive = [
                self._predict_by_step(submodel[step][1], step, sequence_number)
                for step in self._steps
            ]

            pred_concat_binary = pd.concat(pred_by_step_binary, axis=0)

            pred_concat_binary.rename(
                columns={f"pred_{self._targets}": "Classification"}, inplace=True
            )
            pred_concat_positive = pd.concat(pred_by_step_positive, axis=0)
            pred_concat_positive.rename(
                columns={f"pred_{self._targets}": "Regression"}, inplace=True
            )
            pred_concat = pd.concat([pred_concat_binary, pred_concat_positive], axis=1)
            pred_concat["submodel"] = submodel_number

            final_preds.append(pred_concat)
            submodel_number += 1

        final_preds_aslists = pd.concat(final_preds, axis=0)

        # Drawing samples from the classification model
        # Ensuring that the classification probabilities are between 0 and 1:
        final_preds_aslists["Classification"] = final_preds_aslists[
            "Classification"
        ].apply(lambda x: np.clip(x, 0, 1))
        final_preds_aslists["ClassificationSample"] = final_preds_aslists[
            "Classification"
        ].apply(lambda x: np.random.binomial(1, x, self._pred_samples))

        # Drawing samples from the regression model
        if self._log_target == True:
            if (
                self._draw_dist == "Poisson"
            ):  # Note: the Poisson distribution assumes a non-log-transformed target, so not defined here
                final_preds_aslists["RegressionSample"] = final_preds_aslists[
                    "Regression"
                ]
            if self._draw_dist == "Lognormal":
                # Draw from normal distribution for log-transformed outcomes, then exponentiate, then round to integer
                final_preds_aslists["RegressionSample"] = final_preds_aslists[
                    "Regression"
                ].apply(
                    lambda x: np.abs(
                        np.rint(
                            np.expm1(
                                np.random.normal(
                                    x, self._draw_sigma, self._pred_samples
                                )
                            )
                        )
                    )
                )

        if self._log_target == False:
            if (
                self._draw_dist == "Poisson"
            ):  # Note: this assumes a non-log-transformed target
                final_preds_aslists["RegressionSample"] = final_preds_aslists[
                    "Regression"
                ]
            if self._draw_dist == "Lognormal":
                final_preds_aslists["RegressionSample"] = final_preds_aslists[
                    "Regression"
                ].apply(
                    lambda x: np.abs(
                        np.rint(
                            np.expm1(
                                np.random.normal(
                                    np.log1p(x), self._draw_sigma, self._pred_samples
                                )
                            )
                        )
                    )
                )

        if self._draw_dist == "":
            final_preds_aslists["RegressionSample"] = final_preds_aslists["Regression"]

        # 'Explode' the samples to get one row per sample
        final_preds_full = final_preds_aslists.explode(
            ["ClassificationSample", "RegressionSample"]
        )
        final_preds_full["Prediction"] = (
            final_preds_full["ClassificationSample"]
            * final_preds_full["RegressionSample"]
        )

        # Ensuring that the final predictions are positive:
        final_preds_full["Prediction"] = final_preds_full["Prediction"].apply(
            lambda x: np.clip(x, 0, None)
        )

        # Column for the main prediction:
        pred_col_name = "pred_" + self._targets
        final_preds_full[pred_col_name] = final_preds_full["Prediction"]

        # Log-transforming the final predictions if the target is log-transformed, exponentiating if not, 
        # and adding a column with the log-transformed predictions
        if self._log_target == True:
            final_preds_full["LogPrediction"] = final_preds_full["Prediction"]
            final_preds_full["Prediction"] = np.expm1(final_preds_full["Prediction"])
        if self._log_target == False:
            final_preds_full["LogPrediction"] = np.log1p(final_preds_full["Prediction"])

        final_preds_full.drop(
            columns=[
                "Classification",
                "Regression",
                "ClassificationSample",
                "RegressionSample",
                "submodel",
                "Prediction",
                "LogPrediction",
            ],
            inplace=True,
        )
        final_preds = pd.DataFrame(
            final_preds_full.groupby(["month_id", "country_id"])[pred_col_name].apply(
                list
            )
        )

        return final_preds

    def predict(self, run_type: str, eval_type: str = "standard") -> pd.DataFrame:
        """
        Predicts outcomes based on the provided DataFrame and run type.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data for making predictions.
        run_type : str
            The type of run to perform. If 'forecasting', a single prediction is made.
            Otherwise, multiple predictions are made based on the evaluation sequence number.
        eval_type : str, optional
            The type of evaluation to perform. Default is "standard".

        Returns:
        --------
        pd.DataFrame
            The final predictions as a DataFrame.
        """
        check_is_fitted(self, "is_fitted_")

        if run_type != "forecasting":
            preds = []
            if eval_type == "standard":
                for sequence_number in tqdm(
                    range(ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)),
                    desc=f"Predicting for sequence number",
                    leave=True,
                ):
                    temp_preds_full = self.predict_sequence(sequence_number)
                    preds.append(temp_preds_full)
            else:
                raise ValueError(
                    f"{eval_type} is not supported now. Please use 'standard' evaluation type."
                )

        else:
            sequence_number = 0
            preds = self.predict_sequence(sequence_number)

        return preds
