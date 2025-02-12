from views_pipeline_core.managers.model import ModelManager
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.hurdle_model import HurdleModel
from views_stepshifter.models.validation import views_validate
from sklearn.utils.validation import check_is_fitted
from darts.models import RegressionModel
import pandas as pd
from typing import List, Dict
import numpy as np
import logging
from darts.models import RandomForest
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StepShiftedHurdleUncertainRF(HurdleModel):
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
        # print(config)
        # self._clf_params = self._get_parameters(config)['clf']
        # self._reg_params = self._get_parameters(config)['reg']
        # self._clf = self._resolve_clf_model(config["model_clf"])
        # self._reg = self._resolve_reg_model(config["model_reg"])

        # self._submodel_list = []

        # self._partitioner_dict = partitioner_dict
        self._submodels_to_train = config["submodels_to_train"]
        # self._n_estimators = config['parameters']['n_estimators']
        self.log_target = config["log_target"]
        # self._max_features = config["max_features"]
        # self._max_depth = config["max_depth"]
        # self._max_samples = config["max_samples"]
        self._pred_samples = config["pred_samples"]
        self._draw_dist = config["draw_dist"]
        self._draw_sigma = config["draw_sigma"]
        self._geo_unit_samples = config["geo_unit_samples"]
        # self._n_jobs = config["n_jobs"]

    # @views_validate
    # def fit(self, df: pd.DataFrame):
    #     """
    #     Generate predictions using the trained submodels.
    #     This method performs the following steps:
    #     1. Prepares the data for classification and regression stages.
    #     2. Iterates over each submodel to generate predictions:
    #         - Predicts probabilities using the classification model.
    #         - Predicts target values using the regression model.
    #         - Handles infinite values in predictions.
    #     3. Draws samples from the distributions:
    #         - For each prediction sample, combines classification and regression predictions.
    #         - Applies binomial, Poisson, or lognormal distributions to generate final predictions.
    #     4. Aggregates the predictions from all submodels into a final DataFrame.
    #     Returns:
    #         pd.DataFrame: A DataFrame containing the final set of predictions with indices set to 'draw'.
    #     """
    #     df = self._process_data(df)
    #     self._prepare_time_series(df)

    #     target_binary = [
    #         s.map(lambda x: (x > 0).astype(float)) for s in self._target_train
    #     ]

    #     # Positive outcome (for cases where target > 0)
    #     target_pos, past_cov_pos = zip(
    #         *[
    #             (t, p)
    #             for t, p in zip(self._target_train, self._past_cov)
    #             if (t.values() > 0).any()
    #         ]
    #     )

    #     for i in tqdm(range(self._submodels_to_train), desc="Training submodel"):
    #         # logger.info(f"Training submodel {i+1}/{self._submodels_to_train}")
    #         for step in tqdm(self._steps, desc=f"Steps for submodel {i+1}"):
    #             # logger.info(f"Training step {step}")
    #             # Fit binary-like stage using a regression model, but the target is binary (0 or 1)
    #             binary_model = RegressionModel(
    #                 lags_past_covariates=[-step], model=self._clf
    #             )
    #             binary_model.fit(target_binary, past_covariates=self._past_cov)

    #             positive_model = RegressionModel(
    #                 lags_past_covariates=[-step], model=self._reg
    #             )
    #             positive_model.fit(target_pos, past_covariates=past_cov_pos)
    #         submodel_dict = {}
    #         submodel_dict["model_clf"] = binary_model
    #         submodel_dict["model_reg"] = positive_model
    #         self._submodel_list.append(submodel_dict)
    #         logger.info(
    #             f"Submodel {i+1}/{self._submodels_to_train} trained successfully"
    #         )
    #     self.is_fitted_ = True

    def predict_sequence(self, run_type, sequence_number) -> pd.DataFrame:
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

        sample_number = 0
        final_pred_samples = []

        pred_by_step_binary = [
            self._predict_by_step(self._models[step][0], step, sequence_number)
            for step in self._steps
        ]

        pred_by_step_positive = [
            self._predict_by_step(self._models[step][1], step, sequence_number)
            for step in self._steps
        ]
        pred_concat_binary = pd.concat(pred_by_step_binary, axis=0)
        pred_concat_positive = pd.concat(pred_by_step_positive, axis=0)

        # Loop over submodels
        for submodel_number in tqdm(
            range(self._submodels_to_train),
            desc=f"Predicting submodel: {run_type}",
            leave=True,
        ):
            # pred_by_step_binary = [
            #     self._predict_by_step(self._models[step][0], step, sequence_number)
            #     for step in self._steps
            # ]

            # pred_by_step_positive = [
            #     self._predict_by_step(self._models[step][1], step, sequence_number)
            #     for step in self._steps
            # ]
            # pred_concat_binary = pd.concat(pred_by_step_binary, axis=0)
            # pred_concat_positive = pd.concat(pred_by_step_positive, axis=0)
            for subsample_number in range(self._pred_samples):
                # Draw samples from the classifier; 0/1 according to the classifier's prediction
                pred_concat_binary_drawn = pred_concat_binary.where(
                    np.random.binomial(n=1, p=pred_concat_binary) == 1, 0
                )
                if self.log_target:
                    if (
                        self._draw_dist == "Poisson"
                    ):  # Note: this assumes a non-log-transformed target
                        pred_concat_positive_drawn = np.log1p(
                            np.random.poisson(np.expm1(pred_concat_positive))
                        )

                    if self._draw_dist == "Lognormal":
                        # Application of lognormal distribution for log-transformed outcomes
                        # Draw from lognormal only for those with predictions larger than zero
                        pred_concat_positive_drawn = np.where(
                            pred_concat_positive > 0,
                            np.random.normal(pred_concat_positive, self._draw_sigma),
                            0,
                        )

                    if self._draw_dist == "":
                        # Use the predicted value without a random draw
                        pred_concat_positive_drawn = pred_concat_positive

                if not self.log_target:
                    if (
                        self._draw_dist == "Poisson"
                    ):  # Note: this assumes a non-log-transformed target
                        pred_concat_positive_drawn = np.random.poisson(
                            pred_concat_positive
                        )

                    if self._draw_dist == "Lognormal":
                        # Draw from lognormal only for those with predictions larger than zero
                        pred_concat_positive_drawn = np.where(
                            pred_concat_positive > 0,
                            np.round(
                                np.random.lognormal(
                                    np.log1p(pred_concat_positive), self._draw_sigma
                                )
                                - 1
                            ),
                            0,
                        )

                    if self._draw_dist == "":
                        # Use the predicted value without a random draw
                        pred_concat_positive_drawn = pred_concat_positive

                # Combine the binary and positive predictions
                sample_pred = pred_concat_binary_drawn * pred_concat_positive_drawn
                # sample_pred["submodel"] = submodel_number
                # sample_pred["draw"] = sample_number
                # # Add 'sample_number' to the indices of the sample predictions DataFrame:

                # sample_pred.set_index(["draw"], append=True, inplace=True)

                # Append the combined predictions to the predictions list for this sequence number
                final_pred_samples.append(sample_pred)
                # sample_number += 1

        # Generate a DataFrame from the final predictions list for this sequence number
        # final_pred_full = pd.concat(final_pred_samples, axis=0)
        final_pred_full = pd.DataFrame(
            {
                col: [list(x) for x in zip(*[df[col] for df in final_pred_samples])]
                for col in final_pred_samples[0].columns
            },
            index=final_pred_samples[0].index,
        )
        return final_pred_full

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

        # Check if the model has been fitted before making predictions
        check_is_fitted(self, "is_fitted_")

        # If the run type is not 'forecasting', perform multiple predictions
        if run_type != "forecasting":
            final_preds = []
            final_preds_full = []
            # If the evaluation type is "standard", iterate over the evaluation sequence number
            if eval_type == "standard":
                for sequence_number in tqdm(
                    range(ModelManager._resolve_evaluation_sequence_number(eval_type)),
                    desc=f"Sequence",
                    leave=True,
                ):
                    final_pred_full = self.predict_sequence(run_type, sequence_number)

                    # final_pred = final_pred_full.groupby(["month_id", "country_id"]).mean()
                    # final_pred.pop("submodel")

                    final_preds_full.append(final_pred_full)
                    # final_preds.append(final_pred)

            return final_preds_full

        else:

            # If the run type is 'forecasting', perform a single prediction
            # submodel_preds = {}
            # for i in tqdm(range(self._submodels_to_train), desc=f"Predicting submodel: {run_type}"):
            #     pred_by_step_binary = [self._predict_by_step(self._models[step][0], step, 0)
            #                         for step in self._steps]
            #     pred_by_step_positive = [self._predict_by_step(self._models[step][1], step, 0)
            #                             for step in self._steps]
            #     # Combine binary and positive predictions by multiplying them
            #     final_preds = pd.concat(pred_by_step_binary, axis=0) * pd.concat(pred_by_step_positive, axis=0)
            #     submodel_preds[i] = final_preds

            # # Return the final predictions as a DataFrame
            # return final_preds
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
