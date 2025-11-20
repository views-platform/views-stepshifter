import pickle
import numpy as np
import pandas as pd
import logging
from darts import TimeSeries
from sklearn.utils.validation import check_is_fitted
from typing import List, Dict
from views_stepshifter.models.validation import views_validate
from views_pipeline_core.managers.model import ForecastingModelManager
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from functools import partial

logger = logging.getLogger(__name__)


class StepshifterModel:
    def __init__(self, config: Dict, partitioner_dict: Dict[str, List[int]]):
        self._config = config
        self._steps = config["steps"]
        self._reg_params = self._get_parameters(config)
        self._train_start, self._train_end = partitioner_dict["train"]
        self._test_start, self._test_end = partitioner_dict["test"]
        self._models = {}

        # Multiple targets handling
        if not isinstance(config["targets"], list):
            raise ValueError("Dependent variable must be a list")
        elif len(config["targets"]) > 1:
            raise ValueError("Stepshifter only supports one dependent variable")
        else:
            self._targets = config["targets"][0]

    @staticmethod
    def get_device_params():
        if torch.cuda.is_available():
            return {"device": "cuda"}
        elif torch.backends.mps.is_available():
            return {"device": "mps"}
        else:
            return {}

    def _resolve_reg_model(self, func_name: str):
        """
        Lookup table for supported regression models
        Note that stepshifter doesn't support cuda for now before we figure out how to move the input data to the GPU. (fit works intelligently but predict doesn't)
        """

        match func_name:
            case "XGBRFRegressor":
                from views_stepshifter.models.darts_model import XGBRFModel
                # if self.get_device_params().get("device") == "cuda":
                #     logger.info("\033[92mUsing CUDA for XGBRFRegressor\033[0m")
                #     cuda_params = {"tree_method": "hist", "device": "cuda"}
                #     return partial(XGBRFModel, **cuda_params)
                return XGBRFModel
            case "XGBRegressor":
                from darts.models import XGBModel
                # if self.get_device_params().get("device") == "cuda":
                #     logger.info("\033[92mUsing CUDA for XGBRegressor\033[0m")
                #     cuda_params = {"tree_method": "hist", "device": "cuda"}
                #     return partial(XGBModel, **cuda_params)
                return XGBModel
            case "LGBMRegressor":
                from darts.models import LightGBMModel
                # if self.get_device_params().get("device") == "cuda":
                #     logger.info("\033[92mUsing CUDA for LGBMRegressor\033[0m")
                #     cuda_params = {"device": "cuda"}
                #     return partial(LightGBMModel, **cuda_params)
                return LightGBMModel
            case _:
                raise ValueError(
                    f"Model {func_name} is not a valid forecasting model or is not supported now. "
                    f"Change the model in the config file."
                )

    def _get_parameters(self, config: Dict):
        """
        Get the parameters from the config file.
        If not sweep, then get directly from the config file, otherwise have to remove some parameters.
        """

        if config["sweep"]:
            parameters = {k: v for k, v in config.items() if k in ["clf", "reg"]}
        else:
            parameters = config["parameters"]

        return parameters

    def _process_data(self, df: pd.DataFrame):
        """
        Countries appear and disappear, so we are predicting countries that exist in the last month of the training data.
        If the country appeared earlier but don't have data previously, we will fill the missing data with 0.
        """

        # set up
        self._time = df.index.names[0]
        self._level = df.index.names[1]
        self._independent_variables = [c for c in df.columns if c != self._targets]

        last_month_id = df.index.get_level_values(self._time).max()
        existing_country_ids = df.loc[last_month_id].index.unique()
        df = df[df.index.get_level_values(self._level).isin(existing_country_ids)]

        all_months = df.index.get_level_values(self._time).unique()
        all_combinations = pd.MultiIndex.from_product(
            [all_months, existing_country_ids], names=[self._time, self._level]
        )
        missing_combinations = all_combinations.difference(df.index)

        missing_df = pd.DataFrame(0, index=missing_combinations, columns=df.columns)
        df = pd.concat([df, missing_df]).sort_index()

        df[self._targets] = np.log1p(df[self._targets]) # Calculates log(1 + x).
        return df

    def _prepare_time_series(self, df: pd.DataFrame):
        """
        Prepare time series for training and prediction
        """

        df_reset = df.reset_index(level=[1])
        self._series = TimeSeries.from_group_dataframe(
            df_reset,
            group_cols=self._level,
            value_cols=self._independent_variables + [self._targets],
        )

        self._target_train = [
            series.slice(self._train_start, self._train_end + 1)[self._targets]
            for series in self._series
        ]  # ts.slice is different from df.slice
        self._past_cov = [
            series[self._independent_variables] for series in self._series
        ]

    def _fit_by_step(self, step):
        model = self._reg(lags_past_covariates=[-step], **self._reg_params)
        model.fit(self._target_train, past_covariates=self._past_cov)
        return model

    def _predict_by_step(self, model, step: int, sequence_number: int):
        """
        Keep predictions with last-month-with-data, i.e., diagonal prediction
        """
        # logger.info(f"Starting prediction for step: {step}")
        target = [
            series.slice(self._train_start, self._train_end + 1 + sequence_number)[
                self._targets
            ]
            for series in self._series
        ]
        ts_pred = model.predict(
            n=step,
            series=target,
            # darts automatically locates the time period of past_covariates
            past_covariates=self._past_cov,
            show_warnings=False,
        )

        # process the predictions
        index_tuples, df_list = [], []
        for pred in ts_pred:
            df_pred = pred.to_dataframe().loc[
                [self._test_start + step + sequence_number - 1]
            ]
            level = int(pred.static_covariates.iat[0, 0])
            index_tuples.extend([(month, level) for month in df_pred.index])
            df_list.append(df_pred.values)

        df_preds = pd.DataFrame(
            data=np.concatenate(df_list),
            index=pd.MultiIndex.from_tuples(
                index_tuples, names=[self._time, self._level]
            ),
            columns=[f"pred_{self._targets}"],
        )

        df_preds[f"pred_{self._targets}"] = np.expm1(df_preds[f"pred_{self._targets}"]) # Calculates exp(x) - 1
        
        return df_preds.sort_index()

    def _predict_by_sequence(self, sequence_number):
        pred_by_step = []
        for step in self._steps:
            pred = self._predict_by_step(self._models[step], step, sequence_number)
            pred_by_step.append(pred)
        return pd.concat(pred_by_step, axis=0).sort_index()

    # @views_validate
    # def fit(self, df: pd.DataFrame):
    #     df = self._process_data(df)
    #     self._prepare_time_series(df)
    #     self._reg = self._resolve_reg_model(self._config["model_reg"])
    #     for step in tqdm.tqdm(
    #         self._steps, desc="Fitting model for step", leave=True
    #     ):
    #         model = self._reg(lags_past_covariates=[-step], **self._reg_params)
    #         model.fit(self._target_train,
    #                   past_covariates=self._past_cov) # Darts will automatically ignore the parts of past_covariates that go beyond the training period
    #         self._models[step] = model
    #     self.is_fitted_ = True

    @views_validate
    def fit(self, df: pd.DataFrame):
        df = self._process_data(df)
        self._prepare_time_series(df)
        self._reg = self._resolve_reg_model(self._config["model_reg"])

        models = {}
        if self.get_device_params().get("device") == "cuda":
            for step in tqdm.tqdm(
                self._steps, desc="Fitting model for step", leave=True
            ):
                model = self._reg(lags_past_covariates=[-step], **self._reg_params)
                model.fit(self._target_train,
                            past_covariates=self._past_cov) # Darts will automatically ignore the parts of past_covariates that go beyond the training period
                self._models[step] = model
        else:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self._fit_by_step, step): step for step in self._steps
                }
                for future in tqdm.tqdm(
                    futures.keys(), desc="Fitting models for steps", total=len(futures)
                ):
                    step = futures[future]
                    models[step] = future.result()
                self._models = models  # Use local variable to avoid concurrent execution issues
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
                        pred_by_step = [
                            self._predict_by_step(self._models[step], step, sequence_number)
                            for step in self._steps
                        ]
                        pred = pd.concat(pred_by_step, axis=0)
                        preds.append(pred)
                else:
                    preds = [None] * total_sequence_number
                    with ProcessPoolExecutor() as executor:
                        futures = {
                            executor.submit(
                                self._predict_by_sequence, sequence_number
                            ): sequence_number
                            for sequence_number in range(total_sequence_number)
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
                preds = []
                for step in tqdm.tqdm(self._steps, desc="Predicting for steps"):
                    preds.append(self._predict_by_step(self._models[step], step, 0))
                preds = pd.concat(preds, axis=0).sort_index()
                
            else:
                with ProcessPoolExecutor() as executor:
                    futures = {
                        step: executor.submit(
                            self._predict_by_step, self._models[step], step, 0
                        )
                        for step in self._steps
                    }
                    preds_by_step = [
                        future.result()
                        for future in tqdm.tqdm(
                            as_completed(futures.values()),
                            desc="Predicting outcomes",
                            total=len(futures),
                        )
                    ]

                preds = pd.concat(preds_by_step, axis=0).sort_index()

        return preds

    def save(self, path: str):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
            logger.info(f"Model successfully saved to {path}")
        except Exception as e:
            logger.exception(f"Failed to save model: {e}")

    @property
    def models(self):
        return list(self._models.values())

    @property
    def steps(self):
        return self._steps

    @property
    def targets(self):
        return self._targets
