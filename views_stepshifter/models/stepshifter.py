import pickle
import numpy as np
import pandas as pd
import logging
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.utils.validation import check_is_fitted
from typing import List, Dict
from views_stepshifter.models.validation import views_validate
from views_pipeline_core.managers.model import ModelManager
import tqdm
from concurrent.futures import ProcessPoolExecutor

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
        if not isinstance(config["depvar"], list):
            raise ValueError("Dependent variable must be a list")
        elif len(config["depvar"]) > 1:
            raise ValueError("Stepshifter only supports one dependent variable")
        else:
            self._depvar = config["depvar"][0]

    def _resolve_reg_model(self, func_name: str):
        """Lookup table for supported regression models"""

        match func_name:
            case "XGBRFRegressor":
                from xgboost import XGBRFRegressor

                return XGBRFRegressor(**self._reg_params)
            case "XGBRegressor":
                from xgboost import XGBRegressor

                return XGBRegressor(**self._reg_params)
            case "LGBMRegressor":
                from lightgbm import LGBMRegressor

                return LGBMRegressor(**self._reg_params)
            case "GradientBoostingRegressor":
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(**self._reg_params)
            case "RandomForestRegressor":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(**self._reg_params)
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
        self._independent_variables = [c for c in df.columns if c != self._depvar]

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

        return df

    def _prepare_time_series(self, df: pd.DataFrame):
        """
        Prepare time series for training and prediction
        """

        df_reset = df.reset_index(level=[1])
        self._series = TimeSeries.from_group_dataframe(
            df_reset,
            group_cols=self._level,
            value_cols=self._independent_variables + [self._depvar],
        )

        self._target_train = [
            series.slice(self._train_start, self._train_end + 1)[self._depvar]
            for series in self._series
        ]  # ts.slice is different from df.slice
        self._past_cov = [
            series[self._independent_variables] for series in self._series
        ]

    def _predict_by_step(self, model, step: int, sequence_number: int):
        """
        Keep predictions with last-month-with-data, i.e., diagonal prediction
        """

        # logger.info(f"Predicting for step {step} and sequence number {sequence_number}")
        slice_end = self._train_end + 1 + sequence_number
        target = [
            series.slice(self._train_start, slice_end)[
                self._depvar
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

        index_tuples, df_list = [], []
        test_index = self._test_start + step + sequence_number - 1
        for pred in ts_pred:
            df_pred = pred.pd_dataframe().loc[[test_index]]
            level = int(pred.static_covariates.iat[0, 0])
            index_tuples.extend([(month, level) for month in df_pred.index])
            df_list.append(df_pred.values)

        df_preds = pd.DataFrame(
            data=np.concatenate(df_list),
            index=pd.MultiIndex.from_tuples(
                index_tuples, names=[self._time, self._level]
            ),
            columns=[f"pred_{self._depvar}"],
        )

        return df_preds.sort_index()

    @views_validate
    def fit(self, df: pd.DataFrame):
        df = self._process_data(df)
        self._prepare_time_series(df)
        self._reg = self._resolve_reg_model(self._config["model_reg"])
        for step in tqdm.tqdm(self._steps, desc="Fitting model for step", leave=True):
            # model = self._reg(lags_past_covariates=[-step], **self._params)
            model = RegressionModel(lags_past_covariates=[-step], model=self._reg)
            model.fit(
                self._target_train, past_covariates=self._past_cov
            )  # Darts will automatically ignore the parts of past_covariates that go beyond the training period
            self._models[step] = model
        self.is_fitted_ = True

    def predict(
        self, run_type: str, eval_type: str = "standard"
    ) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")

        if run_type != "forecasting":
            preds = []
            if eval_type == "standard":
                # for sequence_number in tqdm.tqdm(
                #     range(ModelManager._resolve_evaluation_sequence_number(eval_type)),
                #     desc="Predicting for sequence number",
                # ):
                #     pred_by_step = [
                #         self._predict_by_step(self._models[step], step, sequence_number)
                #         for step in self._steps
                #     ]
                #     pred = pd.concat(pred_by_step, axis=0)
                #     preds.append(pred)

                with ProcessPoolExecutor() as executor:
                    for sequence_number in tqdm.tqdm(
                        range(ModelManager._resolve_evaluation_sequence_number(eval_type)),
                        desc="Predicting for sequence number",
                    ):
                        futures = [
                            executor.submit(self._predict_by_step, self._models[step], step, sequence_number)
                            for step in self._steps
                        ]
                        pred_by_step = [future.result() for future in futures]  
                        pred = pd.concat(pred_by_step, axis=0)
                        preds.append(pred)

        else:
            preds = [
                self._predict_by_step(self._models[step], step, 0)
                for step in self._steps
            ]
            preds = pd.concat(preds, axis=0)
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
    def depvar(self):
        return self._depvar
