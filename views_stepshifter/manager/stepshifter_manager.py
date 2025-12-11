from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe, generate_model_file_name
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.hurdle_model import HurdleModel
from views_stepshifter.models.shurf_model import ShurfModel
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict

logger = logging.getLogger(__name__)


class StepshifterManager(ForecastingModelManager):
    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        super().__init__(model_path, wandb_notifications, use_prediction_store)
        self._is_hurdle = self._config_meta["algorithm"] == "HurdleModel"
        self._is_shurf = self._config_meta["algorithm"] == "ShurfModel"

    @staticmethod
    def _get_standardized_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the DataFrame based on the run type

        Args:
            df: The DataFrame to standardize

        Returns:
            The standardized DataFrame
        """

        def standardize_value(value):
            # 1) Replace inf, -inf, nan with 0; 
            # 2) Replace negative values with 0
            if isinstance(value, list):
                return [0 if (v == np.inf or v == -np.inf or v < 0 or np.isnan(v)) else v for v in value]
            else:
                return 0 if (value == np.inf or value == -np.inf or value < 0 or np.isnan(value)) else value

        df = df.applymap(standardize_value)

        return df

    def _split_hurdle_parameters(self):
        """
        Split the parameters dictionary into two separate dictionaries, one for the
        classification model and one for the regression model.

        Returns:
            A dictionary containing original config, the split classification and regression parameters.
        """
        clf_dict = {}
        reg_dict = {}
        config = self.configs

        for key, value in config.items():
            if key.startswith("clf_"):
                clf_key = key.replace("clf_", "")
                clf_dict[clf_key] = value
            elif key.startswith("reg_"):
                reg_key = key.replace("reg_", "")
                reg_dict[reg_key] = value

        config["clf"] = clf_dict
        config["reg"] = reg_dict

        return config

    def _get_model(self, partitioner_dict: dict):
        """
        Get the model based on the algorithm specified in the config

        Args:
            partitioner_dict: The dictionary of partitioners.

        Returns:
            The model object based on the algorithm specified in the config
        """
        if self._is_hurdle:
            model = HurdleModel(self.configs, partitioner_dict)
        elif self._is_shurf:
            model = ShurfModel(self.configs, partitioner_dict)
        else:
            self.configs = {"model_reg": self.configs["algorithm"]}
            model = StepshifterModel(self.configs, partitioner_dict)

        return model

    def _train_model_artifact(self):
        """
        Train the model and save it as an artifact if not a sweep.

        Returns:
            The trained model object
        """
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        # W&B does not directly support nested dictionaries for hyperparameters
        if self.configs["sweep"] and (self._is_hurdle or self._is_shurf):
            self.configs = self._split_hurdle_parameters()

        run_type = self.configs["run_type"]
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        partitioner_dict = self._data_loader.partition_dict
        stepshift_model = self._get_model(partitioner_dict)
        stepshift_model.fit(df_viewser)

        if not self.configs["sweep"]:
            model_filename = generate_model_file_name(
                run_type, file_extension=".pkl"
            )
            stepshift_model.save(path_artifacts / model_filename)
        return stepshift_model

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str
    ) -> List[pd.DataFrame]:
        """
        Evaluate the model artifact based on the evaluation type and the artifact name.

        Args:
            eval_type: The evaluation type
            artifact_name: The name of the artifact to evaluate

        Returns:
            A list of DataFrames containing the evaluation results
        """
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact {path_artifact.name}"
            )

        self.configs = {"timestamp": path_artifact.stem[-15:]}

        try:
            with open(path_artifact, "rb") as f:
                stepshift_model = pickle.load(f)
        except FileNotFoundError:
            logger.exception(f"Model artifact not found at {path_artifact}")
            raise
        
        df_predictions = stepshift_model.predict(run_type, eval_type)
        df_predictions = [
            StepshifterManager._get_standardized_df(df) for df in df_predictions
        ]
        return df_predictions

    def _forecast_model_artifact(self, artifact_name: str) -> pd.DataFrame:
        """
        Forecast using the model artifact.

        Args:
            artifact_name: The name of the artifact to use for forecasting

        Returns:
            The forecasted DataFrame
        """
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact {path_artifact.name}"
            )
            
        self.configs = {"timestamp": path_artifact.stem[-15:]}

        try:
            with open(path_artifact, "rb") as f:
                stepshift_model = pickle.load(f)
        except FileNotFoundError:
            logger.exception(f"Model artifact not found at {path_artifact}")
            raise

        df_prediction = stepshift_model.predict(run_type)
        df_prediction = StepshifterManager._get_standardized_df(df_prediction)

        return df_prediction

    def _evaluate_sweep(self, eval_type: str, model: any) -> List[pd.DataFrame]:
        run_type = self.configs["run_type"]

        df_predictions = model.predict(run_type, eval_type)
        df_predictions = [
            StepshifterManager._get_standardized_df(df) for df in df_predictions
        ]

        return df_predictions