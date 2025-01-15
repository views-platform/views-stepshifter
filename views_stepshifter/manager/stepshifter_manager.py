from views_pipeline_core.managers.model import ModelPathManager, ModelManager
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_stepshifter.models.stepshifter import StepshifterModel
from views_stepshifter.models.hurdle_model import HurdleModel
from views_forecasts.extensions import *
import logging
import pickle
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StepshifterManager(ModelManager):

    def __init__(self, model_path: ModelPathManager, wandb_notification: bool = False) -> None:
        super().__init__(model_path, wandb_notification)
        self._is_hurdle = self._config_meta["algorithm"] == "HurdleModel"

    @staticmethod
    def _get_standardized_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the DataFrame based on the run type

        Args:
            df: The DataFrame to standardize

        Returns:
            The standardized DataFrame
        """

        # post-process: replace negative values with 0
        df = df.replace([np.inf, -np.inf], 0)
        df = df.mask(df < 0, 0)
        return df

    def _update_sweep_config(self, args):
        """
        Updates the configuration object with config_hyperparameters, config_meta, config_deployment, and the command line arguments.

        Args:
            args: Command line arguments

        Returns:
            The updated configuration object.
        """

        config = self._config_sweep
        config["parameters"]["run_type"] = {"value": args.run_type}
        config["parameters"]["sweep"] = {"value": True}
        config["parameters"]["name"] = {"value": self._config_meta["name"]}
        config["parameters"]["depvar"] = {"value": self._config_meta["depvar"]}
        config["parameters"]["algorithm"] = {"value": self._config_meta["algorithm"]}

        if self._is_hurdle:
            config["parameters"]["model_clf"] = {
                "value": self._config_meta["model_clf"]
            }
            config["parameters"]["model_reg"] = {
                "value": self._config_meta["model_reg"]
            }

        return config

    def _split_hurdle_parameters(self):
        """
        Split the parameters dictionary into two separate dictionaries, one for the
        classification model and one for the regression model.

        Returns:
            A dictionary containing original config, the split classification and regression parameters.
        """

        clf_dict = {}
        reg_dict = {}
        config = self.config

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
            model = HurdleModel(self.config, partitioner_dict)
        else:
            self.config["model_reg"] = self.config["algorithm"]
            model = StepshifterModel(self.config, partitioner_dict)

        return model

    def _train_model_artifact(self):
        # print(config)
        # path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

        # W&B does not directly support nested dictionaries for hyperparameters
        if self.config["sweep"] and self._is_hurdle:
            self.config = self._split_hurdle_parameters()

        run_type = self.config["run_type"]
        # df_viewser = read_dataframe(
        #     path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        # )

        df_viewser = pd.DataFrame.forecasts.read_store(run=self._pred_store_name,
                                                       name=f"{self._model_path.model_name}_{self.config['run_type']}_viewser_df")

        partitioner_dict = self._data_loader.partition_dict
        stepshift_model = self._get_model(partitioner_dict)
        stepshift_model.fit(df_viewser)

        if not self.config["sweep"]:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = ModelManager.generate_model_file_name(
                run_type, file_extension=".pkl"
            )
            stepshift_model.save(path_artifacts / model_filename)
        return stepshift_model

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str) -> List[pd.DataFrame]:
        # path_raw = self._model_path.data_raw
        path_generated = self._model_path.data_generated
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type

        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
                )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        self.config["timestamp"] = path_artifact.stem[-15:]
        # df_viewser = read_dataframe(
        #     path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        # )

        df_viewser = pd.DataFrame.forecasts.read_store(run=self._pred_store_name,
                                                       name=f"{self._model_path.model_name}_{self.config['run_type']}_viewser_df")

        with open(path_artifact, "rb") as f:
            stepshift_model = pickle.load(f)

        df_predictions = stepshift_model.predict(df_viewser, run_type, eval_type)
        df_predictions = [
            StepshifterManager._get_standardized_df(df) for df in df_predictions
        ]

        return df_predictions

    def _forecast_model_artifact(self, artifact_name: str) -> pd.DataFrame:
        # path_raw = self._model_path.data_raw
        path_generated = self._model_path.data_generated
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
                )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        self.config["timestamp"] = path_artifact.stem[-15:]

        df_viewser = pd.DataFrame.forecasts.read_store(run=self._pred_store_name,
                                                       name=f"{self._model_path.model_name}_{self.config['run_type']}_viewser_df")

        try:
            with open(path_artifact, "rb") as f:
                stepshift_model = pickle.load(f)
        except FileNotFoundError:
            logger.exception(f"Model artifact not found at {path_artifact}")

        df_prediction = stepshift_model.predict(df_viewser, run_type)
        df_prediction = StepshifterManager._get_standardized_df(df_prediction)

        return df_prediction

    # def _evaluate_sweep(self, model, eval_type):
    #     path_raw = self._model_path.data_raw
    #     run_type = self.config["run_type"]
    #     steps = self.config["steps"]

    #     df_viewser = pd.read_pickle(path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}")
    #     df = model.predict(run_type, df_viewser)
    #     df = self._get_standardized_df(df)

    #     # Temporarily keep this because the metric to minimize is MSE
    #     pred_cols = [f"step_pred_{str(i)}" for i in steps]
    #     df["mse"] = df.apply(lambda row: mean_squared_error([row[self.config["depvar"]]] * len(steps),
    #                                                         [row[col] for col in pred_cols]), axis=1)

    #     wandb.log({"MSE": df["mse"].mean()})

    #     evaluation, _ = generate_metric_dict(df, self.config)
    #     log_wandb_log_dict(self.config, evaluation)
