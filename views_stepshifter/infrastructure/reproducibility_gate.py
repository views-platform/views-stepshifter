import logging

from .exceptions import MissingHyperparameterError

logger = logging.getLogger(__name__)


class ReproducibilityGate:
    """
    Canonical hyperparameter contract for views-stepshifter models.

    Defines which configuration keys are required by all stepshifter models
    (CORE_GENOME) and which are required per algorithm (ALGORITHM_GENOMES).
    The audit_manifest() method enforces these contracts at runtime.

    This class is importable by downstream packages (e.g. views-models)
    so they can validate their config_hyperparameters.py files statically.
    """

    class Config:
        """Gates related to configuration and hyperparameter integrity."""

        # Core keys required by ALL stepshifter models regardless of algorithm.
        CORE_GENOME = ["steps", "time_steps", "parameters"]

        # Algorithm-specific requirements.
        # Each entry maps to a dict with:
        #   "parameter_keys": keys that must exist inside config["parameters"]
        #   "config_keys":    additional top-level keys required in config
        ALGORITHM_GENOMES = {
            "HurdleModel": {
                "parameter_keys": ["clf", "reg"],
                "config_keys": [],
            },
            "ShurfModel": {
                "parameter_keys": ["clf", "reg"],
                "config_keys": [
                    "submodels_to_train",
                    "pred_samples",
                    "log_target",
                    "draw_dist",
                    "draw_sigma",
                ],
            },
            "XGBRegressor": {
                "parameter_keys": ["n_estimators", "n_jobs"],
                "config_keys": [],
            },
            "XGBRFRegressor": {
                "parameter_keys": ["n_estimators", "n_jobs"],
                "config_keys": [],
            },
            "LGBMRegressor": {
                "parameter_keys": ["n_estimators", "n_jobs"],
                "config_keys": [],
            },
        }

        @staticmethod
        def audit_manifest(config: dict) -> None:
            """
            Verify that all mandatory hyperparameters are present and non-None.

            Checks in order:
            1. All CORE_GENOME keys are present.
            2. The algorithm is registered in ALGORITHM_GENOMES.
            3. All algorithm-specific keys are present.
            4. No required key has a value of None.

            Raises MissingHyperparameterError on any violation.
            """
            gate = ReproducibilityGate.Config

            # 1. Audit Core Genome
            missing_core = [k for k in gate.CORE_GENOME if k not in config]
            if missing_core:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Missing core parameters: {missing_core}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 2. Identify algorithm and check it is registered
            if "algorithm" not in config:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    "Missing required key: 'algorithm'"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            algo = config["algorithm"]
            if algo not in gate.ALGORITHM_GENOMES:
                available = list(gate.ALGORITHM_GENOMES.keys())
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Unknown algorithm '{algo}'. "
                    f"Available: {available}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            genome = gate.ALGORITHM_GENOMES[algo]
            parameters = config["parameters"]

            # 2c. Guard against parameters=None before iterating over it.
            # Without this guard, step 3a would raise TypeError instead of the
            # contracted MissingHyperparameterError.
            if parameters is None:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    "Mandatory parameters set to None: ['parameters']. "
                    "Implicit defaults are forbidden."
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 3a. Audit algorithm-specific parameter keys
            missing_params = [
                k for k in genome["parameter_keys"] if k not in parameters
            ]
            if missing_params:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Algorithm '{algo}' requires missing parameter keys: "
                    f"{missing_params}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 3b. Audit algorithm-specific top-level config keys
            missing_config = [
                k for k in genome["config_keys"] if k not in config
            ]
            if missing_config:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Algorithm '{algo}' requires missing config keys: "
                    f"{missing_config}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 4. Reject None values for all required keys
            none_core = [
                k for k in gate.CORE_GENOME if config.get(k) is None
            ]
            none_params = [
                k for k in genome["parameter_keys"]
                if parameters.get(k) is None
            ]
            none_config = [
                k for k in genome["config_keys"] if config.get(k) is None
            ]
            explicit_nones = none_core + none_params + none_config
            if explicit_nones:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Mandatory parameters set to None: {explicit_nones}. "
                    "Implicit defaults are forbidden."
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)
