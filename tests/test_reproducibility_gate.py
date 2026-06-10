import pytest

from views_stepshifter.infrastructure.exceptions import MissingHyperparameterError
from views_stepshifter.infrastructure.reproducibility_gate import ReproducibilityGate

# -----------------------------------------------------------------------
# Green Team — Structural correctness
# -----------------------------------------------------------------------


def test_core_genome_is_list_of_strings():
    genome = ReproducibilityGate.Config.CORE_GENOME
    assert isinstance(genome, list)
    assert all(isinstance(k, str) for k in genome)
    assert len(genome) > 0


def test_algorithm_genomes_covers_all_supported_models():
    expected = {
        "HurdleModel", "ShurfModel",
        "XGBRegressor", "XGBRFRegressor",
        "LGBMRegressor",
    }
    assert set(ReproducibilityGate.Config.ALGORITHM_GENOMES.keys()) == expected


def test_audit_manifest_accepts_valid_xgb_config():
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100, "n_jobs": 4},
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_accepts_valid_hurdle_config():
    config = {
        "algorithm": "HurdleModel",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {
            "clf": {"n_estimators": 100},
            "reg": {"n_estimators": 100},
        },
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_accepts_valid_shurf_config():
    config = {
        "algorithm": "ShurfModel",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "submodels_to_train": 50,
        "pred_samples": 10,
        "log_target": False,
        "draw_dist": "Lognormal",
        "draw_sigma": 0.6,
        "parameters": {
            "clf": {"n_estimators": 2},
            "reg": {"n_estimators": 2},
        },
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_core_key():
    config = {
        "algorithm": "XGBRegressor",
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100, "n_jobs": 4},
        # "steps" is missing
    }
    with pytest.raises(MissingHyperparameterError, match="steps"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_parameter_key():
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100},
        # "n_jobs" missing from parameters
    }
    with pytest.raises(MissingHyperparameterError, match="n_jobs"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_shurf_config_key():
    config = {
        "algorithm": "ShurfModel",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "submodels_to_train": 50,
        # missing pred_samples, log_target, draw_dist, draw_sigma
        "parameters": {
            "clf": {"n_estimators": 2},
            "reg": {"n_estimators": 2},
        },
    }
    with pytest.raises(MissingHyperparameterError, match="pred_samples"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_unknown_algorithm():
    config = {
        "algorithm": "NonExistentModel",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {},
    }
    with pytest.raises(MissingHyperparameterError, match="NonExistentModel"):
        ReproducibilityGate.Config.audit_manifest(config)


# -----------------------------------------------------------------------
# Beige Team — Cross-module integration
# -----------------------------------------------------------------------


_has_pipeline_core = True
try:
    import views_pipeline_core  # noqa: F401
except ImportError:
    _has_pipeline_core = False


@pytest.mark.skipif(not _has_pipeline_core, reason="views_pipeline_core not installed")
def test_manager_gate_rejects_incomplete_config(monkeypatch):
    """End-to-end: the manager rejects a config missing core keys."""
    from unittest.mock import MagicMock, patch
    from views_stepshifter.manager.stepshifter_manager import StepshifterManager

    incomplete_hp = {
        "run_type": "train",
        "sweep": False,
        # "steps" and "time_steps" and "parameters" are missing
    }
    meta = {
        "name": "test",
        "algorithm": "XGBRegressor",
        "targets": "t",
        "metrics": [],
    }

    with patch.object(
        StepshifterManager,
        "_ModelManager__load_config",
        side_effect=lambda file, func: {
            "config_meta.py": meta,
            "config_deployment.py": {"deployment_status": "s"},
            "config_hyperparameters.py": incomplete_hp,
            "config_sweep.py": {"parameters": {}},
        }.get(file, None),
    ):
        mock_path = MagicMock()
        mock_path.model_dir = "/test"
        mock_path.target = "model"
        mock_path.model_name = "test"
        mock_path.logging = MagicMock()
        mock_path.models = MagicMock()
        mock_path.data_raw = MagicMock()
        mock_path.artifacts = MagicMock()

        mgr = StepshifterManager(mock_path, use_prediction_store=False)
        mgr._data_loader = MagicMock()
        mgr._data_loader.partition_dict = {"train": [0, 10], "test": [11, 20]}

        with pytest.raises(MissingHyperparameterError, match="steps"):
            mgr._train_model_artifact()


def test_downstream_import_contract():
    """The gate is importable and exposes the expected interface."""
    from views_stepshifter.infrastructure.reproducibility_gate import ReproducibilityGate

    assert hasattr(ReproducibilityGate, "Config")
    assert hasattr(ReproducibilityGate.Config, "CORE_GENOME")
    assert hasattr(ReproducibilityGate.Config, "ALGORITHM_GENOMES")
    assert hasattr(ReproducibilityGate.Config, "audit_manifest")
    assert callable(ReproducibilityGate.Config.audit_manifest)


def test_all_algorithms_have_required_genome_keys():
    """Every algorithm genome must have parameter_keys and config_keys."""
    for algo, genome in ReproducibilityGate.Config.ALGORITHM_GENOMES.items():
        assert "parameter_keys" in genome, f"{algo} missing parameter_keys"
        assert "config_keys" in genome, f"{algo} missing config_keys"
        assert isinstance(genome["parameter_keys"], list)
        assert isinstance(genome["config_keys"], list)


# -----------------------------------------------------------------------
# Red Team — Adversarial inputs
# -----------------------------------------------------------------------


def test_none_value_injection():
    """A required key present but set to None must be rejected."""
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": None,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100, "n_jobs": 4},
    }
    with pytest.raises(MissingHyperparameterError, match="None"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_none_parameter_value_rejected():
    """A required parameter key set to None must be rejected."""
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100, "n_jobs": None},
    }
    with pytest.raises(MissingHyperparameterError, match="None"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_empty_string_algorithm():
    """An empty-string algorithm must be rejected as unknown."""
    config = {
        "algorithm": "",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {},
    }
    with pytest.raises(MissingHyperparameterError, match="Unknown algorithm"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_missing_algorithm_key():
    """Config with no 'algorithm' key at all must be rejected explicitly."""
    config = {
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {},
    }
    with pytest.raises(MissingHyperparameterError, match="algorithm"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_extra_keys_ignored():
    """Surplus keys in the config must not cause errors."""
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": {"n_estimators": 100, "n_jobs": 4, "extra": True},
        "totally_unknown_key": "should be fine",
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_parameters_dict_none_rejected():
    """config['parameters'] = None must raise MissingHyperparameterError, not TypeError."""
    config = {
        "algorithm": "XGBRegressor",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "target_transform": "identity",
        "parameters": None,
    }
    with pytest.raises(MissingHyperparameterError, match="None"):
        ReproducibilityGate.Config.audit_manifest(config)
