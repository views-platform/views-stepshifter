# Class Intent Contract: ReproducibilityGate

**Date:** 2026-04-07
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001

---

## Purpose

`ReproducibilityGate` is the canonical definition of which hyperparameters each stepshifter algorithm requires. It centralises two contracts — `CORE_GENOME` (keys required by all models) and `ALGORITHM_GENOMES` (keys required per algorithm) — in a single importable location. The `audit_manifest()` method enforces these contracts at runtime by raising `MissingHyperparameterError` on any violation.

The class is importable by downstream packages (e.g. views-models) so they can validate their `config_hyperparameters.py` files in CI without running the full pipeline.

---

## Non-Goals

- Does not validate hyperparameter *values* (types, ranges, semantics). Only validates *presence* and *non-None*.
- Does not implement temporal or data integrity gates.
- Does not own model instantiation. That remains with `StepshifterManager._get_model()`.

---

## Responsibilities and Guarantees

**`Config.CORE_GENOME`**: Class attribute. List of config keys required by every stepshifter model: `["steps", "time_steps", "parameters"]`.

**`Config.ALGORITHM_GENOMES`**: Class attribute. Dict mapping each algorithm name to a dict with two lists:
- `parameter_keys`: keys required inside `config["parameters"]`.
- `config_keys`: additional top-level keys required in the config.

**`Config.audit_manifest(config)`**: Static method. Validates a config dict in sequential checks:
1. All `CORE_GENOME` keys are present in `config`.
2. `config["algorithm"]` is a key in `ALGORITHM_GENOMES`.
3. All algorithm-specific `parameter_keys` are present in `config["parameters"]`.
4. All algorithm-specific `config_keys` are present in `config`.
5. No required key (core + parameter + config) has value `None`.

If any check fails, raises `MissingHyperparameterError` with a message identifying the missing or null keys. Logs the error at `ERROR` level before raising.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `config` | `dict` | Must contain `"algorithm"` key. Expected to contain all keys declared in `CORE_GENOME` and the relevant `ALGORITHM_GENOMES` entry. `config["parameters"]` must be a dict. |

---

## Outputs and Side Effects

- **`audit_manifest()`**: Returns `None` on success. Raises `MissingHyperparameterError` on failure. Emits `ERROR` log on failure. No other side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Missing core key | `MissingHyperparameterError` (crash) | Lists all missing core keys. |
| Unknown algorithm | `MissingHyperparameterError` (crash) | Names the unknown algorithm and lists available ones. |
| Missing parameter key | `MissingHyperparameterError` (crash) | Lists missing keys for the declared algorithm. |
| Missing config key | `MissingHyperparameterError` (crash) | Lists missing top-level keys for the declared algorithm. |
| Required key set to `None` | `MissingHyperparameterError` (crash) | Lists all `None`-valued required keys. |
| `config` missing `"algorithm"` key | `MissingHyperparameterError` (crash) | Explicit guard: `"Missing required key: 'algorithm'"`. |

All failures are loud and immediate. No silent fallbacks.

---

## Boundaries and Interactions

- **Depends on:** `views_stepshifter.infrastructure.exceptions.MissingHyperparameterError`.
- **No external dependencies** beyond standard library and logging.
- **Imported by:** `StepshifterManager` (calls `audit_manifest()` in `_train_model_artifact()`).
- **Importable by:** `views-models` tests for static config validation.

---

## Examples of Correct Usage

```python
from views_stepshifter.infrastructure.reproducibility_gate import ReproducibilityGate

# Valid config — passes silently
config = {
    "algorithm": "XGBRegressor",
    "steps": [*range(1, 37)],
    "time_steps": 36,
    "parameters": {"n_estimators": 100, "n_jobs": 4},
}
ReproducibilityGate.Config.audit_manifest(config)  # no error

# Downstream validation in views-models
CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.Config.ALGORITHM_GENOMES
```

---

## Examples of Incorrect Usage

```python
# Missing core key — raises
config = {"algorithm": "XGBRegressor", "time_steps": 36, "parameters": {}}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Missing core parameters: ['steps']

# None value — raises
config = {"algorithm": "XGBRegressor", "steps": None, "time_steps": 36, "parameters": {}}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Mandatory parameters set to None: ['steps']

# Unknown algorithm — raises
config = {"algorithm": "SVR", "steps": [1], "time_steps": 36, "parameters": {}}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Unknown algorithm 'SVR'. Available: [...]
```

---

## Test Alignment

File: `tests/test_reproducibility_gate.py`

| Test | What it verifies |
|---|---|
| `test_core_genome_is_list_of_strings` | `CORE_GENOME` is a non-empty list of strings. |
| `test_algorithm_genomes_covers_all_supported_models` | All 5 algorithm names are registered in `ALGORITHM_GENOMES`. |
| `test_manager_gate_rejects_incomplete_config` | End-to-end: manager rejects config missing core keys. |
| `test_audit_manifest_accepts_valid_xgb_config` | Valid flat-parameter config passes without error. |
| `test_audit_manifest_accepts_valid_hurdle_config` | Valid nested clf/reg config passes. |
| `test_audit_manifest_accepts_valid_shurf_config` | Valid ShurfModel config with all 6 extra keys passes. |
| `test_audit_manifest_rejects_missing_core_key` | Missing `steps` raises `MissingHyperparameterError`. |
| `test_audit_manifest_rejects_missing_parameter_key` | Missing `n_jobs` in parameters raises. |
| `test_audit_manifest_rejects_missing_shurf_config_key` | Missing ShurfModel top-level key raises. |
| `test_audit_manifest_rejects_unknown_algorithm` | Unknown algorithm name raises. |
| `test_downstream_import_contract` | Gate is importable and exposes expected attributes. |
| `test_all_algorithms_have_required_genome_keys` | Every genome entry has `parameter_keys` and `config_keys`. |
| `test_none_value_injection` | Required core key set to `None` raises. |
| `test_none_parameter_value_rejected` | Required parameter key set to `None` raises. |
| `test_empty_string_algorithm` | Empty-string algorithm raises. |
| `test_missing_algorithm_key` | Missing `"algorithm"` key raises with explicit message. |
| `test_extra_keys_ignored` | Surplus keys do not cause errors. |

---

## Evolution Notes

- If `Temporal` or `Data` gates are added, they should follow the nested-class pattern (`ReproducibilityGate.Temporal`, `ReproducibilityGate.Data`) established in views-r2darts2.
- If hyperparameter *value* validation is needed, it should be added as a separate `audit_values()` method, not mixed into `audit_manifest()`.
- The `ALGORITHM_GENOMES` structure uses `parameter_keys`/`config_keys` to handle the stepshifter-specific split between nested parameters and top-level config keys.

---

## Known Deviations

- Unlike views-baseline where all algorithm keys are flat top-level config keys, stepshifter algorithms split requirements between `config["parameters"]` (nested) and top-level config keys. The `ALGORITHM_GENOMES` structure reflects this with `parameter_keys` and `config_keys`.
