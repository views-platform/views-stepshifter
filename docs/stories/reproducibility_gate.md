# Story: ReproducibilityGate for views-stepshifter

**Status:** Accepted
**Date:** 2026-04-06
**Origin:** views-models risk register C-05
**Reference implementation:** `views_r2darts2.infrastructure.reproducibility_gate.ReproducibilityGate`

---

## Problem

views-stepshifter has no canonical definition of which hyperparameters each algorithm requires. A missing hyperparameter in a model's `config_hyperparameters.py` surfaces only at training time — often hours into a pipeline run.

In views-models, 20+ stepshifter models each define their own HP configs. There is no way to validate these configs statically because the package doesn't expose what it expects.

views-r2darts2 solved this with a `ReproducibilityGate` class that defines `CORE_GENOME` (params required by all DARTS models) and `ALGORITHM_GENOMES` (params required per architecture). This contract is:
- Enforced at runtime (audit runs before training starts)
- Importable by downstream tests (views-models imports it to validate all 15 DARTS models)
- Documented via a CIC (Class Intent Contract)

views-stepshifter has no equivalent. This story proposes adding one.

---

## Scope

### CORE_GENOME (required by all stepshifter models)

Based on empirical analysis of all 20+ stepshifter model configs in views-models:

```python
CORE_GENOME = ["steps", "time_steps", "parameters"]
```

- `steps`: list of forecast step indices (e.g., `[1, 2, ..., 36]`)
- `time_steps`: integer matching `len(steps)`
- `parameters`: dict of estimator hyperparameters (structure varies by algorithm)

### ALGORITHM_GENOMES (per-algorithm requirements)

| Algorithm | Required in `parameters` | Additional top-level keys |
|-----------|-------------------------|--------------------------|
| `HurdleModel` | `clf` (dict), `reg` (dict) | — |
| `ShurfModel` | `clf` (dict), `reg` (dict) | `submodels_to_train`, `pred_samples`, `log_target`, `draw_dist`, `draw_sigma` |
| `XGBRegressor` | `n_estimators`, `n_jobs` | — |
| `XGBRFRegressor` | `n_estimators`, `n_jobs` | — |
| `LGBMRegressor` | `n_estimators`, `n_jobs` | — |

### Runtime enforcement

The gate should audit hyperparameters when `StepshifterManager` initializes a model run:
1. Check all `CORE_GENOME` keys are present
2. Look up the model's algorithm and check `ALGORITHM_GENOMES` keys
3. Raise `MissingHyperparameterError` (or equivalent) if any are absent

### Importable contract

The gate must be importable by views-models tests:

```python
from views_stepshifter.infrastructure.reproducibility_gate import ReproducibilityGate

CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.Config.ALGORITHM_GENOMES
```

---

## Acceptance Criteria

1. `ReproducibilityGate` class exists with `CORE_GENOME` and `ALGORITHM_GENOMES`
2. Runtime audit runs before training in `StepshifterManager`
3. `MissingHyperparameterError` raised for missing params
4. views-models can import the gate and validate all stepshifter models (same pattern as `test_darts_reproducibility.py`)
5. Tests in views-stepshifter cover the gate itself

---

## Notes

- ADR-001 and a CIC for `ReproducibilityGate` were created alongside this implementation, establishing the governance pattern for views-stepshifter.
- The `parameters` dict structure varies significantly between algorithms (nested clf/reg for Hurdle/Shurf, flat for XGB/LGBM/RF). The gate validates structure, not just key presence.
- ShurfModel has the most complex param requirements (5 additional top-level keys for sampling/distribution control).
