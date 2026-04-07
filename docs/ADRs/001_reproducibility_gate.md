# ADR-001: Reproducibility Gate

- **Status:** Accepted
- **Date:** 2026-04-07
- **Deciders:** Project maintainers

---

## Context

views-stepshifter has no canonical, importable definition of which hyperparameters each algorithm requires. The `StepshifterManager` consumes pipeline-level keys (`steps`, `time_steps`) and algorithm-specific `parameters` dicts silently — a missing key surfaces as a `KeyError` deep inside model training, far from the configuration boundary where it belongs.

Downstream in views-models, 20+ stepshifter model configs each declare their own `config_hyperparameters.py` dicts. There is no way for views-models tests to validate these configs statically against what views-stepshifter actually expects, because the contract is implicit — spread across the manager, model `__init__` signatures, and individual algorithm requirements.

views-r2darts2 and views-baseline both solved this problem with a `ReproducibilityGate` class that defines `CORE_GENOME` (params required by all models) and `ALGORITHM_GENOMES` (params required per architecture). This contract is enforced at runtime via `audit_manifest()` and is importable by downstream tests.

---

## Decision

Add a `ReproducibilityGate` class to `views_stepshifter.infrastructure.reproducibility_gate` that serves as the single source of truth for stepshifter hyperparameter requirements.

### Structure

The gate defines:

- **`CORE_GENOME`**: `["steps", "time_steps", "parameters"]` — required by all stepshifter models regardless of algorithm.
- **`ALGORITHM_GENOMES`**: per-algorithm required keys, split into `parameter_keys` (keys inside the `parameters` dict) and `config_keys` (additional top-level keys).

The gate's `audit_manifest(config)` static method validates:
1. All core keys are present.
2. The algorithm is registered.
3. All algorithm-specific parameter and config keys are present.
4. No required key is `None`.

### Integration

`StepshifterManager._train_model_artifact()` calls `ReproducibilityGate.Config.audit_manifest()` as its first operation, before any training begins. The algorithm name is injected from `_config_meta["algorithm"]`.

Violations raise `MissingHyperparameterError` (a subclass of `ReproducibilityError`).

### Importable contract

Downstream packages can validate configs statically:

```python
from views_stepshifter.infrastructure.reproducibility_gate import ReproducibilityGate

CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.Config.ALGORITHM_GENOMES
```

---

## Consequences

**Positive:**

- Missing pipeline-level keys (`steps`, `time_steps`, `parameters`) are caught at the configuration boundary, not deep inside training logic.
- views-models can validate all 20+ stepshifter configs against the gate in CI, catching mismatches before runtime.
- The pattern matches views-r2darts2 and views-baseline `ReproducibilityGate`, providing cross-project consistency.

**Negative:**

- Existing model configs that omit required parameters (e.g. LGBMRegressor configs missing `n_jobs`) will now fail at the gate. This is intentional — it surfaces pre-existing config gaps.

**Scope limitation:**

- The stepshifter gate implements only `Config` validation (no `Temporal` or `Data` gates). These can be added later if the need arises.
- The gate validates key *presence* and *non-None*, not value correctness (types, ranges).

---

## Related

- **Story:** `docs/stories/reproducibility_gate.md`
- **Origin:** views-models risk register C-05
- **Reference implementations:** `views_r2darts2.infrastructure.reproducibility_gate.ReproducibilityGate`, `views_baseline.infrastructure.reproducibility_gate.ReproducibilityGate`
- **CIC:** `docs/CICs/ReproducibilityGate.md`
