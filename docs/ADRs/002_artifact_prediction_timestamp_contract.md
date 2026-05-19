# ADR-002: Artifact-Prediction Timestamp Contract

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Simon, VIEWS platform team
**Consulted:** views-pipeline-core ADR-052 (central contract)

---

## Context

The VIEWS pipeline requires that prediction filenames carry the timestamp of the **trained model artifact** that produced them, not the wall-clock time of prediction generation (views-pipeline-core ADR-013, ADR-052). This contract enables:

1. **Traceability:** Any prediction file can be linked back to the exact artifact that produced it.
2. **Ensemble resolution:** The ensemble manager matches constituent predictions to artifacts by timestamp. A mismatch causes silent fallback to subprocess re-execution or failure.

A violation of this contract was discovered in views-baseline in May 2026, where `datetime.now()` overwrote the artifact timestamp. An audit of all model-specific repos confirmed that views-stepshifter implements the contract correctly.

This ADR documents the existing correct implementation for the record and to ensure it is preserved during future refactors.

---

## Decision

`StepshifterManager` (in `views_stepshifter/manager/stepshifter_manager.py`) correctly implements the artifact-prediction timestamp contract:

- `_evaluate_model_artifact()` resolves the latest artifact path and extracts the 15-character timestamp from the filename stem.
- The timestamp is persisted via `self.configs = {"timestamp": ...}`, which invokes the property setter on `ForecastingModelManager` and delegates to `self._config_manager.add_config(...)`. This is equivalent to calling `add_config()` directly (see ADR-052 for both valid forms).
- `_forecast_model_artifact()` follows the same pattern.

No code changes are required. This ADR is documentation-only.

---

## Rationale

- **Defensive documentation.** The contract is subtle (the `config` property trap makes the wrong approach silently fail). Documenting the correct implementation guards against regression during refactors.
- **Cross-repo consistency.** All model-specific repos now have a local ADR referencing the central contract in views-pipeline-core ADR-052.

---

## Considered Alternatives

Not applicable — the implementation is already correct. This ADR documents existing behavior.

---

## Consequences

### Positive

- The timestamp contract is documented in this repo, reducing the risk of accidental regression.
- Future contributors can reference this ADR when modifying artifact loading or prediction generation.

### Negative

- None.

---

## Implementation Notes

- **No code changes required.**
- **Actual pattern** (from `stepshifter_manager.py:160,203`):
  ```python
  self.configs = {"timestamp": path_artifact.stem[-15:]}
  ```
  This uses the `configs` property setter, which delegates to `self._config_manager.add_config(...)`. Both forms are valid per ADR-052.
- **Key invariant:** No method in the evaluate/forecast flow should overwrite the timestamp with a value other than the one extracted from the artifact filename.

---

## Validation & Monitoring

- **Existing tests:** The stepshifter test suite validates the evaluate and forecast flows.
- **Failure signal:** Ensemble evaluation failing with "prediction file not found" for stepshifter constituents would indicate a regression.

---

## Open Questions

- None.

---

## References

- views-pipeline-core ADR-052: Artifact-Prediction Timestamp Contract (central)
- views-pipeline-core ADR-013: Prediction Naming Convention
- views-hydranet ADR-026: Model Artifact Fetcher Specification
- views-baseline ADR-016: Artifact-Prediction Timestamp Contract (bugfix record)
