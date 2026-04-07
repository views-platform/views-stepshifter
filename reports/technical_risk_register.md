# Technical Risk Register — views-stepshifter

**Last updated:** 2026-04-07
**Governing ADR:** Pending (will be added when base docs are adopted)
**Total entries:** 3
**Concerns:** Open 2 | Accepted 1

---

## Open Concerns

### D-01 — Destructive config mutation in `_get_model()` destroys HP dict for regression models

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any non-hurdle, non-shurf model is trained via `StepshifterManager._train_model_artifact()` |
| **Source** | Tech debt audit (2026-04-07) |
| **Status** | Open |
| **Notes** | `stepshifter_manager.py:92` does `self.configs = {"model_reg": self.configs["algorithm"]}`, replacing the entire HP config dict with a single key. `StepshifterModel.__init__` then crashes with `KeyError: 'steps'` at `stepshifter.py:21` because `steps`, `targets`, `parameters`, and all other keys are gone. This is a regression from the `self.config` -> `self.configs` migration on the `feature/reproducibilitygate` branch: the old code did `self.config["model_reg"] = self.config["algorithm"]` (key insertion, non-destructive), the new code does `self.configs = {"model_reg": ...}` (full replacement, destructive). Fix: change to `self.configs["model_reg"] = self.configs["algorithm"]` to restore the original non-destructive semantics. Deferred because it requires integration testing with the full pipeline to verify no downstream assumptions changed. |

---

### D-07 — Darts dependency commented out but code imports darts extensively

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Fresh `poetry install` or CI environment without pre-installed darts |
| **Source** | Tech debt audit (2026-04-07) |
| **Status** | Open |
| **Notes** | `pyproject.toml:18` has `#darts = "^0.38.0"` (commented out). Five import sites across `stepshifter.py:5`, `stepshifter.py:54,57` (lazy), `hurdle_model.py:45` (lazy), and `darts_model.py:2` depend on darts at runtime. Darts is not pulled transitively — `views_forecasts` does not depend on it (verified from dist-info metadata). The package cannot be installed from its own `pyproject.toml` without manually uncommenting line 18 or pre-installing darts by other means. The comment was introduced in a prior commit (likely working around a darts version compatibility issue). Fix: uncomment the dependency. Deferred because uncommenting may reintroduce whatever compatibility issue motivated commenting it out. |

---

## Accepted Concerns

### D-02 — Destructive config mutation in eval/forecast methods clobbers config state

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Future code changes that read `self.configs` after `_evaluate_model_artifact()` or `_forecast_model_artifact()` completes |
| **Source** | Tech debt audit (2026-04-07) |
| **Status** | Accepted |
| **Notes** | `stepshifter_manager.py:164` and `stepshifter_manager.py:207` both do `self.configs = {"timestamp": path_artifact.stem[-15:]}`, replacing the entire config dict with a single timestamp key. Currently safe: `run_type` is extracted into a local variable before the mutation (line 147/190), and nothing reads `self.configs` after these methods return. The parent class `ForecastingModelManager.execute_single_run()` uses the returned predictions directly. Accepted because the current code path is safe and the mutation happens at method-lifecycle end. Any future code adding post-eval config reads should be aware of this pattern. |
