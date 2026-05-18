# Technical Risk Register — views-stepshifter

**Last updated:** 2026-04-11
**Governing ADR:** Pending (will be added when base docs are adopted)
**Total entries:** 3
**Concerns:** Open 1 | Invalidated 2

---

## Open Concerns

### D-07 — Darts dependency commented out but code imports darts extensively

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Fresh `poetry install` or CI environment without pre-installed darts |
| **Source** | Tech debt audit (2026-04-07) |
| **Status** | Open |
| **Notes** | `pyproject.toml:18` had `#darts = "^0.38.0"` (commented out). Five import sites across `stepshifter.py:5`, `stepshifter.py:54,57` (lazy), `hurdle_model.py:45` (lazy), and `darts_model.py:2` depend on darts at runtime. Fix: uncomment the dependency. **Likely already resolved** by the `bump darts to ^0.40.0` commit on `fix/darts04` — verify on next audit and close if confirmed. |

---

## Invalidated Concerns

### D-01 — Destructive config mutation in `_get_model()` destroys HP dict — INVALIDATED

| Field | Value |
|---|---|
| **Tier** | — |
| **Original trigger** | Any non-hurdle, non-shurf model is trained via `StepshifterManager._train_model_artifact()` |
| **Source** | Tech debt audit (2026-04-07); falsified by review-diff audit (2026-04-11) |
| **Status** | Invalidated (2026-04-11) |
| **Notes** | **Original claim (WRONG):** `stepshifter_manager.py:92` does `self.configs = {"model_reg": self.configs["algorithm"]}`, replacing the entire HP config dict with a single key and causing `KeyError: 'steps'` in `StepshifterModel.__init__`. **Why the claim was wrong:** The `configs` setter on `ModelManager` (base class in `views_pipeline_core/managers/model/model.py:376-410`) is **not** a replacement — it delegates to `ConfigurationManager.add_config()` (configuration.py:581-634) which performs `self._runtime_config.update(config)`. Assigning `self.configs = {"model_reg": ...}` **merges** the key into `_runtime_config`, leaving all other keys (`steps`, `time_steps`, `parameters`, etc.) intact. Conversely, the "pre-migration" pattern `self.configs["key"] = value` would mutate the ephemeral dict returned by `get_combined_config()` and would be a **no-op** — the fix documented in the original entry would have silently dropped `model_reg`. **Resolution:** No code change required. The existing implementation is correct. A strengthened `test_get_model` test in `tests/test_stepshifter_manager.py` now asserts that `steps`/`time_steps`/`parameters`/`model_reg` all coexist in `self.configs` after `_get_model` returns, documenting the merge semantics. |

---

### D-02 — Destructive config mutation in eval/forecast methods — INVALIDATED

| Field | Value |
|---|---|
| **Tier** | — |
| **Original trigger** | Future code that reads `self.configs` after `_evaluate_model_artifact` / `_forecast_model_artifact` returns |
| **Source** | Tech debt audit (2026-04-07); falsified by review-diff audit (2026-04-11) |
| **Status** | Invalidated (2026-04-11) |
| **Notes** | **Original claim (WRONG):** `stepshifter_manager.py:164` and `:207` do `self.configs = {"timestamp": path_artifact.stem[-15:]}`, replacing the entire config dict. **Why the claim was wrong:** Same root cause as D-01 — the `configs` setter merges via `ConfigurationManager.add_config()`. The assignment adds a `timestamp` key to `_runtime_config` without disturbing any other config source (hyperparameters, deployment, meta, partitions). Parent class reads of `self.configs` after these methods return get the fully-merged config including the new timestamp. **Resolution:** No code change required. The current implementation is the correct idiom for adding runtime config values. |
