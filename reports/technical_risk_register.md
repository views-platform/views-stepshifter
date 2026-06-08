# Technical Risk Register — views-stepshifter

**Last updated:** 2026-06-08
**Governing ADR:** Pending (will be added when base docs are adopted)
**Total entries:** 15
**Concerns:** Open 12 | Resolved 1 | Invalidated 2

> **ID convention:** This register uses the `D-xx` (Debt) prefix for all concern entries; there are no disagreement entries. IDs are permanent and sequential.

---

## Open Concerns

### D-08 — ShurfModel appends the same mutable `self._models` dict for every submodel

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When evaluating a ShurfModel run for ensemble spread, or when refactoring `ShurfModel.fit`, verify each element of `_submodel_list` is a distinct object. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/models/shurf_model.py:55-71` |
| **Notes** | `fit` does `self._submodel_list.append(self._models)` (`:71`) inside the submodel loop while reusing/overwriting the single instance dict `self._models[step] = (binary_model, positive_model)` (`:69`). All appended references may point to the same dict, so every "submodel" sampled in `predict_sequence` could be the last-trained one — collapsing the ensemble's draw diversity to a single model with no error signal. Not Tier 1 only because the failure surfaces as degenerate uncertainty bands rather than confirmed corruption (unverified without a test). The defect is hidden by the total absence of ShurfModel tests — see also D-13. |

---

### D-09 — ReproducibilityGate audits an unused key (`time_steps`) but not keys the models dereference (`model_clf`/`model_reg`)

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When adding or editing a Hurdle/Shurf `config_hyperparameters.py` in views-models, verify the gate's `ALGORITHM_GENOMES` actually covers the top-level keys (`model_clf`, `model_reg`) that `fit` reads. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/infrastructure/reproducibility_gate.py:25,30-57`; `views_stepshifter/models/hurdle_model.py:95-96`; `views_stepshifter/models/shurf_model.py:40-41` |
| **Notes** | `CORE_GENOME` requires `time_steps`, which no model class reads (`StepshifterModel.__init__` consumes `steps`, not `time_steps`). Conversely, `ALGORITHM_GENOMES["HurdleModel"]`/`["ShurfModel"]` declare only `parameter_keys: ["clf","reg"]` and do **not** require the top-level `model_clf`/`model_reg` keys that `HurdleModel.fit`/`ShurfModel.fit` dereference. A Hurdle config missing `model_clf` therefore passes `audit_manifest` and then raises `KeyError` deep inside `fit` — exactly the late-failure mode ADR-001 was created to eliminate. The gate gives a false sense of completeness. |

---

### D-10 — Gate audit runs only on the train path; evaluate/forecast bypass it

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When relying on the reproducibility gate to catch a malformed runtime config during an `_evaluate_model_artifact` or `_forecast_model_artifact` run (the gate is invoked only in `_train_model_artifact`). |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/manager/stepshifter_manager.py:104-108` (audit present) vs `:129`, `:175` (absent) |
| **Notes** | `ReproducibilityGate.Config.audit_manifest()` is the first operation of `_train_model_artifact` only. The evaluate and forecast paths reconstruct config and inject `timestamp` without re-auditing. Severity is bounded because those paths unpickle a model whose config is already baked in, but the contract is asymmetric and the asymmetry is undocumented. See also D-02 (same methods, different — invalidated — concern). |

---

### D-11 — `views_validate` selects the DataFrame by hardcoded positional index; predict paths are undecorated

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When adding a keyword argument to or changing the positional signature of `fit`/`predict` in any model class, verify `views_validate` still resolves the DataFrame (it inspects `args[0]`/`args[1]`). |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/models/validation.py:38,51` |
| **Notes** | `dataframe = args[0] if isinstance(args[0], pd.DataFrame) else args[1]` — the author's inline comment reads "Hardcoding this makes me uncomfortable" and the function header says "Needs update". A signature change that shifts the df's positional slot would validate the wrong object or crash. Additionally, only `fit` is decorated; `predict` methods receive no schema validation, so malformed inputs at predict time fail deep in Darts rather than at the boundary (invariant I-1). |

---

### D-12 — ShurfModel hardcodes `["month_id","country_id"]` grouping, breaking PRIO-grid runs

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When configuring a ShurfModel run on PRIO-grid data (second index level `priogrid_gid`/`priogrid_id`). |
| **Source** | repo-assimilation (2026-06-08) |
| **Location** | `views_stepshifter/models/shurf_model.py:229` |
| **Status** | Open |
| **Notes** | `final_preds_full.groupby(["month_id", "country_id"])` is hardcoded to the country level, whereas the rest of the package supports PRIO-grid levels via `self._level` (`stepshifter.py:_process_data`) and `validation.py:10` (`priogrid_gid`/`priogrid_id`). A PRIO-grid ShurfModel run would group on a nonexistent column and raise `KeyError`. The other models parameterise on `self._level`; ShurfModel alone hardcodes it. |

---

### D-13 — Model tests verify orchestration wiring, not numerical behavior; ShurfModel untested

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When modifying real fit/predict numerical logic (diagonal slicing in `_predict_by_step`, Hurdle binary×positive multiplication, Shurf sampling) and relying on the existing suite to catch the regression. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `tests/test_stepshifter.py`, `tests/test_hurdle_model.py`; untested: `views_stepshifter/models/shurf_model.py`, `views_stepshifter/models/darts_model.py` |
| **Notes** | `test_stepshifter.py` and `test_hurdle_model.py` patch `ProcessPoolExecutor`, `tqdm`, `as_completed`, and the model classes, so they assert wiring/call-counts but never exercise real Darts fitting, the diagonal "last-month-with-data" prediction, or Hurdle's binary×positive product on real data — the suite would stay green if the integration broke. `ShurfModel` (276 lines: sampling, explode, distribution draws) and `darts_model.py` have **zero** tests. Both files also carry large dead commented-out assertion blocks (`hurdle_model.py` test `:93-117`, `:200-215`). See also D-08 (the ShurfModel mutable-dict bug this coverage gap conceals). |

---

### D-14 — Version, dist artifact, and lockfile drift undermine reproducible builds

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When cutting a release or reproducing the environment from a fresh clone / CI run. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `pyproject.toml:3` (version `1.2.0`); `dist/views_stepshifter-0.1.2*` (stale 0.1.2 wheels); `README.md:151` (lists `poetry.lock`); `.github/workflows/run_pytest.yml:33` (`poetry install`, no lockfile) |
| **Notes** | `pyproject.toml` declares `version = "1.2.0"` while the committed `dist/` artifacts are `0.1.2`. The README "Project Structure" lists `poetry.lock`, but no lockfile exists in the repo, so CI's `poetry install` resolves dependencies unpinned on every run — a reproducibility hazard only partially mitigated by hard pins on `lightgbm` (4.6.0) and `scipy` (1.15.1). Build cruft should also not be committed. See also D-07 (related pyproject dependency hygiene). |

---

### D-15 — Committed `MagicMock/` log tree is test cruft tracked in the repo

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | When auditing tracked files or packaging the repo; and when a test that derives a logging path from a `MagicMock` runs, to prevent re-accumulation. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `MagicMock/mock.logging/*/views_pipeline_*.log` (dozens of directories) |
| **Notes** | A test run with a mocked logging path wrote real `views_pipeline_{INFO,DEBUG,WARNING,ERROR,CRITICAL}.log` files under a literal `MagicMock/` directory and they were committed. Pure repository noise with no correctness impact, but it indicates a test writing to a path derived from a `MagicMock` repr and a missing `.gitignore` entry. |

---

### D-16 — `_get_standardized_df` silently clamps all negative predictions to zero

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When a stepshifter/hurdle/shurf model is applied to a target where negative predictions are meaningful (not a zero-bounded count). |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/manager/stepshifter_manager.py:39-47` |
| **Notes** | `standardize_value` maps `inf`, `-inf`, `nan`, and any value `< 0` to `0`, applied unconditionally via `applymap` to every prediction frame returned by evaluate/forecast. For conflict-fatality counts this is domain-correct (counts cannot be negative), so it is **not** currently corrupting output — hence Tier 3 not Tier 1 — but the clamp is silent and unflagged, so any future reuse on a signed target would zero legitimate negatives with no signal. Secondary: `DataFrame.applymap` is deprecated in pandas ≥2.1, a forward-compat snag. |

---

### D-17 — Frozen artifacts trained in the `5fcfe43` log-space window may emit silently mis-scaled predictions

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When evaluating or forecasting from a stepshifter artifact whose training predates the `target_transform` mechanism — notably the `big_chungus` constituents — confirm the scale it was trained in matches the raw-output contract. |
| **Source** | expert-review (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/manager/stepshifter_manager.py:163,207` (unpickle + predict); `views_stepshifter/models/stepshifter.py:_predict_by_step` (no inverse); ADR-003 Obligations |
| **Notes** | Commit `5fcfe43` (2025-11-20 → reverted 2026-04-11 by `08ee2eb`) forced stepshifter training into log space via `np.log1p`/`np.expm1`. An artifact pickled during that window carries log-space behavior; loaded under current raw-output code it receives **no** `expm1`, so predictions stay log-compressed but are written as raw `lr_ged_sb` — silently wrong, with **no error signal**. The 2026-06-04 `big_chungus` ensemble (MSLE 2.519, **worse than the all-zeros baseline 2.147**, ~3× the fatalities002 gold standard 0.835) is the suspected materialization; the stepshifter-family constituents are broken while the four r2darts2 DL constituents (own asinh transform) are fine. **Escalates to Tier 1** if the views-models artifact audit confirms any in-production constituent was trained in a non-raw space. ADR-003 mandates load-time guard E2 (fail-loud on a missing transform stamp), but the guard is unimplemented and the artifact audit is open. See also D-16 (separate silent prediction-output transform in the same manager path) and D-13 (numerical behavior untested). Root cause documented in `docs/ADRs/003_raw_target_space_io_contract.md`. |

---

### D-18 — Raw-I/O contract is unfalsifiable in CI until guards E1/E2 and the gate key are implemented

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When a contributor reintroduces or adds an internal target transform to `StepshifterModel` before the ADR-003 characterization test (E1), load-time guard (E2), and the gate `target_transform` key are implemented. |
| **Source** | expert-review (2026-06-08) |
| **Status** | Open |
| **Location** | ADR-003 §"Enforcement required at ratification"; `views_stepshifter/infrastructure/reproducibility_gate.py:24` (`CORE_GENOME` lacks `target_transform`); no characterization test in `tests/` |
| **Notes** | ADR-003 ratifies a raw-in/raw-out contract but is documentation-only until guards E1 (fit→predict characterization test asserting raw-space output) and E2 (load-time transform-name guard) ship and `target_transform` is added to the gate. Until then, a future silent-transform commit — a `5fcfe43` rhyme — passes CI green, which is exactly the regression that produced D-17. The fix is cheap (E1 needs no registry) and should not wait for the full declarative `TRANSFORMS` mechanism. See also D-09 (gate omits keys models dereference), D-10 (gate runs only on the train path), and D-13 (existing tests verify wiring, not numerical behavior). |

---

### D-19 — ADR-003 depends on a not-yet-ratified views-pipeline-core platform ADR (dangling `⟨PENDING⟩` reference)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When views-pipeline-core ratifies (or decides not to ratify) the platform transform/prefix ADR — reconcile ADR-003's `⟨PENDING⟩` reference and its prefix-deprecation / raw-IO assumptions against the ratified text. |
| **Source** | expert-review (2026-06-08) |
| **Status** | Open |
| **Location** | `docs/ADRs/003_raw_target_space_io_contract.md` (`⟨PENDING — views-pipeline-core ADR-XXX⟩` placeholder + clause 6) |
| **Notes** | ADR-003 scopes its own authority to stepshifter but cites a forthcoming **platform-wide** views-pipeline-core ADR as the body that will ratify (a) model-internal transforms with raw I/O, (b) deprecation of the `lr_`/`ln_`/`lx_` prefix scheme, and (c) config-as-source-of-truth. That ADR does not exist yet; the platform rule currently lives only in pipeline-core **plans** (`2026-03-15`, `2026-06-01` C-140), not a ratified ADR. If it stalls, ADR-003 holds an unbacked cross-repo claim and the `ADR-XXX` reference stays dangling; if it ratifies with different semantics (e.g. keeps the prefix scheme), ADR-003 must be amended. Closes when the platform ADR is ratified and its number is filled into the placeholder. |

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

---

## Resolved Concerns

### D-07 — Darts dependency commented out but code imports darts extensively — RESOLVED

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Original trigger** | Fresh `poetry install` or CI environment without pre-installed darts |
| **Source** | Tech debt audit (2026-04-07); confirmed resolved by repo-assimilation (2026-06-08) |
| **Status** | Resolved (2026-06-08) |
| **Resolution** | The dependency is now declared: `pyproject.toml:18` reads `darts = "^0.40.0"` (uncommented), satisfying the five runtime import sites (`stepshifter.py:5`, `stepshifter.py:54,57` lazy, `hurdle_model.py:45` lazy, `darts_model.py:2`). The original entry flagged this as "likely already resolved" pending verification; verified on the 2026-06-08 assimilation audit. |
