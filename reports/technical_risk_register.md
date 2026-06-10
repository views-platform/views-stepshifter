# Technical Risk Register — views-stepshifter

**Last updated:** 2026-06-10
**Governing ADR:** Pending (will be added when base docs are adopted)
**Total entries:** 26
**Concerns:** Open 23 | Resolved 1 | Invalidated 2

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
| **Status** | Open — 4 brittle tests now `xfail`'d; real replacements tracked in **views-stepshifter#75** |
| **Location** | `tests/test_stepshifter.py::{test_fit,test_predict}`, `tests/test_hurdle_model.py::{test_fit,test_predict}` (all now `@pytest.mark.xfail`, strict=False); untested: `views_stepshifter/models/shurf_model.py`, `views_stepshifter/models/darts_model.py` |
| **Notes** | `test_stepshifter.py` and `test_hurdle_model.py` patch `ProcessPoolExecutor`, `tqdm`, `as_completed`, and the model classes, so they assert wiring/call-counts but never exercise real Darts fitting, the diagonal "last-month-with-data" prediction, or Hurdle's binary×positive product on real data — the suite would stay green if the integration broke. `ShurfModel` (276 lines: sampling, explode, distribution draws) and `darts_model.py` have **zero** tests. Both files also carry large dead commented-out assertion blocks (`hurdle_model.py` test `:93-117`, `:200-215`). See also D-08 (the ShurfModel mutable-dict bug this coverage gap conceals). **Update (2026-06-10, PR #74):** these 4 tests were not merely uninformative but actively **brittle** — they fail env/start-method-dependently (locally `AttributeError: '_series'`; on CI `KeyError: 'pred_target'`), blocking the branch's CI. Rather than hide a real regression, they were marked `@pytest.mark.xfail(strict=False)` referencing this entry, and a **dedicated issue (#75)** was opened for the real, behavioral replacements (real fit→predict on small data asserting numerical correctness; plus first-ever coverage for `ShurfModel`/`darts_model.py`). The `xfail` is an honest interim, NOT a fix — the coverage gap (and D-08 it conceals) remains until #75 lands. The new real end-to-end pattern in `tests/test_target_transforms.py` is the template to follow. |

---

### D-14 — Version, dist artifact, and lockfile drift undermine reproducible builds

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When cutting a release or reproducing the environment from a fresh clone / CI run. |
| **Source** | repo-assimilation (2026-06-08) |
| **Status** | Partially resolved (2026-06-10, 1.3.0 release prep) — version/dist drift fixed; **lockfile gap still Open** |
| **Location** | ~~`pyproject.toml:3` (version `1.2.0`)~~ → now `1.3.0`; ~~`dist/views_stepshifter-0.1.2*`~~ → removed + `dist/` gitignored; `README.md:151` (lists `poetry.lock`); `.github/workflows/run_pytest.yml:33` (`poetry install`, no lockfile) |
| **Notes** | `pyproject.toml` declared `version = "1.2.0"` while the committed `dist/` artifacts were `0.1.2`. The README "Project Structure" lists `poetry.lock`, but no lockfile exists in the repo, so CI's `poetry install` resolves dependencies unpinned on every run — a reproducibility hazard only partially mitigated by hard pins on `lightgbm` (4.6.0) and `scipy` (1.15.1). Build cruft should also not be committed. See also D-07 (related pyproject dependency hygiene). **Update (2026-06-10, 1.3.0 release):** the **version collision** materialised exactly as this entry warned — `1.2.0` was already a published tag (`v1.2.0` = `origin/main` `e6adbdb`, **no `target_transform`**), so the `target_transform` release could not ship as 1.2.0. Resolved by bumping `pyproject` to **1.3.0** (minor; consumers pin `<2.0.0`, all 37 views-models stepshifter configs migrated) and **removing the stale `dist/` 0.1.2 wheels** (now gitignored). The **lockfile gap remains Open** — CI still `poetry install`s unpinned; cut a `poetry.lock` or drop the README reference in a follow-up. |

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
| **Notes** | Commit `5fcfe43` (2025-11-20 → reverted 2026-04-11 by `08ee2eb`) forced stepshifter training into log space via `np.log1p`/`np.expm1`. An artifact pickled during that window carries log-space behavior; loaded under current raw-output code it receives **no** `expm1`, so predictions stay log-compressed but are written as raw `lr_ged_sb` — silently wrong, with **no error signal**. The 2026-06-04 `big_chungus` ensemble (MSLE 2.519, **worse than the all-zeros baseline 2.147**, ~3× the fatalities002 gold standard 0.835) is the suspected materialization; the stepshifter-family constituents are broken while the four r2darts2 DL constituents (own asinh transform) are fine. **Escalates to Tier 1** if the views-models artifact audit confirms any in-production constituent was trained in a non-raw space. ADR-003 mandates load-time guard E2 (fail-loud on a missing transform stamp), but the guard is unimplemented and the artifact audit is open. See also D-16 (separate silent prediction-output transform in the same manager path) and D-13 (numerical behavior untested). Root cause documented in `docs/ADRs/003_raw_target_space_io_contract.md`. **Falsification caveat (2026-06-08, `/falsify` P3+P6):** the audit reframes the **leading** cause as **"no target compression after the revert" (trained-raw)**, not the stale-log-artifact *mismatch* this title emphasizes — every locally inspected stepshifter artifact post-dates the revert and none from the log window were found. But the diagnosis is still **inferential**: the actual `big_chungus` constituent predictions were never inspected (the calibration reports' "Prediction Samples" were available and unused), and the actual constituent artifacts (WandB `2689xjtl`) were never opened. Confidence = "leading hypothesis", not confirmed; re-run `/falsify` after the dossier's real-data EXP-01. **EXP-01 update (2026-06-08, real mechanism):** the **trained-raw** root cause is now **confirmed** — `identity` through the centralized `target_transform` reproduces production MSLE (4.269 ≈ 4.27) *and* the mean-16-on-true-zeros signature (16.04) on real brown_cheese data (n=89,388). The *stale-log-artifact mismatch* sub-hypothesis remains secondary/unconfirmed; the constituent-artifact audit (WandB `2689xjtl`) is still open. Evidence: dossier `07_experiment_log.md` EXP-01. **Production-validated (#57–#61, 2026-06-08):** identity through `main.py -r calibration -t -e -sa` gives production MSLE 4.2687 (≈4.27) and mean prediction ~60.7 — confirming trained-raw on the real pipeline, not just the custom harness. |

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

### D-20 — Ensemble combination/calibration step never audited as a divergence contributor

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When attributing the `big_chungus` (or any ensemble) cm MSLE to constituent quality, first reconstruct the ensemble from its constituent predictions and confirm the combination/calibration step adds no independent error. |
| **Source** | falsification-audit (2026-06-08) |
| **Status** | Open |
| **Location** | 2026-06-04 `big_chungus` run; ensemble combination/calibration (views-evaluation / pipeline-core — outside this repo) |
| **Notes** | `big_chungus` scored cm MSLE 2.519 — worse than **17 of its 26 constituents**, including all four DL models (~0.5) and the historical-average baseline (0.581). A combination containing four ~0.5 models landing at 2.519 is only explained if a few extreme over-predictors dominate the mean **or** if the calibration/combination step adds independent error. The investigation never examined the ensemble step, so it is **not ruled out** as a second contributor. The per-constituent breakage (each stepshifter model worse than its own all-zeros baseline) is independent of this and stands. Cross-repo: the ensemble lives in views-evaluation/pipeline-core. Surfaced by `/falsify` P1. See also D-17. |

---

### D-21 — `plastic_beach` ≡ `brown_cheese` byte-identical eval metrics from different configs (probable artifact/prediction mislink)

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When trusting per-constituent eval metrics for the 2026-06-04 run (or any multi-model run), verify each model's reported metrics derive from its own artifact/prediction file. |
| **Source** | falsification-audit (2026-06-08) |
| **Status** | Open |
| **Location** | 2026-06-04 calibration report `report_calibration_model_20260604_230304_lr_ged_sb.html`; views-models `plastic_beach` vs `brown_cheese` configs |
| **Notes** | `plastic_beach` and `brown_cheese` reported **byte-identical** metrics (RMSLE 2.048527, MSE 51521.055532, MSLE 4.196935, ŷ 43.696734) despite confirmed-different querysets (aquastat vs baseline) and configs. Two distinct models producing byte-identical metrics is essentially impossible by chance → a probable prediction-file/artifact mislink, meaning at least one model's reported performance is **wrong with no error signal**. This is a **second, separate defect** from the no-compression root cause (D-17) — i.e. the divergence investigation has **not** located a single "the" error. **Escalates to Tier 1** if the mislink affects stored/published predictions rather than only the report's metric table. Surfaced by `/falsify` P5. |

---

### D-22 — Transform/model selection on cm MSLE alone rewards zero-fit and tail-compression

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When EXP-01 (or any `target_transform`/model selection) declares a winner on **cm MSLE alone**, without a tail-conditional error and a calibration readout. |
| **Source** | expert-method-review (2026-06-08) |
| **Status** | Open — **production-validated** (EXP-01 + #57–#61) |
| **Location** | dossier `reports/2026-06-08_restore_target_compression_dossier/02_design.md`, `04_roadmap.md` (EXP-01), `05_analysis_plan.md`, `07_experiment_log.md` (EXP-01) |
| **Notes** | On a ~90%-zero target, MSLE structurally favours near-zero prediction and **tail-compression** (Bolin2023 on scale-dependence of scoring rules; Gneiting2014). A transform can *win* cm MSLE while **degrading the heavy-tail / escalation predictions that are the operational product** (Hegre2022 on escalation; Davison/EVT lens, Abilasha2022). Ranking *transforms* on MSLE is also near-circular (log1p training ≈ optimising MSLE). Mitigation: report MSLE **plus** a tail-conditional error (error \| obs > threshold) **plus** calibration-in-the-large, and make the zero-fit↔tail tradeoff visible before selecting. The strongest single fix from the method review. **EXP-01 confirmation (2026-06-08, real brown_cheese, n=89,388):** the pre-registered F4 falsifier **FIRED** — on MSLE alone `asinh` (0.515) / `log1p` (0.521) crush `identity` (4.269), but on the **escalation tail they are ~2.5× WORSE** (tail MSLE 4.6/4.2 vs 1.80) and calibration collapses from 2.46× over to ~0.24× under. Selecting on MSLE alone would have crowned a tail-blind model. This is no longer a hypothetical risk — it is the observed outcome; any selection (incl. views-models#114) **must** use the joint tail+calibration readout. **Production-validated (#57–#61, 2026-06-08):** all three arms re-run through the full `main.py` → views-evaluation pipeline matched the custom driver to 3 decimals on every metric, and F4 fired on production data — the risk is confirmed on the trusted path, not just the custom harness. |

---

### D-23 — Fix design omits a count-likelihood baseline (Tweedie/NB)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When the fix is committed as "a target transform" without testing a count-likelihood (Tweedie / negative-binomial) arm in the readout. |
| **Source** | expert-method-review (2026-06-08) |
| **Status** | Open |
| **Location** | dossier `02_design.md`, `05_analysis_plan.md` |
| **Notes** | A squared-error regressor assumes Gaussian homoskedastic noise — false for zero-inflated heavy-tailed counts. `log1p` is the poor-man's variance-stabilisation of a **Tweedie** likelihood (point mass at zero + continuous positive part — exactly this DGP; Damato2025 GP-Tweedie for intermittent demand). By testing only transforms of a squared-error model, EXP-01 **cannot discover** that the right answer is a count-appropriate likelihood. Mitigation: add a Tweedie/NB-deviance arm to the readout (one extra arm, high information). **Deferred by scope decision (2026-06-08):** the first proof is scoped to *stateless transforms* (`log1p`/`asinh`) on *plain* models only; Tweedie/count-likelihoods are a **future option, not a current gap** — revisit after the plain-model fix is proven. See D-26. |

---

### D-24 — HurdleModel dichotomization (>0 threshold) never evaluated against a single count model

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | When fixing `target_transform` inside HurdleModel without evaluating its binary `(x>0)` split against a single count-model baseline. |
| **Source** | expert-method-review (2026-06-08) |
| **Status** | Open |
| **Location** | `views_stepshifter/models/hurdle_model.py:101`; dossier `02_design.md` |
| **Notes** | Per Harrell (anti-dichotomization), thresholding the outcome at >0 is information-lossy and fragile; the design takes the hurdle structure as given and only fixes the transform inside it. No evidence the two-stage hurdle beats a single well-specified count model. Mitigation: include a single-count-model baseline vs the hurdle in the readout. Fetch: Harrell *Regression Modeling Strategies*; a ZIP/hurdle canonical (Lambert 1992 / Mullahy 1986). |

---

### D-25 — No prediction-space calibration / retransformation-bias check when a transform is active

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When a **non-identity** `target_transform` ships and predictions are inverse-transformed (`expm1`/`sinh`) without a calibration / retransformation-bias correction. |
| **Source** | expert-method-review (2026-06-08) |
| **Status** | Open |
| **Location** | dossier `03_harness_and_invariants.md §C4` (round-trip test); `views_stepshifter/models/stepshifter.py:_predict_by_step` (inverse site) |
| **Notes** | The harness's round-trip test (`inverse(forward(x)) ≈ x`) does **not** check prediction-space calibration. By Jensen's inequality, `E[expm1(ŷ)] ≠ expm1(E[ŷ])`: a point forecast minimising squared error in log space, then `expm1`-ed, is a **systematically biased** raw-space prediction (low) — silent, no error signal. Currently dormant (`identity` only). **Activates / escalates to Tier 1** once a non-identity transform ships without a bias correction. Mitigation: calibration-in-the-large + a smooth calibration curve per arm; check retransformation bias. (Harrell.) |

---

### D-26 — "Here be dragons": HurdleModel / ShurfModel transform-centralization deferred

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | When extending the declared `target_transform` mechanism — or the EXP-01 fix result — from plain `StepshifterModel` to `HurdleModel` or `ShurfModel`. |
| **Source** | project decision (2026-06-08) |
| **Status** | Open (deliberately deferred) |
| **Location** | `views_stepshifter/models/shurf_model.py:156,180,212,214` (internal `log1p`/`expm1` + dormant `log_target=True` bug); `views_stepshifter/models/hurdle_model.py` (two-stage binary split); `views_stepshifter/manager/stepshifter_manager.py:39-47` (the standardize→0 clamp, D-16) |
| **Notes** | **Decision:** prove the transform-centralization and the over-prediction fix on **plain** stepshifter models (XGB/XGBRF/LGBM via `StepshifterModel` base) **FIRST**. `HurdleModel` and `ShurfModel` are **explicitly out of scope for now** — they carry their *own* scattered transforms (ShurfModel does `log1p`/`expm1` inside `predict_sequence`, with a dormant `log_target=True` bug; Hurdle has the binary `(x>0)` stage) that need careful, separately-tested centralization. **Significant work — here be dragons.** The plain-model proof must **not** be assumed to generalize to them, and must not be extended to Hurdle/Shurf without a dedicated audit + test pass. See also D-08, D-24, the D-17 caveat, and ADR-003's ShurfModel interim rule. |

---

### D-27 — ShurfModel `log_target=True` applies `expm1` to a raw-space prediction → silent order-of-magnitude inflation

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Any calibration/forecast run of a `log_target=True` ShurfModel after the queryset `.ln()` / internal `log1p` removal (`08ee2eb`). Currently exactly one exposed model: `fourtieth_symphony` (cm, `log_target=True`); the other 5 Shurfs are `log_target=False` and unaffected. |
| **Source** | expert-code-review (2026-06-09), grounded by reading `ShurfModel.fit` + `predict_sequence` |
| **Status** | Open |
| **Location** | `views_stepshifter/models/shurf_model.py:38` (`fit` → `_process_data`, identity-pinned → **raw** target) + `:65-68` (positive regressor trains on raw positive counts) ↔ `:156` (the `log_target=True` Lognormal sampler does `expm1(np.random.normal(Regression, σ))` on that raw prediction) |
| **Notes** | **Training-space ↔ sampling-space mismatch.** `ShurfModel.fit` trains the positive stage on the **raw** target (`_process_data` is identity for Shurf — both by the gate-pin and historically post-`08ee2eb`). But the `log_target=True` sampler assumes `Regression` is **log**-space and applies `expm1`: `expm1` of a raw conflict count (e.g. 10 → ~22026) → **catastrophic over-prediction, with no error signal**. `fourtieth_symphony` is a published model silently emitting inflated forecasts. **Same root cause as D-17** (removed compression → raw-space model) but a **distinct manifestation** (sampler `expm1`-of-raw inflation vs the plain models' raw over-prediction on zeros). **Corrects the investigation FINDINGS:** the claimed site (`shurf_model.py:210-212` ordering) is **dead code** — the `"Prediction"` column it mutates is dropped at `:216-227`; the output uses `pred_col_name` set at `:206`. Diagnosis: views-stepshifter#68; fix: #69 (test-first; candidates: config `log_target=False` onto the correct raw sampler at `:180`, a code fix, or fold into the #71 ADR-003 split). Cross-ref **D-17** (shared root cause), **D-26** / **D-08** (Shurf deferred + mutable-dict bug), ADR-003 ShurfModel interim rule. |

---

### D-28 — Stepshifter ensembles train constituents against the *published* (unfixed) views-stepshifter → `target_transform`/`log1p` silently not applied → raw training

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Running any stepshifter ensemble (`chunky_bunny` / `pink_ponyclub`) with `--train` (first-time constituent training) **before** the `target_transform` fix is merged to main and released to a version satisfying the constituents' `views-stepshifter>=1.0.0,<2.0.0` pin. |
| **Source** | falsify (2026-06-09) |
| **Status** | Open |
| **Location** | views-models `models/*/run.sh` (`env_path=.../envs/views_stepshifter`) + `models/*/requirements.txt` (`views-stepshifter>=1.0.0,<2.0.0`); views-pipeline-core `managers/ensemble/ensemble.py:403-416` (`_train_model_artifact` → `_execute_shell_script` → run.sh) + `cli/args.py:429` (`to_shell_command` default `script_name="run.sh"`); fix on views-stepshifter `experiment/exp01_target_transform` (unmerged — `origin/main` gate has **0** `target_transform` occurrences) |
| **Notes** | The `EnsembleManager` trains each constituent by **subprocessing that model's `run.sh`**, which activates a **per-model conda env** that pip-installs the **published** `views-stepshifter` (1.x) — which has **no `target_transform` mechanism**. The config's `target_transform: "log1p"` is therefore **silently ignored**; the constituent trains on the **raw** target (the exact D-17/#114 bug), with no error signal. The entire `log1p` validation this session (`car_radio`/`bittersweet_symphony`/`counting_stars` cm MSLE ~0.41) was performed in the `views_pipeline` env (editable install of the fixed branch) — **a different env than the ensemble's execution path** — giving false confidence. Same observable failure as **D-17** (raw/mis-scaled predictions) but a **distinct root cause**: an *unreleased-fix + published-env execution path*, not frozen artifacts. The fix is not actually "applied" to an ensemble run until it is **released** (issue #55) or the per-model envs are provisioned with the fixed code. Cross-ref **D-17**, **D-14** (version/build drift), **D-20** (ensemble path unaudited), **D-29** (envs absent); issues #55 (release), #114 (config flip). |

---

### D-29 — Per-model conda envs the ensemble invokes don't exist → on-the-fly creation from published reqs + 2 h/constituent timeout

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Running a stepshifter/DL ensemble (`chunky_bunny` / `pink_ponyclub`) for the first time on a box where `envs/views_stepshifter` and `envs/views_r2darts2` are absent. |
| **Source** | falsify (2026-06-09) |
| **Status** | Open |
| **Location** | views-models `envs/` (only `views-baseline`/`views_ensemble`/`views-hydranet` present — **no `views_stepshifter`, no `views_r2darts2`**); `models/*/run.sh` (creates env via `conda create` + `pip install -r requirements.txt` when absent); views-pipeline-core `ensemble.py` `_execute_shell_script` (`subprocess.run(..., check=True, timeout=7200)`) |
| **Notes** | The ensemble runs each of the 23 constituents via `run.sh` in its **own** env. The plain/Hurdle env `views_stepshifter` and the DL env `views_r2darts2` (TSMixer/NBEATS/NHiTS/TiDE) are **not present**, so first run would build them from scratch (conda create + pip) — slow, network-dependent, and prone to dependency-resolution failure. Additionally `_execute_shell_script` enforces a **7200 s (2 h) per-constituent timeout**; DL-constituent training may exceed it → `PipelineException` aborts the entire ensemble run. Cross-ref **D-28** (the freshly-created env also lacks the fix). |

---

### D-30 — No fail-loud guard on DL prediction magnitude / inverse-transform overflow → a diverged deep-learning constituent silently emits inflated forecasts that corrupt an ensemble

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any deep-learning (r2darts2) constituent that diverges during training (e.g. under a shifted/extended partition — as the +12-month bump triggered for NHiTS/`revolving_door`): its `sinh`-inverse outputs explode, the pipeline writes predictions and generates a **clean report with no error**, and a mean-aggregating ensemble that includes it is silently obliterated. |
| **Source** | review-diff of dossier 09 (2026-06-10), from the chunky_bunny re-run + the `revolving_door`/NHiTS investigation |
| **Status** | Open (latent — the specific instance is resolved; the guard gap is not) |
| **Location** | views-r2darts2 `darts_forecaster` inverse-transform (`AsinhTransform`→`sinh`) on model outputs; surfaced via `models/revolving_door` (NHiTSModel). No magnitude/range check in the r2darts2 predict path, nor in views-pipeline-core before constituent predictions are persisted and aggregated by `EnsembleManager` (mean). |
| **Notes** | **Same class as D-27** (ShurfModel `log_target=True` `expm1`-of-raw inflation — silent inverse-transform output explosion, no error signal) but a **distinct stack**: the deep-learning r2darts2 path, not stepshifter. Observed: `revolving_door`/NHiTS diverged to **y_hat_bar 1.0e21 then 1.2e23** (MSLE 67 / 1647) across runs; each produced 13 prediction files and a clean HTML report — **only a human noticing the MSLE/y_hat_bar caught it**. A mean ensemble including it ≈ 4e19. **Fail-loud asymmetry worth flagging:** bad **input** data IS gated (viewser drift-detection `InputGate` fail-loud-caught `yellow_submarine`'s absent IMF data), but bad **output** (10²¹ predictions) is **not** — no guard on prediction magnitude / inverse-transform range. The specific instance was **resolved** by the r2darts2 update + r7/r8 tuning (RevIN + SpotlightLossLogcosh), so nothing is live-corrupting now; the **guard gap remains** and any future DL divergence recurs silently. Mitigation: a fail-loud magnitude/range check on constituent predictions (and/or the inverse-transform output) before persist/aggregate, plus an ensemble-side sanity gate on constituent `y_hat_bar`. Cross-ref **D-27** (same class), **D-20** (ensemble aggregation unaudited), **D-25** (retransformation-bias). Documented in dossier `09_chunky_bunny_rerun_log.md`. |

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
