# 03 — Harness & Invariants (crown jewel)

**Date:** 2026-06-08. Audited from the views-stepshifter repo-assimilation + the FINDINGS investigation. ~60–70% of the standing harness already exists; the program-specific build is the offline readout + locked eval.

---

## A. Invariant taxonomy

### 1. Hard invariants — never break (an experiment violating one is INVALID)
- **Raw I/O contract (ADR-003):** model output must be raw target space. Any transform is internal + inverted before output. Predictions leaving the model are raw.
- **Identity = byte-identical baseline:** with `target_transform="identity"`, behavior must equal the current raw pipeline exactly. This is the default and the control.
- **Reproducibility gate passes:** `ReproducibilityGate.Config.audit_manifest` (`infrastructure/reproducibility_gate.py`) — required keys present & non-None; unknown algorithm/transform fails loud (`MissingHyperparameterError`).
- **Round-trip + shape:** `inverse(forward(x)) ≈ x` on the target domain; monotonic & zero-preserving (`forward(0)=0`) so Hurdle/Shurf binarization `forward(x)>0 ⇔ x>0` holds (ADR-003 clause 5).
- **Schema guard:** `views_validate` (`models/validation.py`) — month_id × {country_id|priogrid} float64 MultiIndex on `fit`.
- **No silent masking:** improvements must come from better target representation, not clamps/caps that hide the symptom. (`_get_standardized_df`'s negative→0 clamp is pre-existing, register D-16 — do not lean on it.)

### 2. Deliberately changed by this program (behind the `target_transform` flag, default `identity`)
- **What we replace:** broken models training on raw heavy-tailed counts (over-predict onto the 90%-zero mass → MSLE worse than all-zeros). 
- **How:** a declared per-model `target_transform` (`log1p`/`asinh`) applied in `_process_data` (forward) and inverted in `_predict_by_step` (inverse). Only active when declared non-identity. Reviewers should NOT defend "raw training" as sacred — it is the thing under test.

### 3. Respect while changing (not targeted, breakable in passing)
- **HurdleModel binarization** `(x>0)` — preserved only if the transform is monotonic + zero-preserving (see Hard).
- **ShurfModel sampling space** — `log_target`/`draw_dist` control *sampling*, orthogonal to training-space `target_transform`. Do NOT entangle. (Note dormant bug `shurf_model.py:210-212` for `log_target=True`; `fourtieth_symphony` exposed — out of this program's scope but don't worsen it.)
- **Feature transforms** — queryset `ln_*` features are separate from the target; out of scope (ADR-003 §8).
- **The 4 deep-learning constituents** (r2darts2) — already compress internally; unaffected; keep as the healthy comparison.

---

## B. Standing harness — what ALREADY exists (reuse)

| Mechanism | Status in repo | Reference |
|---|---|---|
| Config/manifest audit (reproducibility) | **Exists** | `ReproducibilityGate` + ADR-001; tests `tests/test_reproducibility_gate.py` (red/beige/green) |
| Schema/contract guard | **Exists (fit only)** | `views_validate` (`models/validation.py`); predict undecorated (gap, D-11) |
| Default-off flag pattern | **Partial** | `sweep` flag exists; `target_transform` default-identity is specified in ADR-003 but **unbuilt** |
| Test suite + CI | **Exists, weak** | `tests/`, `.github/workflows/run_pytest.yml` (push to main/development + PRs). Tests mock-heavy → verify wiring not numerics (D-13) |
| Locked baseline + metric | **Exists (external)** | fatalities002 gold (cm MSLE 0.835), all-zeros (2.147), no-change; MSLE protocol; rolling-origin. Lives in views-evaluation / the calibration reports |
| Negative-result discipline | **Exists** | `reports/` area, risk register, this dossier's `07` + postmortem template |
| Run discipline / artifacts | **Partial** | artifacts timestamped (`YYYYMMDD_HHMMSS` stems); WandB runs; no enforced one-job-at-a-time |
| Reproducibility seed lock | **Gap** | no enforced seed/entropy lock for XGB/LGBM `random_state`; ProcessPoolExecutor + spawn |
| Hardware fail-loud | **Gap (low risk)** | `get_device_params` silently falls back to CPU (tree models → low risk) |

## C. New harness this program needs (build BEFORE the first experiment)

1. **Fast offline readout** — retrain ONE representative model on a **real cached queryset** for one partition, three arms (`identity`/`log1p`/`asinh`), score cm MSLE vs gold + baselines. Must run **without** the full views pipeline if possible (sklearn/lightgbm offline), so we don't burn a full pipeline run to learn what minutes would tell us. (The synthetic `/tmp` repro is the seed of this; upgrade to real data.) **Must include a HurdleModel arm** (`/falsify` P2): the synthetic repro used a *flat* regressor, but the broken cohort is heavily HurdleModels whose binary `(x>0)` stage is unaffected by compression — the readout must exercise the two-stage structure, not just a flat regressor, or the result doesn't transfer to that family.
2. **Locked eval protocol** — pin metric (cm MSLE), baselines (all-zeros, no-change, average), partition/rolling-origin to match the fatalities002 comparison, so results compare honestly to the 0.835 gold.
3. **Parity test** — `target_transform="identity"` ⇒ predictions byte-identical to current raw pipeline.
4. **Round-trip + monotonic/zero-preserving test** — for each candidate transform (this is also Half A's test, reused here).
5. *(If using the real pipeline)* the `target_transform` mechanism itself (Half A) — but the offline readout (C1) lets us pre-screen transforms before Half A lands.
6. **Direct prediction inspection** (`/falsify` P3) — before trusting any readout, open the **actual** `big_chungus` constituent predictions (the calibration reports' "Prediction Samples", or the prediction parquet) and confirm the real broken predictions over-predict on true-zero country-months. Grounds the diagnosis in observation, not inference.
7. **Artifact provenance fetch** (`/falsify` P6) — fetch WandB run `2689xjtl` (or the stored constituent pkls) and directly confirm ≥1 actual constituent was trained **post-revert in raw space** (rules out the log-trained/predicted-raw mismatch directly, not by absence-of-counterexample).

## D. Pre-flight checklist (must be GREEN before EXP-01)

- [ ] **Runnable environment** — poetry env (or CI) with `lightgbm`/`xgboost`/`numpy`/`pandas`/`scikit-learn` (+ `darts`/`views_pipeline_core` only if using the real pipeline). **BLOCKER — currently red:** no poetry on PATH; anaconda base lacks deps.
- [ ] **Real data** — a cached queryset `.parquet` for ≥1 representative `big_chungus` model, with the gold-standard partition. **BLOCKER.**
- [ ] **Fast offline readout** built (C1) and sanity-checked on the synthetic case (already shows raw 4.79 vs log1p 1.66 vs zeros 2.16).
- [ ] **Eval protocol locked** (C2) to the fatalities002 comparison.
- [ ] **Parity test** (C3) green: identity == current raw.
- [ ] **Round-trip/monotonic test** (C4) green for log1p, asinh, identity.
- [ ] **HurdleModel arm** (C1 / `/falsify` P2) — the two-stage structure reproduces (or refines) the raw-vs-log1p result, not just a flat regressor. **BLOCKER for cohort generality.**
- [ ] **Direct prediction evidence** (C6 / `/falsify` P3) — actual broken constituent predictions inspected and shown to over-predict on true zeros (not inferred).
- [ ] **Artifact provenance confirmed** (C7 / `/falsify` P6) — ≥1 actual constituent verified raw-trained post-revert (WandB `2689xjtl`). Closes risk-register D-17's open verification.
- [ ] **Pre-analysis plan pre-registered** (`05` via `preregister`) — hypothesis + falsifiers + thresholds vs the locked baseline.
- [ ] **New failure modes** routed to `register-risk` (e.g. asinh vs log1p mis-declaration; the ShurfModel dormant bug interaction).

`status` must refuse "ready to run" until D is green.

## E. Decision/experiment protocol
One variable at a time (the transform, behind the flag, identity-default). Pre-register → run. Cheap offline readout → only then a full real-pipeline run. A fired falsifier kills the hypothesis (no ad-hoc rescue). Gains must come from representation (compression), not masking.

---

### Honest finding
Most of the *safety* harness (config audit, schema guard, locked external baseline, negative-result norms) is already here. The real build is small and program-specific: **the fast offline readout on real data + the locked eval protocol + the parity/round-trip tests.** The dominant blocker is operational — **a runnable environment and real cached data** — not methodology.
