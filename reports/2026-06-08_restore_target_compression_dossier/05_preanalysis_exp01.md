# Pre-Analysis Plan — EXP-01: compression/likelihood pre-screen (restore-target-compression)

**Date:** 2026-06-08 (pre-registered *before* execution)
**Dossier:** `reports/2026-06-08_restore_target_compression_dossier/` · **Builds on:** `02_design.md` (widened by expert-method-review 2026-06-08), `FINDINGS.md`, the `/falsify` audit, risk register **D-22–D-25** + the **D-17** caveat.

> Committed before running the fix arms. Newest evidence at registration: brown_cheese's **real production identity baseline** is already measured — cm MSLE **≈4.27** (vs all-zeros **2.15**), mean prediction **16.0** on true-zero cells (85% zeros). That is the control and a parity target.

> **Amendment (2026-06-08, pre-execution — supersedes Tweedie/scope mentions below):** arms reduced to **{identity, log1p, asinh}** (no Tweedie — count-likelihoods deferred, D-23/D-26). Scope = **plain `StepshifterModel`** only (brown_cheese is XGBRF = plain ✓); HurdleModel/ShurfModel deferred ("here be dragons", D-26). Falsifier F5 (Tweedie) is **dropped**. **Prerequisite:** this experiment is only valid once the repo's transform infrastructure is centralized + audited + tested (transform-once / invert-once, nothing scattered) — the current boundary-hack run does **not** count.

## 1. Hypothesis
**H:** On real brown_cheese data, a monotonic zero-preserving target compression (`log1p` or `asinh`) **or** a count likelihood (Tweedie-deviance) reduces cm MSLE **below the all-zeros baseline (2.15)** and sharply reduces over-prediction on true-zero cells — i.e. the absence of compression is the cause, and restoring it fixes the flat-model failure.

## 2. Intervention (the ONE variable)
`objective/target_transform` ∈ **{identity (control), log1p, asinh, Tweedie-deviance}**, applied at the model boundary (forward on the target before fit; inverse on the prediction before scoring) for the transforms, and as the XGBoost objective for Tweedie. **Held constant:** data (`brown_cheese/.../calibration_viewser_df.parquet`), the 4 features, the partition, estimator family (XGBRF / XGB), seed, the stepshift structure.

## 3. Skepticism ledger
1. **`log1p` winning MSLE is near-circular** — training in log space ≈ optimising MSLE; a "win" may be tautological, not better forecasting.
2. **Compression may help zeros but hurt the escalation tail** — the policy-relevant product (D-22; Davison). Must be measured, not assumed away.
3. **Retransformation bias** — `E[expm1(ŷ)] ≠ expm1(E[ŷ])` (Jensen); a log-space point forecast back-transformed is biased low (D-25).
4. **This is brown_cheese (flat), not Hurdle** — `/falsify` P2 (Hurdle two-stage) stays open; do not generalise to the cohort.
5. **One model, one partition** — not the 19-model cohort; a positive result is suggestive, not decisive for the migration.
6. **Identity must reproduce production (~4.27)** or the harness is wrong and every arm is suspect.

## 4. Pre-registered predictions
| Endpoint (primary first) | Prediction | Threshold (pass / fail) |
|---|---|---|
| cm MSLE — log1p, asinh, Tweedie | each **< 2.15** (beats all-zeros) | pass if < 2.15; fail if ≥ 2.15 |
| cm MSLE — identity (control/parity) | **≈ 4.27** (reproduces production) | pass if within ±0.5 of 4.27; else harness invalid |
| mean prediction on true-zero cells | compressed arms **≪ 16.0** (toward ~0) | pass if < 2.0; concern if still > 5 |
| tail-conditional error (obs > τ) | compressed arms **not materially worse** than identity | flag if MSLE-winner is worst on the tail |
| calibration / retransformation bias | back-transformed mean ≈ observed mean | flag if systematic low-bias |

## 5. Falsifiers (pre-committed — any one fires ⇒ rejected, not rescued)
- **F1 (ineffective):** no compressed/Tweedie arm beats all-zeros on MSLE ⇒ compression is *not* the fix; the diagnosis is wrong.
- **F2 (works-but-degenerate):** an arm beats MSLE only by predicting ~0 everywhere (mean-on-nonzero collapses too) ⇒ symptom-mask, not a fix.
- **F3 (control/parity fails):** identity does **not** reproduce production ~4.27 ⇒ harness/parity broken — **run invalid**, fix before interpreting any arm.
- **F4 (MSLE misleads):** the MSLE winner is **worst** on the tail-conditional error ⇒ MSLE-only selection is operationally misleading; the tail metric must drive the choice (Davison).
- **F5 (transform was a detour):** Tweedie beats all three transforms on the joint readout ⇒ re-scope the fix toward a count likelihood, not a transform.

## 6. Method
- **Model/data:** real `brown_cheese` `calibration_viewser_df.parquet` (69,732 rows, `month_id`×`country_id`, target `lr_ged_sb` + 4 `lr_` features); brown_cheese config (algorithm XGBRFRegressor, its `steps`/`parameters`, partition from `config_partitions.py`).
- **Execution:** the **real `StepshifterModel`** (faithful to production), with the transform applied manually at the boundary (forward on target pre-fit; inverse on `pred_lr_ged_sb`) for identity/log1p/asinh, and `reg:tweedie` objective for the Tweedie arm. One variable per run; fixed seed.
- **Readout order (cheap → full):** (1) identity parity vs ~4.27 first — abort on F3; (2) then the three fix arms; (3) compute the full 4-part readout (§4) for each.
- **Scoring:** cm MSLE vs all-zeros (2.15) and parity; tail-conditional error on `observed > τ` (τ to be fixed at run time, e.g. the 99th percentile or a fatality count like 100); mean-on-true-zeros; calibration-in-the-large + retransformation-bias.
- **Run discipline:** `views_pipeline` conda env (GPU box; deps present); read-only on repo; experiment script + outputs under a scratch dir; results logged to `07_experiment_log` via `/rnd-dossier log` (including a negative).

## 7. Decision rules
- **F3 fires** → stop, fix the harness, re-run; interpret nothing.
- **F1 fires** → log a **negative** (postmortem); the no-compression diagnosis is wrong for flat models — re-open the investigation.
- **F2 fires** → log negative; the arm is a mask — exclude it, prefer a genuinely calibrated arm.
- **All compressed/Tweedie arms pass MSLE & not F4** → provisional winner = best on the **joint** (MSLE + tail + calibration); proceed to EXP-02 (Hurdle arm + more models) before any config migration.
- **F4 fires** → the tail metric overrides MSLE; pick the arm best on the tail among the MSLE-passers.
- **F5 fires** → re-scope `02_design` toward a count likelihood; note for the proposed ADR.
- **Any accept is provisional** — a single model/partition; the cohort decision needs EXP-02 (Hurdle + ≥1 per family) before promote.
