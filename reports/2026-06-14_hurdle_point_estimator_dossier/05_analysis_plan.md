# Pre-Analysis Plan — EXP-01: HurdleModel tail/distributional readout (C-1)

**Date:** 2026-06-15 (pre-registered *before* execution — only aggregate stats inspected so far:
calibration window months 457–492, mean obs ≈ 26.4, zero-fraction 0.845; **no tail/missed-escalation
result seen**).
**Dossier:** `00_README.md` · **Builds on:** Story A diagnosis (`reports/2026-06-13_hurdle_point_estimate_diagnosis/findings.md`),
the method-review consensus (C-1-first; relative decision rule), risks **D-22 / D-25 / D-33 / D-37**.
**Pre-flight (03 §D):** C-3 (Story B tests) **done**; C-2 (MCR/MSLE baseline) from Story A; **C-1 is the
readout this plan commits** — preregistration precedes its build/run by design.

## 1. Hypothesis
**H:** the identity HurdleModel's hard `{0,1}` gate does **not** materially drop escalations — its
aggregate magnitude honesty (MCR 0.75–1.29, Story A) extends to the operational **tail**, so the
estimator is tail-acceptable and **Option 0** (keep + document) holds.

## 2. Intervention (the ONE thing measured — no model change)
For each of the 6 `chunky_bunny` Hurdle models, **pool** its 13 rolling calibration prediction
sequences (`predictions_calibration_*_00..12.parquet`, `pred_lr_ged_sb`), **join** to observed
actuals (`lr_ged_sb` from `models/<name>/data/raw/calibration_viewser_df.parquet`) on
`(month_id, country_id)` over months 457–492, and compute the tail readout. Retrain-free; no code or
model is changed.

## 3. Skepticism ledger
1. **Davison/EVT:** the hard gate may zero low-probability/high-magnitude escalation cells; aggregate
   MCR is blind to this.
2. **Gneiting:** aggregate calibration arises from two offsetting mis-specifications (D-33 hard gate +
   D-37 series-level conditioning) — a coincidence that may break in the tail.
3. **Untuned-LGBM confound (D-23):** tail error could be driven by the 3 effectively-untuned LGBM
   hurdles, not the gate itself.
4. **Alignment caveat:** `pred≈0` conflates gate-closed vs positive-stage-zero — but for the operational
   question ("the model predicts ~0 on a real escalation") the **final** `pred` is what matters.
5. **Pooling caveat:** pooling 13 rolling sequences may repeat some `(month,country)` cells at different
   horizons; report `n`, the pooling method, and (if feasible) a per-step sensitivity.

## 4. Pre-registered predictions
| Endpoint (primary first) | Prediction (H) | Reference |
|---|---|---|
| Missed-escalation rate `mean(pred≈0 \| obs>τ)`, τ∈{1,10,25,100} | Hurdles **no worse** than references | car_radio (log1p) + all-zeros |
| Tail error (MSLE & MAE on `obs>τ`) | Hurdles **no worse** than references | car_radio + all-zeros |
| Calibration (mean pred vs mean obs by observed decile) | top deciles **not** systematically collapsed | — |
| Confound split (matched-feature XGB vs LGBM) | tail gap, if any, **concentrated in untuned LGBMs** | fast_car/fluorescent_adolescent vs high_hopes/little_lies/twin_flame |

`pred≈0` operationalised as `pred < 0.5` (sub-one-fatality); sensitivity at `pred==0` exactly reported.

## 5. Falsifiers (pre-committed — any one fires ⇒ the relevant conclusion, not rescued)
- **F1 — gate drops escalations (Option 0 FALSIFIED):** the Hurdle family's tail missed-escalation rate
  **and** tail error are **materially worse** than the reference baselines (car_radio + all-zeros) — the
  hard gate misses escalations the references catch ⇒ reject Option 0; recommend **Option 1** (probability
  gate) via a **deferred** ADR/issue linking #66 (do **not** implement — salvage phase).
- **F2 — confound, not the gate:** the tail degradation is **concentrated in the untuned LGBM hurdles**
  (high_hopes/little_lies/twin_flame) while the matched-feature XGB hurdles are tail-acceptable ⇒ it is a
  **tuning** problem (D-23), not an estimator one; Option 0 holds for the estimator, tuning tracked separately.
- **F3 — aggregate masks the tail:** MCR≈1 but tail **calibration collapses** (systematic under-prediction
  in the top deciles) ⇒ aggregate honesty is confirmed misleading; the tail readout **supersedes MCR** for
  the decision (regardless of F1/F2).

## 6. Method
Retrain-free. Per model: load `predictions_calibration_*_00..12.parquet`, concatenate, join actuals on
`(month_id, country_id)`, restrict to 457–492; compute §4 endpoints. **Controls:** car_radio (plain log1p)
and an all-zeros baseline computed identically; matched-feature XGB-vs-LGBM pairs for F2. Readout order:
this **is** the cheap retrain-free probe (no expensive run gated behind it). **Run discipline:** a committed
script under the dossier (`exp01_tail_readout.py`); results → `07_experiment_log.md` with the verdict vs
F1/F2/F3 and the decision.

## 7. Decision rules (the RELATIVE rule — user-chosen)
- **Option 0 holds** (keep the hard gate; document it as a *known mis-specification that aggregates
  benignly*, **not** a deliberate property) **iff** the Hurdle tail missed-escalation **and** tail error
  are **no worse than** the references **and** tail calibration does not collapse (F1 not fired, F3 not fired).
- **F2 is adjudicated first:** if the tail gap is a tuning artdefact, Option 0 holds for the estimator and a
  tuning issue is opened — **no** Option-1 call.
- **Else (F1 and/or F3 fire on the estimator, not tuning):** recommend **Option 1** (probability gate) via a
  **deferred** ADR/issue linking #66 — **do not implement** (salvage phase).
- All outcomes (including a clean confirm of H) are logged with equal prominence; a negative gets a postmortem.
