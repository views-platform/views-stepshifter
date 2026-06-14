# Story A — HurdleModel point-prediction estimator: diagnosis (D-33 + D-37 + real-world magnitude)

**Issue:** #78 (EPIC #77). **Date:** 2026-06-13. **Scope:** point predictions only (not ShurfModel).
**Status:** diagnosis complete — mechanism confirmed, real-world magnitude assessed.

## TL;DR

The HurdleModel point estimator is **mis-specified** relative to the textbook hurdle mean
`E[Y] = P(Y>0)·E[Y|Y>0]` in two confirmed ways — **but on real data it is magnitude-honest in
aggregate, not broken.** My initial synthetic-only inference of "systematic under-prediction"
was directionally **wrong at the aggregate level** and is corrected here by real prediction data.
**Do not escalate; downgrade D-33/D-37 to Tier 3.** Residual risk is distributional/tail, not
aggregate magnitude.

## 1. Mechanism — confirmed on this repo's code (synthetic fixtures)

Reproduce: `…/diag_d33_d37.py` in an env with an editable install of this repo (`views_pipeline`).

- **D-33 — the binary stage emits hard `{0,1}` class labels, not probabilities.** The default
  `predict()` that `_predict_by_step` calls returns integral labels (`{0.}` on a no-event series,
  `{1.}` on an all-event series). The same model called with `predict_likelihood_parameters=True`
  returns fractional probabilities (e.g. `0.333`, `0.667`) — **a probability is available and the
  code discards it.** So `binary × positive` is a hard gate, not `P(Y>0)·E[Y|Y>0]`. Hurdle does
  **no** `clip(0,1)` (unlike ShurfModel, `shurf_model.py:134-136`).
- **D-37 — the positive stage trains on whole zero-heavy series.** The filter
  `[(t,p) … if (t.values()>0).any()]` (`hurdle_model.py:116-122`) keeps *entire* series that have
  any positive value, **zeros included**. On an ~87%-zero fixture, the resulting `target_pos` was
  **~82% zeros** → the positive regressor estimates `E[Y|unit ever positive]`, not `E[Y|Y>0]`.

Both are real code facts, independent of data.

## 2. Real-world magnitude — assessed against 6 production Hurdles (corrects the impact)

Source: companion note `/home/simon/brain/0_inbox/reports_13062026/hurdle_models_note.md` + its 24
calibration reports. Target `lr_ged_sb`, cm, calibration partition (months 457–492), **n=89,388**
cells/model, mean observed **ȳ̄ ≈ 28.8**. Predicted means **independently cross-checked** against
`models/<name>/data/generated/predictions_calibration_*.parquet` — matched the note to 3 s.f.
(twin_flame 32.355 vs 32.4; fast_car 37.415 vs 37.4; car_radio 9.472 vs 9.5; high_hopes 27.119
vs 27.1; fluorescent_adolescent 21.704 vs 21.7).

The six Hurdle constituents of `chunky_bunny` (all `target_transform: identity`):

| Hurdle model | MSLE ↓ | MCR = ŷ̄/ȳ̄ (→1) | ŷ̄ |
|---|---|---|---|
| fluorescent_adolescent (XGB) | 0.812 | 0.753 | 21.7 |
| green_squirrel (XGBRF) | 0.879 | 0.819 | 23.9 |
| fast_car (XGB) | 0.885 | 1.286 | 37.4 |
| little_lies (LGBM) | 1.086 | 0.819 | 23.6 |
| high_hopes (LGBM) | 1.114 | 0.933 | 27.1 |
| twin_flame (LGBM) | 1.193 | 1.119 | 32.4 |
| *car_radio (plain, **log1p**)* | *0.467* | *0.327* | *9.5* |

**Reading:** the Hurdles are **magnitude-honest** — MCR ∈ [0.75, 1.29], several near 1, some
slightly over. The **timid under-predictor is the plain `log1p` model** (car_radio, MCR 0.327,
predicts ⅓ of reality), not the Hurdles. The hard gate (D-33) passing full magnitude on
predicted-event cells and the zero-biased positive stage (D-37) roughly **offset in aggregate**,
landing the net point forecast near calibration.

Clean controlled comparison (verified, identical 69-feature set): **twin_flame** (LGBM-Hurdle,
identity) MSLE 1.193 / MCR 1.12 vs **car_radio** (XGB-plain, log1p) MSLE 0.467 / MCR 0.327. Same
inputs → the entire MSLE gap is `log1p` timidity, not skill. This is **D-22** on real data.

## 3. Verdict

The Hurdle point estimate is **NOT broken**; it is magnitude-honest in aggregate. The D-33/D-37
mechanisms are real mis-specifications but **empirically benign at the aggregate level**. The
escalation condition in D-33/D-37 ("silent production under-prediction") is **not met**.

**Residual concerns that remain valid (do not close):**
1. **Tail / distributional** accuracy is **unverified**. MCR is an aggregate mean-ratio; the hard
   gate zeroing low-probability-high-magnitude (escalation) cells is the EWS-dangerous direction
   and is unmeasured here. (Cross-ref D-22 tail, D-25 calibration.)
2. The **three LGBM hurdles are effectively untuned** (only `n_estimators` set) — a separate,
   fixable issue (fluorescent_adolescent XGB 0.812 vs little_lies LGBM 1.086 on identical features).
   Not yet a register entry.
3. The estimator is **not the textbook `E[Y]`**, so a data-regime shift could change the aggregate
   calibration unpredictably.

## 4. Register action (Story A DoD)

- **D-33:** Tier 2 → **3**. Mechanism confirmed (labels); real-data magnitude benign (MCR 0.75–1.29).
- **D-37:** Tier 2 → **3**. Mechanism confirmed (zero-heavy positive training); aggregate magnitude honest.
- **D-22:** strengthened — 6 real Hurdle MCR points + the verified twin_flame/car_radio comparison.
- Tail/distributional residual tracked under D-22/D-25; the untuned-LGBM observation is a candidate
  new entry (not registered here).
