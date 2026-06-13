# 01 — Literature

**Date:** 2026-06-14. Grounding for the estimator decision. Held items are in the `library` corpus
(`/library search`); to-fetch items are named and should be added before they're leaned on.

## Held (in corpus) — and what we take from each
- **Gneiting2007 / Jordan2019 / Brocker2007 / Matheson1976** — proper scoring & **elicitability**: a
  point forecast is only meaningful relative to the functional its loss elicits (squared error → mean;
  abs error → median). *Take:* the basis for D-38 — the Hurdle product elicits no clean functional, and
  MSLE (≈median-of-log) ≠ what squared-error stages target.
- **Bolin2023 (local scale-invariance)** & **Lerch2017 (Forecaster's Dilemma)** — scores on
  heavy-tailed targets; standard scores reward ignoring extremes. *Take:* motivates the **tail-weighted**
  readout (C-1) and warns that MSLE alone selects timid models (D-22).
- **Hegre2025 (VIEWS Prediction Challenge)** — operational product is *full predictive distributions*,
  not point estimates. *Take:* context for why point-estimate work is bounded; the distributional path
  (Shurf) is the longer arc.
- **Vesco2026 (underreported death toll)** — UCDP counts systematically under-reported, non-linearly.
  *Take:* the "observed" target the magnitude stage fits is itself biased low — relevant to interpreting MCR.

## To fetch (decision rests on these; not yet held)
- **Mullahy 1986** (hurdle models) & **Lambert 1992** (ZIP) — the canonical two-part / zero-inflated
  likelihoods; the definition `E[Y]=P(Y>0)·E[Y|Y>0]` the current code approximates badly.
- **Croston 1972** & **Syntetos–Boylan 2005 (SBA)** — intermittent-demand occurrence×size decomposition
  and its *known bias* + correction — the forecasting-literature twin of this exact estimator.
- **Harrell, *Regression Modeling Strategies*** — anti-dichotomization (the `(x>0)` split, D-24);
  calibration-in-the-large; retransformation.
- **Duan 1983 (smearing)** — retransformation bias for inverse-transformed point forecasts (D-25; dormant
  while identity, but relevant if Option 1/3).
- **A Tweedie reference** (Jørgensen 1987 or Dunn & Smyth 2005) — the count-likelihood alternative (D-23, Option 3).

## Gaps / notes
- No held source directly on **gate-as-label vs gate-as-probability** in a GBM hurdle — the empirical
  C-1 readout is our own evidence; SBA is the nearest analogue.
