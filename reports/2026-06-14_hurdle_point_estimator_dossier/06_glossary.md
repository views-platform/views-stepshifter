# 06 — Glossary

**Date:** 2026-06-14. Shared vocabulary for this dossier.

- **Hurdle (two-stage) estimator** — point forecast = `gate × magnitude`: a binary "any event?" stage
  times a positive-magnitude regressor. Here `HurdleModel` in `hurdle_model.py`.
- **Hard gate (D-33)** — the binary stage outputs a class **label** in `{0,1}` (argmax), not a
  probability; the available `P(Y>0)` is discarded. A cell rounded to 0 contributes exactly 0.
- **`target_pos` (D-37)** — the positive-stage training subset: *whole series* that have any positive
  value, **zeros included** → it estimates `E[Y|unit ever positive]`, not `E[Y|Y>0]`.
- **MCR (magnitude calibration ratio)** — `ŷ̄ / ȳ̄`, mean predicted over mean observed; →1 is honest.
  The aggregate guardrail against the *timid prophet*. Real Hurdles: 0.75–1.29.
- **MSLE** — mean squared *log* error; primary point metric; structurally rewards near-zero/timid
  prediction on ~95%-zero data (D-22). Optimises ≈ the geometric mean (median), under-predicts totals.
- **Timid prophet** — a model that wins MSLE by predicting near-zero everywhere (low MCR). The plain
  `log1p` model `car_radio` (MCR 0.327) is the exemplar; the Hurdles are **not** timid.
- **Tail-conditional readout (C-1)** — error / calibration computed on cells with `obs > τ` (the
  escalation tail), plus the **missed-escalation rate** (gate=0 on `obs>τ`). The decisive missing measurement.
- **Elicitability (D-38)** — a point forecast is meaningful only relative to the functional its loss
  elicits; the Hurdle product elicits none cleanly.
- **Retransformation / smearing bias (D-25)** — `E[g⁻¹(ŷ)] ≠ g⁻¹(E[ŷ])`; biases inverse-transformed
  point forecasts low. Dormant under identity; live under a non-identity transform.
- **Salvage phase** — current program stance: restore sane bug-free values, defer model-improvement
  (Hurdle/tail/MSLE) work. Bounds this dossier to measure+decide, not redesign (D-26).
- **MCR-vs-MSLE trade-off** — `identity` → honest magnitude, higher MSLE; `log1p` → low MSLE, timid. The
  central tension (D-22).
