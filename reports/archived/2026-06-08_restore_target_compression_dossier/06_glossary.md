# 06 — Glossary

**Date:** 2026-06-08 · append as terms arise.

- **target_transform** — the per-model declared compression of the target (`identity`/`log1p`/`asinh`), applied inside the model library and inverted before output (ADR-003). The variable this dossier validates.
- **raw / native space** — untransformed conflict-fatality counts (`ged_sb`), the contracted model I/O scale (ADR-003).
- **log1p / expm1** — `ln(1+x)` and its inverse `e^x − 1`. Zero-preserving, monotonic. What `5fcfe43` applied.
- **asinh / sinh** — `arcsinh(x)` and inverse. Log-like for large x, gentle near 0, defined on all reals. What the DL/r2darts2/hydranet models use.
- **cm MSLE** — country-month Mean Squared Log Error; the primary metric (stable under the zero-inflated heavy-tail target; preferred over MSE per the retrospective).
- **all-zeros baseline** — predict 0 everywhere; cm MSLE 2.147 (this eval) / 2.833 (gold). A model worse than this is over-predicting onto true zeros.
- **no-change / LOCF baseline** — last-observation-carried-forward.
- **gold standard** — the fatalities002 retrospective model, cm MSLE 0.835.
- **Half A / Half B** — Half A = the engineering mechanism (registry + config key + gate), governed by ADR-003 / the story. Half B = this dossier (the empirical transform-choice decision).
- **big_chungus** — the 2026-06-04 ensemble whose stepshifter constituents were broken (cm MSLE 2.519).
