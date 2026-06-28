# 01 — Literature / prior art

**Date:** 2026-06-08 · append as sources are fetched (use the `library` skill).

## Have locally (in `reports/investigation_metrics_08062026/`)
- **fatalities002 retrospective** (`fatalities002_retrospective-7.pdf`) — the gold-standard benchmark. **Take:** the cm MSLE = 0.835 target, the MSE-vs-MSLE discussion (MSLE preferred for the zero-inflated heavy-tail target), the rolling-origin protocol, baseline definitions (no-change, all-zeros). The locked eval comes from here.
- **model appendix** (`model_appendix_draft_1.md`) — the r2darts2 DL models. **Take:** they map the raw count to **asinh** space ("compresses the right tail, retains zeros, log-like for large counts"); hybrid-space RevIN; SpotlightLossLogcosh. Evidence that asinh is the platform's chosen compression for this exact target.
- **calibration reports** (the two HTML) — the `big_chungus` field metrics (the divergence). **Take:** the per-constituent cm MSLE table.

## Gaps to fetch
- [ ] **Hegre et al. (2020)** Appendix A (stepshift methodology) — the canonical stepshifter description.
- [ ] **Hegre et al. (2022)** — the fatalities-model validation the retrospective compares against (Figure 7).
- [ ] Transform/zero-inflation literature: log1p vs asinh for heavy-tailed counts; MSLE properties under over/under-prediction; hurdle/two-stage count models.
- [ ] **Pedregosa et al. (2011)** (sklearn) — the MSLE recommendation the retrospective cites.

## Cross-repo precedent (not literature, but design prior art)
- views-hydranet ADR-046 + `config_initializer.TRANSFORMS` (log1p/asinh/identity registry).
- views-r2darts2 ADR-012 + `scaler_selector` (LogTransform/AsinhTransform + Darts Pipeline).
- Both compress; both invert to raw before evaluation. Stepshifter is the outlier with no compression.
