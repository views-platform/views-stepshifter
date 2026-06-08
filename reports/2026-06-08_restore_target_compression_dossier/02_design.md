# 02 — Design: which target transform restores stepshifter performance

**Date:** 2026-06-08 · **Status:** draft (precedes `expert-method-review`; graduates to a proposed ADR on `promote`)

## Problem (from the evidence)

Stepshifter models fit a squared-error tree regressor (XGB/LGBM/RF) directly on `ged_sb` — a country-month conflict-fatality count that is ~90% zeros with a tail to 10⁴. With no compression, the model spreads heavy-tail mass onto the dominant zero class and over-predicts there; under MSLE this scores **worse than predicting all zeros** (synthetic: raw 4.79 vs zeros 2.16; field: `big_chungus` 2.519 vs zeros 2.147). The fatalities002 gold standard (cm MSLE 0.835) and the r2darts2 DL models both compress (log/asinh) and are healthy.

## Design question (the ONE decision this dossier validates)

For each stepshifter model family, **which `target_transform` ∈ {`identity`, `log1p`, `asinh`} restores cm MSLE to ≈ gold standard**, declared per-model in config, applied inside the library, inverted before output (per ADR-003)?

## Candidate transforms (and priors)

| Transform | Forward / inverse | Prior / rationale |
|---|---|---|
| `log1p` | `log1p` / `expm1` | What `5fcfe43` used; synthetic repro already restores below-baseline MSLE. Strong prior. Zero-preserving, monotonic. |
| `asinh` | `arcsinh` / `sinh` | What the DL models + hydranet/r2darts2 use; handles the tail like log but is defined on all reals and gentler near 0. Plausibly better-calibrated. Zero-preserving, monotonic. |
| `identity` | — | The control / current behavior; expected to remain worse-than-baseline (the thing we're replacing). |

**Open empirical question:** log1p vs asinh on *real* models against the *real* gold standard — not decidable from the synthetic repro alone.

## Approach

1. **Offline pre-screen** (cheap): one representative model, real cached data, three arms, cm MSLE vs gold + baselines. Filter before any full-pipeline run.
2. **Validate on the real pipeline** for the surviving transform(s) on a small set of representative `big_chungus` models (one per family: Hurdle, flat XGB/LGBM/RF).
3. **Decide per family** and record; feed the choice into the views-models config migration (#111) and a proposed ADR.

## Relationship to ADR-003 / the mechanism

This dossier does **not** build or re-decide the mechanism (registry, gate key, serialization-on-instance) — that is ADR-003 + the story (Half A). It supplies the **validated transform choice** ADR-003's mechanism will carry. The serialization stance (config-driven like hydranet vs serialized-with-artifact like r2darts2) is noted as an adjacent decision (ADR-003 currently leans serialized-on-instance) but is Half A's call.

## Constraints (from `03`)
Monotonic + zero-preserving transforms only (Hurdle/Shurf binarization). Identity must stay byte-identical. Raw output contract preserved. No symptom-masking.
