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

## Widened scope (expert-method-review, 2026-06-08)

The original framing — *"which transform best restores cm MSLE"* — pre-commits to (a) transforms of a squared-error regressor and (b) a single scale-dependent metric, so it cannot discover a better model and can reward tail-compression that hurts the policy-relevant cases. Amendments (→ risk register D-22–D-25):

1. **Add a count-likelihood arm** (Murphy; Damato2025 GP-Tweedie). `log1p` is variance-stabilisation of a **Tweedie** likelihood (point mass at 0 + continuous positive part = this DGP). Arms become **{identity, log1p, asinh, Tweedie/NB-deviance}** — if the count model dominates, the transform framing was a detour. (D-23)
2. **Multi-component readout, not cm MSLE alone** (Gneiting, Davison, Harrell): report **cm MSLE + a tail-conditional error (error | obs > threshold) + calibration-in-the-large**. MSLE on a 90%-zero target favours zero-fit and tail-compression (Bolin2023); the winner on MSLE may be the *worst* escalation forecaster (Hegre2022; Abilasha2022). Make the zero-fit↔tail tradeoff visible. (D-22)
3. **Retransformation-bias / calibration check** (Harrell): round-trip `inverse(forward(x))≈x` is **not** calibration. By Jensen, `E[expm1(ŷ)] ≠ expm1(E[ŷ])` — a log-space point forecast `expm1`-ed is biased low. Add prediction-space calibration per arm. (D-25)
4. **Real-DGP posterior predictive check as an acceptance gate** (Gelman): does the chosen arm reproduce the real held-out zero-fraction *and* tail? Closes the `/falsify` P2/P3 gaps and the synthetic-evidence weakness. (D-17 caveat)
5. **Question the hurdle** (Harrell): include a single-count-model baseline vs the HurdleModel two-stage `(x>0)` split. (D-24)

**Sequencing:** ship `log1p` as the immediate restoration (Operational pragmatism — it is ADR-003-compliant and what the gold standard / DL models use); run the widened EXP-01 to decide the *durable* per-family choice. Strongest dissent to keep live: **Davison** — a transform that wins MSLE may be the worst at escalation; the tail-conditional metric is non-negotiable before declaring a winner.

**Library to fetch:** Harrell *Regression Modeling Strategies*; ZIP/hurdle canonical (Lambert 1992 / Mullahy 1986); Tweedie (Jørgensen); asinh-for-counts source; Breiman *Two Cultures*.

## Relationship to ADR-003 / the mechanism

This dossier does **not** build or re-decide the mechanism (registry, gate key, serialization-on-instance) — that is ADR-003 + the story (Half A). It supplies the **validated transform choice** ADR-003's mechanism will carry. The serialization stance (config-driven like hydranet vs serialized-with-artifact like r2darts2) is noted as an adjacent decision (ADR-003 currently leans serialized-on-instance) but is Half A's call.

## Constraints (from `03`)
Monotonic + zero-preserving transforms only (Hurdle/Shurf binarization). Identity must stay byte-identical. Raw output contract preserved. No symptom-masking.
