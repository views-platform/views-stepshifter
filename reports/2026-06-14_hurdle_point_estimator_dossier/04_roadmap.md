# 04 — Roadmap (phased, gated)

**Date:** 2026-06-14. Gates are hard — a phase does not start until the prior gate is green.
Maps to epic **#77**; salvage phase bounds everything (measure + decide, defer redesign).

## Phase 0 — Understand (DONE)
- Story A (#78): mechanism confirmed (D-33/D-37); real-data magnitude benign (MCR 0.75–1.29).
  Evidence: `reports/2026-06-13_hurdle_point_estimate_diagnosis/`. Register de-escalated to Tier 3.
- **Gate 0 (green):** the "is it broken?" question is answered — not in aggregate; tail unknown.

## Phase 1 — Pin + build the readout (the pre-flight, harness §C/§D)
- **1a — Story B (#75):** real Hurdle characterization tests (C-3). *Prerequisite for any code touch.*
- **1b — C-1 tail/distributional readout:** retrain-free, over existing parquets (tail-conditional
  error, per-cell calibration, missed-escalation rate). **The blocker.**
- **1c — C-2 frozen baseline:** snapshot the 6-Hurdle MSLE/MCR/ŷ̄ + plain controls.
- **1d — C-4 confound control:** name/define the untuned-LGBM control; decide whether to register it.
- **1e — Story D (#80):** diagonal-month assertion (independent hardening; can run anytime).
- **Gate 1:** pre-flight checklist (03 §D) green → program is "ready to run".

## Phase 2 — Critique + pre-register
- `expert-method-review` on `02_design` (attack the Option-0 recommendation + the tail metric choice).
- `/rnd-dossier preregister` **EXP-01**: the tail/calibration readout on the 6 Hurdles vs the frozen
  baseline (hypothesis: the identity Hurdle is tail-acceptable / or: the hard gate drops escalations).
- **Gate 2:** a pre-analysis plan with numeric falsifiers exists, linked from 00 + 07.

## Phase 3 — Execute + log
- Run EXP-01 (retrain-free). `/rnd-dossier log` the outcome (negative = postmortem, first-class).
- If falsifier fires (hard gate drops escalations) → the evidence for a *deferred* Option-1 ADR.
- **Gate 3:** log entry linked to its pre-registration; verdict vs falsifiers recorded.

## Phase 4 — Decide + promote
- Story C (#79): write the estimator decision (likely Option 0 + a documented deferred fix path).
- `/rnd-dossier promote`: draft proposed ADR from `02_design`; route residual risks (`register-risk`);
  archive the dossier. Redesign (Options 1/2/3), if justified, becomes a separate post-salvage epic.

## Decision points
- **DP-1 (after 1b):** does the tail readout show silent dropped escalations? → sets EXP-01's framing.
- **DP-2 (after EXP-01):** keep-and-document vs open a deferred probability-gate ADR.
- **DP-3 (Story C):** is the untuned-LGBM finding worth its own register entry + a tuning issue?
