# 05 — Analysis plan (pre-registrations)

**Date:** 2026-06-08 · **Status:** placeholder — no experiment pre-registered yet.

Pre-analysis plans are written with the `preregister` verb, **before** each experiment runs, and linked from `00_README` and `07_experiment_log`. The first one (EXP-01) is gated behind Phase-1 pre-flight (`03` §D) going green.

## Planned (not yet pre-registered)
- **EXP-01 — offline transform pre-screen.** Intervention: `target_transform` ∈ {identity, log1p, asinh} on one representative model, real cached data. Primary endpoint: cm MSLE vs fatalities002 gold (0.835) + all-zeros (2.147) baseline. Candidate falsifiers (to formalize at `preregister`): F1 log1p/asinh does **not** beat all-zeros ⇒ compression isn't the fix; F2 it beats baseline only by masking (e.g. degenerate near-zero predictions everywhere) ⇒ symptom-mask not a fix; F3 identity (control) unexpectedly matches ⇒ the diagnosis is wrong.

→ Run `/rnd-dossier preregister` for EXP-01 once `03` §D is green.
