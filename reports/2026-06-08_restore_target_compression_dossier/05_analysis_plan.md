# 05 — Analysis plan (pre-registrations)

**Date:** 2026-06-08 · **Status:** placeholder — no experiment pre-registered yet.

Pre-analysis plans are written with the `preregister` verb, **before** each experiment runs, and linked from `00_README` and `07_experiment_log`. The first one (EXP-01) is gated behind Phase-1 pre-flight (`03` §D) going green.

## Planned (not yet pre-registered) — widened per expert-method-review (2026-06-08)

- **EXP-01 — offline compression/likelihood pre-screen.**
  - **Arms** (intervention, one variable): `target_transform` ∈ {identity, log1p, asinh} **+ a count-likelihood arm (Tweedie/NB-deviance)** (D-23) — on one representative model, real cached data.
  - **Readout (multi-component, NOT MSLE alone — D-22):**
    1. *Primary:* cm MSLE vs fatalities002 gold (0.835) + all-zeros (2.147) baseline.
    2. *Tail:* a tail-conditional error (error | observed > threshold) — the escalation signal (Davison/EVT).
    3. *Calibration:* calibration-in-the-large + a smooth calibration curve, incl. a retransformation-bias check `E[expm1(ŷ)] vs expm1(E[ŷ])` (D-25).
    4. *DGP gate (Gelman):* a real-held-out posterior predictive check — does the arm reproduce the zero-fraction **and** the tail? (closes `/falsify` P2/P3.)
  - **Falsifiers (to formalize at `preregister`):**
    - F1 — compression/likelihood does **not** beat all-zeros on MSLE ⇒ compression isn't the fix.
    - F2 — beats MSLE only by degenerate near-zero predictions everywhere ⇒ symptom-mask, not a fix.
    - F3 — identity (control) unexpectedly matches ⇒ the diagnosis is wrong.
    - **F4 — the MSLE winner is *worse* on the tail-conditional error** ⇒ MSLE selection is operationally misleading (D-22; Davison's dissent).
    - **F5 — the Tweedie/NB arm beats all three transforms** ⇒ the transform framing was a detour; re-scope toward a count likelihood (D-23).

→ Run `/rnd-dossier preregister` for EXP-01 once `03` §D is green (incl. the new HurdleModel arm, real-prediction inspection, and artifact provenance items).
