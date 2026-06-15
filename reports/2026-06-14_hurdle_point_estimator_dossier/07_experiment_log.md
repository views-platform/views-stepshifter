# 07 — Experiment Log (append-only)

Append-only ledger of every **pre-registered** run + outcome — **negatives first-class** (postmortem,
equal prominence). Each entry links its pre-registration (05 / a `preregister` artifact) and states the
verdict against the pre-committed falsifiers. Newest at the bottom; never edit past entries except to
add a cross-link. Entry template: `references/templates.md` §2; postmortem: §3.

## Established prior context (NOT a dossier experiment — no pre-registration)
- **Story A diagnosis (#78, 2026-06-13)** — mechanism confirmed (D-33 hard-gate labels; D-37
  zero-dominated `target_pos`); real-data magnitude benign (6 Hurdles, MCR 0.75–1.29, parquet-verified).
  This is *founding context* (see `00_README` §5 and `reports/2026-06-13_hurdle_point_estimate_diagnosis/`),
  recorded here for completeness; it predates the dossier and was a diagnostic, not a pre-registered test.

---

## Pre-registered experiments

### EXP-01 · Hurdle tail/distributional readout (C-1) · 2026-06-15 · **FALSIFIED**
- **Plan (pre-reg):** `05_analysis_plan.md` (committed before any tail result was seen).
- **Variable:** measured the tail readout (no model change) — pooled 13 calibration sequences joined to
  observed `lr_ged_sb`, months 457–492, n≈74,490/model.
- **Driver / artifact / results:** `exp01_tail_readout.py`; full numbers in
  `postmortem_exp01_hard_gate_drops_escalations.md`.
- **Readout:** missed-escalation `mean(pred<0.5 | obs>10)` = **0.07–0.32** across the 6 Hurdles vs
  **0.022** for the plain reference car_radio; tail MSLE worse for all but green_squirrel; per-cell
  calibration wrecked (over-predict small cells, under-predict the extreme tail) under MCR≈1.
- **Verdict vs falsifiers (plan §5):** **F1 FIRED** (Hurdle tail missed-escalation + error materially
  worse than references), **F3 FIRED** (aggregate MCR masks tail collapse), **F2 partial** (LGBMs worse,
  but the best XGB hurdle still misses 3× the reference — the gate effect survives the confound) ⇒
  **H FALSIFIED.**
- **Decision (plan §7):** reject Option 0; **recommend Option 1** (probability gate) via a **deferred**
  ADR/issue linking #66 — **do not implement** (salvage). Re-escalate **D-33**; track tuning (**D-23**)
  separately. Postmortem: `postmortem_exp01_hard_gate_drops_escalations.md`.
