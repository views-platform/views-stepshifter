# 05 — Analysis Plan (first experiment)

**Date:** 2026-06-14. **Status:** STUB — not yet pre-registered.

> Pre-registration is a deliberate, separate step. Do **not** treat this file as committed until
> `/rnd-dossier preregister` fills it via the template (hypothesis · the ONE variable · skepticism
> ledger · pre-registered predictions · falsifiers · method · decision rules), **after**
> `expert-method-review` on `02_design`, and **after** the pre-flight checklist (03 §D: C-1, C-2, C-3) is green.

## Candidate first experiment — EXP-01 (named, not yet committed)

- **Area:** D-22 / D-33 / D-37 tail residual.
- **Question:** does the identity Hurdle's **hard gate silently drop escalations** — i.e. is the
  aggregate-honest (MCR≈1) estimator *tail-acceptable*, or does it zero high-magnitude cells?
- **Shape:** retrain-free readout (C-1) over the 6 `chunky_bunny` Hurdle prediction parquets vs the
  frozen baseline (C-2): tail-conditional error on `obs>τ`, per-cell calibration curve, missed-escalation
  rate (gate=0 | obs>τ). Confound-controlled for tuning (C-4: matched-feature XGB vs LGBM).
- **Candidate falsifiers (to harden at preregister):** F1 — missed-escalation rate is high (gate zeros a
  large share of `obs>τ` cells) ⇒ the hard gate is operationally harmful (→ deferred Option-1 ADR). F2 —
  tail error is dominated by the untuned LGBMs, not the gate ⇒ it's a tuning problem, not an estimator one.

Blocking work before this can be pre-registered: **C-1, C-2, C-3** (03 §D).
