# 03 — Harness & Invariants

**Date:** 2026-06-14. The guardrails that make experimenting on the Hurdle point estimator safe.
Audited against the repo as it stands; **~70% of the standing harness already exists** — the real
build is the program-specific **tail/distributional readout** (§C).

## A. Invariant taxonomy

### A.1 Hard invariants — never break (a run that violates one is invalid)
- **Raw target-space I/O (ADR-003).** Model takes raw target, returns raw predictions; any transform
  is declared + inverted internally. Hurdle is **identity-pinned** (gate + `__init__` guard).
- **Reproducibility gate (ADR-001).** `audit_manifest` passes before training; required hyperparameters
  present + non-None; `target_transform` ∈ closed registry.
- **Fail-loud, no new silent clamps.** The one existing clamp (`_get_standardized_df`, neg/inf→0,
  D-16) is the *only* sanctioned magnitude masking; experiments must not add masking that hides a
  symptom (see §E "magnitude-neutral by construction").
- **Stable baseline stays default + reproducible.** The current identity Hurdle is the default; any
  new path is provably inert when off; full pytest suite green.
- **Package isolation.** No cross-model-package imports (stepshifter ⊥ hydranet/r2darts2/baseline).

### A.2 Deliberately changed by this program (behind a flag) — *currently EMPTY*
Per the salvage phase, **nothing is being changed yet**. Candidate changes (only if Story C/02_design
+ a future ADR approve them, each behind a default-off flag):
- (c1) binary stage **label → probability** (use `predict_likelihood_parameters`/clip) — would make
  the gate `P(Y>0)` instead of a hard class. *Deferred; not in scope until decided.*
- (c2) positive-stage conditioning **series-level → observation-level** (`Y>0`). *Deferred.*
- (c3) a **count-likelihood** arm (Tweedie/NB) as an alternative to the two-stage hurdle. *Deferred.*
State explicitly so reviewers don't defend current behavior as sacred — but **do not implement** in salvage.

### A.3 Respect while changing (breakable in passing, not targeted)
- **Aggregate magnitude honesty (MCR ≈ 0.75–1.29).** Whatever we measure/decide, don't *worsen* the
  in-the-large calibration that the identity Hurdle currently has.
- **The identity-pin / D-26 deferral** and **ShurfModel** (out of scope; shares the gate but has its
  own sampler — D-27).
- **The diagonal "last-month-with-data" prediction semantics** (`_predict_by_step`; D-35).

## B. Standing harness — audit (reuse, don't reinvent)

| Mechanism | Status | Where |
|---|---|---|
| Default-off feature flag | **Present** | declared `target_transform` registry + gate; identity default; non-identity gate-rejected for Hurdle/Shurf (`reproducibility_gate.py` steps 5–7) |
| Parity / regression gates | **Partial** | pytest `tests/`; CI `run_pytest.yml` (3.11). **Gap:** Hurdle numerical core xfail'd (D-13) → Story B. Real template exists: `test_target_transforms.py` |
| Reproducibility | **Partial** | gate audits hyperparameters; **gap:** no seed lock audited — LGBM/XGB `random_state` not asserted (blind spot from review-rr). Note: tiny-data fits are noisy |
| Fast, cheap readout | **Present (extend)** | Story A diagnostic (`reports/2026-06-13_hurdle_point_estimate_diagnosis/diag_d33_d37.py`, seconds, retrain-free) + the parquet MCR cross-check (load `predictions_calibration_*.parquet`, compute mean). **Extend** with the tail readout (§C) |
| Evaluation comparability | **Present (lock it)** | MSLE (primary) **+ MCR = ŷ̄/ȳ̄** (magnitude guardrail) on the calibration partition (months 457–492, ȳ̄≈28.8, n=89,388). The 6 Hurdle + plain baselines are the comparison set — **freeze them** (§C-2) |
| Run discipline | **Present** | artifact-timestamp contract (ADR-002); one-heavy-job pipeline; config trap-restore; predictions/eval parquets timestamped + preserved |
| Negative-result discipline | **Present** | postmortem norm from the archived restore-target-compression dossier; 07 here is append-only |
| Hardware/runtime gates | **N/A-ish** | CPU path is default (`get_device_params`); no accelerator required for these gradient-boosted learners |

## C. New harness this program needs (build before the FIRST experiment)

- **C-1 — Tail/distributional readout (THE blocker).** MCR is an aggregate mean-ratio; it cannot see
  the hard gate zeroing low-probability/high-magnitude **escalation** cells (the EWS-dangerous
  direction). Build a retrain-free readout over the existing prediction parquets that reports, per
  model: (i) **tail-conditional error** (e.g. MSLE / MAE on `obs > τ` for a few τ), (ii) **per-cell
  calibration** (predicted vs observed by magnitude bucket; a reliability curve), (iii) **gate
  behavior** (rate at which the binary stage = 0 on cells with `obs > τ` — i.e. missed escalations).
  Reuse the Story-A parquet-loading pattern. This is the decisive missing measurement.
- **C-2 — Frozen Hurdle comparison baseline.** Snapshot the 6 `chunky_bunny` Hurdle constituents'
  MSLE/MCR/ŷ̄ (and the plain `car_radio`/`twin_flame` controls) as a locked table so any future arm
  compares honestly. (Story A already verified the means to 3 s.f.)
- **C-3 — D-13 Hurdle characterization tests (Story B).** Pin current behavior (zero-series→0, hard
  gate, diagonal index) so the estimator can't silently regress while we measure/decide.
- **C-4 — Tuning confound control.** The 3 LGBM hurdles are effectively untuned (only `n_estimators`).
  Any transform/estimator readout must separate the **transform/estimator** effect from the **tuning**
  effect (matched-feature XGB-vs-LGBM pairs, or a tuned-LGBM arm) — else a result is confounded.

## D. Pre-flight checklist (must be green before the FIRST pre-registered experiment)

- [ ] **C-1 tail/distributional readout** implemented + sanity-checked on one model — **blocker**
- [ ] **C-2** baseline table frozen (the locked comparison)
- [ ] **C-3** Story B characterization tests green; D-13 Hurdle slice closed
- [ ] **C-4** the tuning confound named and a control arm/plan defined
- [ ] readout is **retrain-free** and runs on the existing parquets (cheap-before-expensive)
- [ ] full `pytest tests/` + CI green; no new silent clamp introduced
- [ ] first experiment **pre-registered** (`/rnd-dossier preregister`) with falsifiers vs the frozen baseline
- [ ] any new failure mode routed to `register-risk` (e.g. the untuned-LGBM finding)

`status` must refuse to call this program "ready to run" until C-1, C-2, C-3 are green.

## E. Decision/experiment protocol

- **One variable at a time** — each candidate change (c1/c2/c3) behind its own flag, measured alone.
- **Pre-register, then run** — hypothesis + falsifiers committed before execution.
- **Cheap-before-expensive** — the retrain-free parquet readout (C-1) filters before any retrain.
- **Falsifier honesty** — a fired falsifier kills the hypothesis; postmortem it (07), don't rescue.
- **Magnitude-neutral by construction** — gains must come from better representation (a real
  probability gate, an observation-level conditional, a count likelihood), **never** from clamps/caps
  that mask the symptom. (The existing neg/inf→0 clamp is the sanctioned exception, not a precedent.)
- **Salvage gate** — measurement & decision are in-scope now; *implementing* c1/c2/c3 is **not**
  until a promoted ADR lifts the D-26 deferral.
