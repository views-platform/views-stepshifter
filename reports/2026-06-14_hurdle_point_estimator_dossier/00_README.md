# Dossier — HurdleModel Point-Prediction Estimator

**Status:** ACTIVE. **Phase:** EXP-01 run + **FALSIFIED** (2026-06-15) — the hard gate drops escalations;
Option 0 rejected, **Option 1 recommended (deferred)**. Next: Story C decision + D-33 re-escalation.
**Owner:** Simon Polichinel von der Maase. **Repo:** views-stepshifter.

## 1. Purpose

Govern the investigation of the **HurdleModel point-prediction estimator** (`binary × positive` in
`hurdle_model.py`): decide *whether and how* to act on two confirmed mis-specifications (a hard
`{0,1}` gate that discards an available probability — D-33; a positive stage trained on whole
zero-heavy series, estimating `E[Y|ever-positive]` not `E[Y|Y>0]` — D-37) given that the estimator
is, on real data, **magnitude-honest in aggregate** (MCR 0.75–1.29) yet **unverified in the tail**
that actually matters for an early-warning system. This is a *staging area*; the exit ramp is a
**proposed ADR** carrying the estimator decision (currently Story C / issue #79).

## 2. Relationship to prior work / ADRs

- **Salvage phase (binding):** current program phase is *restore sane bug-free values, NOT make
  models good* — Hurdle estimator **redesign is deferred** (risk **D-26**, issues **#66/#71**).
  This dossier therefore scopes to **understand + decide + pin**, not to redesign. A redesign, if
  recommended, exits as a *separate, deferred* ADR/epic.
- **Epic #77** (this session) and its stories drive the near-term work: **#78** Story A (done),
  **#75** Story B (characterization tests), **#79** Story C (decision), **#80** Story D (assertion).
- **ADR-003** (raw target-space I/O) and **ADR-001** (ReproducibilityGate) are the standing
  contracts this work respects. The estimator decision (02_design) is the candidate **next ADR**.
- **Prior dossier** `reports/archived/2026-06-08_restore_target_compression_dossier/` (EXP-01)
  established the *plain*-model raw/identity behavior and the MSLE↔tail trade-off (risk **D-22**).
  This dossier is its Hurdle-family successor; it does **not** duplicate EXP-01.
- **Story A findings** (already executed): `reports/2026-06-13_hurdle_point_estimate_diagnosis/findings.md`
  — the founding evidence (mechanism confirmed; real-data magnitude benign). Absorbed as established
  context here; not re-run.

## 3. Document index

| # | Doc | Role | Status |
|---|-----|------|--------|
| 00 | `README` | spine / living next-actions | **living** |
| 01 | `literature` | annotated bib + gaps-to-fetch | seeded (append) |
| 02 | `design` | the estimator decision space → future ADR | **draft** (framed; awaits expert-method-review) |
| 03 | `harness_and_invariants` | crown jewel: guardrails + pre-flight checklist | **draft** (audited) |
| 04 | `roadmap` | phased, gated sequencing | **draft** |
| 05 | `analysis_plan` | first experiment, pre-registered | **stub** (awaiting `/rnd-dossier preregister`) |
| 06 | `glossary` | shared vocabulary | seeded (append) |
| 07 | `experiment_log` | append-only ledger (negatives first-class) | **empty** (no pre-registered run yet) |

## 4. Harness at a glance (see 03)

**~70% already in place.** Standing: ReproducibilityGate (ADR-001), the declared-transform
default-off-flag pattern (identity-pinned for Hurdle, D-26), the pytest suite + the real
end-to-end template (`test_target_transforms.py`), the EXP-01 production-parity readout, MSLE+MCR
as the locked metric protocol, and a postmortem norm (prior dossier). **The decisive gap is a
tail/distributional readout** — MCR is aggregate; nothing yet measures error on the escalation
tail or per-cell calibration, which is exactly where the hard gate's risk lives. Secondary gaps:
the D-13 Hurdle characterization tests (Story B) and a frozen Hurdle comparison baseline.

## 5. Current state & next actions

**Established (Story A, real-data, verified):**
- D-33 confirmed: binary stage emits hard `{0,1}` labels; probability available & discarded.
- D-37 confirmed: `target_pos` is zero-dominated (~82% on a representative fixture).
- Magnitude benign in aggregate: 6 `chunky_bunny` Hurdles MCR 0.75–1.29 (n=89,388, parquet-verified).
- D-33/D-37 de-escalated to Tier 3; residual **tail/distributional** risk remains open.
- New candidate finding: the 3 LGBM hurdles are effectively untuned (only `n_estimators`).

**Next actions (living):**
- [ ] **Build the tail/distributional readout** (03 gap C-1) — the blocker for the first experiment.
- [ ] Story B (#75): real Hurdle characterization tests (pin current behavior). *Epic prerequisite.*
- [ ] `expert-method-review` on `02_design` (the estimator decision space) before pre-registering.
- [ ] `/rnd-dossier preregister` the first experiment (EXP-01: tail/calibration readout on the 6 Hurdles).
- [ ] Decide: register the untuned-LGBM finding (`register-risk`)? — confounds the transform readout.
- [ ] Story C (#79): write the estimator decision; if "redesign", open a deferred ADR/epic (not in salvage).

## 6. Conventions

Numbered dated docs; `00_README` living, others point-in-time. **git-tracked via `git add -f`**
(reports/ is gitignored). Risks live in `reports/technical_risk_register.md` (not here). On close,
the directory moves to `reports/archived/`. One experiment = one pre-registration (05/`preregister`)
+ one log entry (07) linked to it; negatives get a postmortem with equal prominence.
