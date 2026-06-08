# 04 — Roadmap (phased, gated)

**Date:** 2026-06-08 · revise as phases complete.

## Phase 0 — Foundations (current; gates everything)
- [x] Root-cause investigation (FINDINGS.md) — **done**
- [x] Dossier init + harness audit — **done**
- [ ] `expert-method-review` on `02_design` (transform-choice design)
- [ ] Fetch prior art (`01`) via `library`
- **Gate G0:** design reviewed + literature gaps closed.

## Phase 1 — Build the program harness (the `03` §C gaps)
- [ ] Runnable env (poetry/CI with lightgbm/xgboost/sklearn) — **operational blocker**
- [ ] Secure real cached queryset data for ≥1 representative `big_chungus` model
- [ ] Fast **offline readout** (real data, 3 arms, cm MSLE vs gold + baselines)
- [ ] Lock the eval protocol to the fatalities002 comparison
- [ ] Parity test (identity == raw) + round-trip/monotonic test
- **Gate G1 (pre-flight `03` §D green):** `status` may now call the program ready.

## Phase 2 — EXP-01: offline transform pre-screen
- [x] `preregister` EXP-01 → `05_preanalysis_exp01.md` (identity/log1p/asinh/Tweedie; F1–F5; multi-component readout)
- [ ] Execute offline readout (outside this skill) → `log` (incl. negatives)
- **Decision D2:** which transform(s) beat the all-zeros baseline and approach gold; drop the rest.

## Phase 3 — EXP-02: real-pipeline validation
- [ ] `preregister` EXP-02 on the surviving transform(s) across one model per family (Hurdle / flat XGB/LGBM/RF) via the real pipeline (requires Half A mechanism or a controlled offline equivalent)
- [ ] Execute → `log`
- **Decision D3:** per-family `target_transform` choice, validated vs gold.

## Phase 4 — Promote
- [ ] `promote`: draft proposed ADR from `02_design`; route risks to `register-risk` (reframe D-17); cross-link issues #174/#111; archive dossier
- Hand the validated choice to the views-models config migration (#111) and Half A's mechanism PR.

## Dependency notes
- Phase 1's env/data blocker is the critical path. Phase 3 depends on Half A (mechanism) **or** an offline equivalent that faithfully mimics `_process_data`/`_predict_by_step` with the transform.
- Half A (mechanism) proceeds in parallel as a normal PR under ADR-003 — not gated by this dossier, but Phase 3 consumes it.
