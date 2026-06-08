# Dossier: Restore Stepshifter Target Compression

**Status:** `init` complete — harness audited, gaps flagged. **Not yet ready to run** (pre-flight checklist red).
**Created:** 2026-06-08 · **Branch:** `investigation/raw-target-space-io`

---

## 1. Purpose

Stepshifter conflict-fatality models currently train on **raw, heavy-tailed counts with no target compression** (confirmed root cause of the 2026-06-04 `big_chungus` divergence — see prior art). This dossier governs the **R&D question only**: *which* declared `target_transform` (`log1p` vs `asinh` vs `identity`) restores real stepshifter models to ≈ gold-standard country-month (cm) MSLE, **validated on real data/models against the fatalities002 gold standard + all-zeros/no-change baselines** — not assumed. The validated choice graduates to a proposed ADR and the views-models config migration.

**Explicitly out of scope:** the *engineering mechanism* (the declarative `TRANSFORMS` registry + `target_transform` config key + gate validation) — that is "Half A", already governed by **ADR-003** and the **`target_transform_declarative` story**. This dossier is "Half B": the empirical transform-choice decision.

## 2. Relationship to prior work / ADRs

| Artifact | Relationship |
|---|---|
| `reports/investigation_metrics_08062026/FINDINGS.md` | **The evidence base.** Root-cause investigation that motivated this dossier. |
| `docs/ADRs/003_raw_target_space_io_contract.md` (Proposed) | The contract this validated choice fits inside (raw I/O; transform internal + declared). This dossier does **not** re-decide it. |
| `docs/stories/target_transform_declarative.md` (Proposed) | The mechanism (Half A) the validated choice graduates into. |
| risk-register **D-17** | To be reframed: cause is "no compression after revert", not stale-artifact mismatch. |
| issues pipeline-core#174, views-models#111 | Downstream governance / config migration this feeds. |
| commits `5fcfe43` (add internal log), `831d3ebc` (disable queryset ln), `08ee2eb` (revert) | The history that created the gap. |

**Exit ramp:** on validation → `promote` drafts a proposed ADR from `02_design` and archives this dossier. Don't let it orphan.

## 3. Document index

| # | Doc | Status |
|---|-----|--------|
| 00 | README (this) | living |
| 01 | literature | seeded — gaps-to-fetch listed |
| 02 | design | drafted — the transform-choice approach + rationale |
| 03 | harness_and_invariants | **drafted (crown jewel)** — audit + pre-flight checklist |
| 04 | roadmap | drafted — phased, gated |
| 05 | analysis_plan | placeholder — first experiment via `preregister` (next verb) |
| 06 | glossary | seeded |
| 07 | experiment_log | empty (append-only) |

## 4. Harness at a glance

See `03_harness_and_invariants.md`. Summary: **~60–70% of the standing harness already exists** (ReproducibilityGate config audit, `views_validate` schema guard, CI pytest, an external locked baseline = fatalities002 gold + all-zeros/no-change, MSLE protocol). **The gaps that block the first experiment:** (a) no locally runnable env (no poetry; anaconda base lacks `darts`/`xgboost`/`views_pipeline_core`), (b) the `target_transform` mechanism (Half A) unbuilt, (c) no formalized **fast offline readout on real data**, (d) eval protocol not yet locked to the exact gold-standard partition.

## 5. Current state & next actions

- [x] `init` — scaffold + harness audit
- [ ] **Fetch prior art** (`01`): fatalities002 retrospective PDF + model appendix (have locally), Hegre et al. 2020/2022, transform/zero-inflation literature → via `library` skill
- [ ] **`expert-method-review` on `02_design`** (the transform-choice design) before pre-registering
- [ ] **Close pre-flight blockers** (`03` §D): build the fast offline readout; secure real cached queryset data for ≥1 representative model; lock the eval protocol
- [x] **`preregister`** EXP-01 → `05_preanalysis_exp01.md` (identity/log1p/asinh/Tweedie on real brown_cheese data; multi-component readout; F1–F5)
- [ ] **Execute EXP-01** on brown_cheese real data → `/rnd-dossier log` (incl. negatives). Real env confirmed runnable (`views_pipeline` conda)
- [ ] *(execute — outside this skill)* → `log` → `status` → `promote`

## 6. Conventions

Numbered dated docs; `00_README` living, rest point-in-time. Append-only `07`. Negatives are first-class (postmortem template). Git-tracked on branch `investigation/raw-target-space-io`. On close → `reports/archived/`.
