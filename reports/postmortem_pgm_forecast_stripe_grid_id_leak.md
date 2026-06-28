# Post-mortem: the PGM forecast "stripe" — a grid-id leak stacked on a UCDP data artifact

**Date:** 2026-06-28 · **Severity:** high (silent, plausible-looking corruption of PGM forecast maps in a production ensemble) · **Status:** Problem B (model) resolved on `main`; Problem A (data) handed off to views-datafactory, open · **Author:** Simon (+ Claude Code)

> **Convention:** follows the views-platform post-mortem discipline — every claim is backed by a named artifact (commit, PR, register entry, figure, run), and **negatives are first-class**: the diagnostic missteps below are recorded with the same prominence as the wins, and not softened. **Blameless:** the focus is the system (library defaults, test gaps, stale assessments) that let a silent error reach a production map and nearly let a regression reach `main`, not the people.

---

## 1. Summary

The June 2026 PGM forecast maps showed a **dead-straight horizontal "stripe" of elevated predicted violence at ~13.25°N**, running across the continent through empty desert. It turned out to be **two independent bugs stacked**:

- **Problem A (data):** UCDP GED records the Tigray war as **low-spatial-precision "summary" events** (`where_prec=5`, "northern Ethiopia") that get geocoded to a single centroid cell. One PRIO-GRID cell (`148759`, Mekelle) carries **~273,076 deaths** — 3.4× the next-worst cell on the continent. Upstream of stepshifter; not a stepshifter bug.
- **Problem B (model):** stepshifter silently fed the **PRIO-GRID cell id to the regressor as a feature** (a darts `use_static_covariates=True` default). Because ids are row-major, the tree memorised the id-neighbourhood of that extreme cell and painted an elevated baseline across its whole latitude — the stripe.

Problem B was diagnosed, **reproduced on the real panel**, fixed (force `use_static_covariates=False`), verified, and landed on `main` (PRs #95, #96; register **D-41**). Problem A was characterised end-to-end and handed to views-datafactory with a full report. **The fix removes the stripe; it does not fix the data, and it changes model behaviour, so the retrained models still need skill re-evaluation** (see §8).

## 2. Impact

- The 2026-06-26 production forecast run (PGM ensemble `skinny_love`; CM ensemble `pink_ponyclub`) produced **visibly wrong maps** — a continent-spanning artificial hotline that any reader would (correctly) distrust, but that raised **no error**.
- **Every PGM-level model** trained on this target is affected by Problem A's hot cell; **every tree-based stepshifter constituent** is affected by Problem B. The two compound: Problem A supplies the outlier, Problem B smears it.
- **Horizon-dependent:** Problem B is invisible at short lead times and severe at long ones — the visible stripe sat at month 583 = **horizon 26**.
- Duration: latent for as long as PGM tree constituents have been trained via `from_group_dataframe` with the darts default (pre-dates this investigation; not bounded here).

## 3. Timeline

| Date | Event | Artifact |
|---|---|---|
| 2026-06-26 | Production run: `skinny_love` (PGM) + `pink_ponyclub` (CM). Stripe appears on the PGM map | `predictions_forecasting_20260626_*` |
| 2026-06-27 | Angelica flags the stripe + a hole + "values look low"; one LGBM feature-name warning noted | Slack / images |
| 2026-06-27 | First diagnosis **wrongly** blames the reporting/plotting tool (raster render shows no line) | §7.1 |
| 2026-06-28 | Re-examination of fixed-scale per-month maps reveals the line **is** in the data → row 207 / lat 13.25 | `DIAGNOSIS_FOR_ANGELICA/07…` |
| 2026-06-28 | Traced to cell `148759`; UCDP source confirmed (`where_prec=5`, 121,848 + 113,368); reconstruction **r=0.99998** | datafactory report §4 |
| 2026-06-28 | Problem B identified (grid-id static covariate); **then prematurely walked back** after a near-horizon test failed to reproduce | §7.3 |
| 2026-06-28 | Full-history far-horizon reproduction **confirms** B: step-36 row-floor **22.6× → 0.87×** with the fix | verify_full run |
| 2026-06-28 | Fix shipped: #95 → `development`; register D-41; datafactory report handed off | `e10ec32`, #95 |
| 2026-06-28 | Near-miss: stale assessment recommended merging all of `development` → `main`; caught the unreleased pipeline-core 3.x pin before it reached main | §7.5 |
| 2026-06-28 | Fold-in release: #96 → `main` with pipeline-core pinned back to 2.x; CI green on 2.x | `eda9103`, #96 |

## 4. Root cause

**Two unrelated causes that happened to share a symptom.**

- **Problem A — encoding of spatial uncertainty.** UCDP must attach a single lat/lon even when it only knows "northern Ethiopia" (`where_prec=5`); that centroid lands in one cell. The ingestion **temporally** distributes such events across their months (`even_split`) but has **no spatial distribution** — so a whole region's multi-year toll piles into one 55 km cell. Real deaths, wrong place.
- **Problem B — a library-default footgun, not a modelling choice.** `_prepare_time_series` groups the panel with `TimeSeries.from_group_dataframe(group_cols=priogrid_gid)` purely to split it into per-cell series. darts *also* attaches the group key as a **static covariate**, and darts regression models use static covariates **as features by default** (`use_static_covariates=True`). So `priogrid_gid` reached XGBoost as a numeric column. Ids are row-major (`id=(row-1)*720+col`), so a tree split on the id memorises a latitude band — and the extreme Problem-A cell makes that band a visible stripe. **Nobody chose to use the id as a feature; two defaults composed into it.**

## 5. Detection

**A human eyeballing a map — no automated signal anywhere.** Nothing failed: the target is "valid", the features are "valid", the predictions are finite, the report rendered cleanly. The only alarm was Angelica noticing a straight line that geography forbids. The one warning that *did* fire (sklearn "X does not have valid feature names") was **benign noise** from darts and a red herring (§7.2). The per-cell evidence that localised the bug (row-207 floor elevated only at long horizons) was reachable the whole time but required a directed investigation to surface.

## 6. Resolution

1. **Problem B (this repo):** a `_new_regressor` helper forces `use_static_covariates=False` (overriding any hyperparameter); the id stays attached only as a **label** for the prediction→cell assembly, never as a feature. Verified on the real `caring_fish` panel (full 437-month history): step-36 row-floor **22.6× → 0.87×**. Registered **D-41** (Tier 1). Shipped #95 → `development`, #96 → `main`.
2. **`main` dependency hygiene:** `development` tracks the unreleased pipeline-core 3.0.0 over git; the release PR (#96) pinned that single line back to `>=2.0.0,<3.0.0` so `main` stays on the published 2.x and CI validated on 2.x.
3. **Problem A (upstream):** characterised end-to-end and handed to views-datafactory with a full report + proposed `spatial_distribution` strategy (`reports/2026-06-28_ucdp_spatial_distribution_of_low_precision_summary_events.md` in that repo). Not fixable in stepshifter.

## 7. What went wrong — contributing factors (negatives first-class)

1. **First diagnosis was wrong: blamed the plotting tool.** A direct raster render of the prediction file showed no line, so the initial call was "it's the reporting layer (Plotly/GeoJSON)." That was confidently stated and **wrong** — the render simply used a colour scale that hid a low-magnitude-but-real line. Lesson: a single visualisation that *doesn't* show an artifact is not evidence the artifact is absent.
2. **A benign warning nearly misdirected.** The LGBM "no valid feature names" warning was investigated as a candidate cause; it was darts noise. Time spent ruling it out, but worse, it was *plausible* enough to anchor on.
3. **The correct diagnosis was prematurely walked back.** Once Problem B (grid-id leak) was identified by a sound logical argument, a **too-shallow reproduction** (toy models, then a 158-month **step-1** fit) failed to reproduce it — and the diagnosis was retracted as "unconfirmed, probably not the id." This was an **over-correction**: the logic was right; the *test horizon* was wrong (§7.4). The retraction would have stalled the fix had the deeper test not been run.
4. **The horizon-dependence trap.** Problem B is **invisible at step 1, severe at step 36**. Three separate reproduction attempts (two toys + one full-data step-1 run) all failed for the *same* reason — they tested the near horizon, where the covariate signal still dominates the memorised id. Only the full-history **far-horizon** test reproduced it. A forecast artifact can live entirely in the tail of the horizon.
5. **A regression nearly reached `main` on a stale assessment.** The first main-merge plan ("development → main is safe, no dependency change") rested on a `pyproject` diff that came back empty — because the unreleased-pipeline-core-3.x git pin was a *local-only* commit not yet on `origin/development` when the diff ran. Recommending that merge would have flipped `main` onto **unreleased 3.x** and regressed Angelica's environment. Caught only because the human had independently flagged cross-repo version skew as a fear, prompting a re-check.
6. **Tests cannot catch this class.** The existing `test_fit`/`test_predict` mock the executor and the model classes and assert call-wiring, not numerical behaviour — they stayed green through the entire incident (same gap as **D-13**). The new tests pin the *construction* (`use_static_covariates=False`), which is reliable, but a true behavioural test needs a real fit **at a far horizon** — too heavy for CI.
7. **No hot-fix path.** Problem B is training-time: production stays striped until every constituent is **retrained** and the ensemble rebuilt. There is no way to remediate already-frozen artifacts.

## 8. The honest caveat (not softened)

- **Stripe gone ≠ data correct.** The fix removes Problem B's *smear*. Problem A's *source* — a quarter-million deaths in one Mekelle cell — **remains** in the data Angelica retrains on. The retrained map will still have an over-concentrated (but no longer continent-spanning) hotspot there. The data fix is upstream and not yet done.
- **Stripe gone ≠ skill validated.** Removing the id feature **changes model behaviour**, not only the artifact. The retrained constituents must be **re-evaluated for skill, weighting far horizons** — they are not assumed equal, and that evaluation has not been run.
- **Not yet confirmed in production.** Verification was on a reconstructed constituent panel, not a live ensemble run, and the author lacked the UU VPN to re-pull the exact training feed. Final confirmation waits on Angelica's rerun.

## 9. What went well

- **Domain intuition was the decisive control.** The human's "*why a line and not a splash?*" rejected the comfortable "it's just the data" stop and forced the second (model) bug into the open. A splash is what a spatial model does; a 1-cell-thick axis-aligned line is what an indexing/feature artifact does.
- **Version-skew caution caught a near-miss.** The same human's pre-stated fear about cross-repo dev dependencies is the only reason the stale-assessment merge didn't reach `main`.
- **Reconstruction without the broken source.** With the new viewser inaccessible, Problem A's mechanism was proven by rebuilding the cell's monthly values from **live UCDP** alone — matching the training file at **r=0.99998**.
- **Decisive real-data verification before shipping.** The fix wasn't shipped on logic alone; the full-history far-horizon experiment showed 22.6× → 0.87× first.
- **Clean release under a dependency hazard.** The fold-in PR kept `main` on 2.x, never transiently held 3.x, and let CI validate the shipped state.
- **Two bugs were correctly separated and routed** to their right owners (model → this repo; data → datafactory) rather than conflated into one mushy fix.

## 10. Lessons

1. **Identifiers must never be model features.** Guard against silent promotion of a grouping key to a covariate; the raw cell id has no metric meaning. Legitimate spatial signal belongs in real covariates (lat/lon, positional encoding), chosen deliberately.
2. **Forecast artifacts can be horizon-dependent — test the far horizon.** A near-horizon reproduction that comes up clean is not evidence of absence.
3. **A diagnosis backed by sound logic shouldn't be retracted on a single failed reproduction** — first check whether the *test* exercised the failure's regime.
4. **Don't trust a stale diff before a `→ main` merge.** Re-fetch and re-assess `pyproject`/git-source dependencies immediately before the merge; cross-repo version skew is a live hazard, not a hypothetical.
5. **Domain "shape" reasoning is a debugging instrument.** "Why this geometry?" localised the bug faster than any feature scan.
6. **Separate stacked causes explicitly.** When one fix could mask another, diagnose each cause independently.

## 11. Action items

| Item | Owner | Tracking |
|---|---|---|
| Retrain constituents + rebuild ensemble; confirm the stripe is gone in production | Angelica | pending rerun |
| **Skill re-evaluation** of the retrained models, weighting far horizons | research | §8 |
| views-datafactory **spatial-distribution** strategy for `where_prec ≥ 4` summary events | datafactory | report `2026-06-28_ucdp_spatial_distribution…` |
| A real **far-horizon behavioural** test for the id-leak class (the unit tests pin construction only) | stepshifter | follow-up to #75 |
| Defense-in-depth: warn/fail if any static covariate reaches the regressor feature set | stepshifter | new |
| Re-pin `development` → PyPI 3.x once views-pipeline-core 3.0.0 publishes | platform | `eda9103` note |

## 12. Evidence index

Fix PRs **#95** (→ development), **#96** (→ main); commits `e10ec32` (fix), `eda9103` (2.x pin-back); register **D-41**. Diagnosis figures and the narrative set in `~/brain/0_inbox/angelica_files/DIAGNOSIS_FOR_ANGELICA/` (`01_symptom…` → `07_problemB_gridID_leak.png`) and the blast-radius CSV. Problem A handoff: `views-datafactory/reports/2026-06-28_ucdp_spatial_distribution_of_low_precision_summary_events.md`. Root mechanism: `stepshifter.py` `_prepare_time_series:127` (`from_group_dataframe`), `_new_regressor` (the fix). Verification: full-history far-horizon reproduction (step-36 22.6× → 0.87×). UCDP source: GED 24.1, `country=530`, events 463137 / 463131 (`where_prec=5`).
