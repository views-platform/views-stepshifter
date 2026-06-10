# 08 — Forensic reconstruction: what actually happened with `big_chungus`

**Date:** 2026-06-09. **Purpose:** a defensible, evidence-linked account of the stepshifter breakage — what broke, what did **not**, and how we know. Written to brief from. Every claim here is backed by a named artifact (report cell, run, or commit).

---

## TL;DR (the one paragraph)

The 2026-06-04 `big_chungus` ensemble scored **MSLE 2.519 — worse than an all-zeros baseline (2.147)**. The breakage is **real but narrow**: only the **plain** stepshifter constituents (XGB/XGBRF/LGBM) are broken — they train on the **raw, heavy-tailed conflict target with no compression**, so they over-predict massively on the ~90 % zero cells (MSLE 4–5). The **Hurdle** constituents (MSLE ~0.8–1.1) and the **deep-learning** constituents (~0.4–0.6) were **healthy all along**. The ensemble looks broken because it averages the broken plain members in, dragging the whole thing above the zero baseline. **The original investigation over-claimed "all 19 constituents broken"; the report's own per-constituent table contradicts that.** We confirmed Hurdle's health twice, independently.

---

## The alarm

- **Artifact:** `report_calibration_model_20260604_230304_lr_ged_sb.html` (WandB run `2689xjtl`, owner Dylan/smellycloud, 2026-06-04 23:01, pipeline 2.3.0).
- `big_chungus` (cm, `lr_ged_sb`) ensemble **MSLE 2.519 > all-zeros 2.147 > fatalities002 gold standard 0.835**. An ensemble doing worse than predicting all-zeros is the red flag.

## The per-constituent leaderboard (the decisive evidence the first pass missed)

That same report contains a **per-constituent metrics table** (time-series-wise `MSLE_mean`). Read directly out of the HTML:

| Constituent type | Models & MSLE (June 2026-06-04) | Range |
|---|---|---|
| **Deep-learning** (internal asinh) | new_rules 0.43, smol_cat 0.44, revolving_door 0.51, elastic_heart 0.57 | **0.43–0.57** ✅ |
| **Hurdle** (two-stage) | fluorescent_adolescent 0.79, **fast_car 0.85**, green_squirrel 0.85, twin_flame 0.93, high_hopes 1.05, little_lies 1.08 | **0.79–1.08** ✅ |
| *all-zeros baseline* | zero_cmbaseline | 2.147 |
| **Ensemble** | **big_chungus** | **2.519** ⚠️ |
| **Plain** (raw target) | counting_stars 1.88, demon_days 1.89, good_riddance 2.04, heavy_rotation 2.56, ominous_ox 2.56, yellow_submarine 2.80, teen_spirit 3.40, national_anthem 3.47, car_radio 4.04, **brown_cheese 4.20**, plastic_beach 4.20, popular_monster 5.15 | **1.9–5.2** ❌ |

**Model type predicts brokenness exactly:** DL (compresses internally) best → Hurdle (architecture gates zeros) sane → plain (raw) broken.

## Why the Hurdles appear in a "broken-models" report

Because the report is the **ensemble** report, and its leaderboard lists **all 23 constituents** — including the 6 Hurdles. But their **individual** scores in that very table were **sane (0.79–1.08)**. They appear there as *members of a broken ensemble*, not as broken models. The ensemble is broken because the **plain** members (4–5) dominate the average.

## Two independent corroborations (the "suddenly fixed?" check)

We were rightly suspicious that Hurdle "suddenly looked fixed." It did not — it was never broken:

| model | June report (2026-06-04) | Fresh run (2026-06-09, clean fetch via `main.py -r calibration -t -e`) |
|---|---|---|
| **fast_car** (Hurdle) | **0.847** | **0.784** (Δ = retraining variance) |
| **brown_cheese** (plain) | **4.197** | **4.269** |

Same models, two independent measurements ~3 days apart → consistent. **Nothing was fixed; nothing changed.** And critically: **this session did not modify HurdleModel's prediction logic** — the only change to `hurdle_model.py` is a 10-line `__init__` guard that *rejects* non-identity transforms (it does not alter fit/predict). So an identity Hurdle is byte-identical to before our work.

## Root cause (plain models only)

- **`5fcfe43`** (2025-11-20): added an internal `np.log1p`/`np.expm1` to stepshifter; the queryset `.ln()` was disabled at the same time to avoid double-compression.
- **`08ee2eb`** (2026-04-11): reverted `5fcfe43` — but **the queryset `.ln()` was never re-enabled.** → 144 days where plain stepshifters had **zero target compression anywhere**.
- The plain regressor minimises squared-log error on a ~90 %-zero target by... over-predicting onto the zeros (mean prediction ~16–60 where truth is 0). MSLE blows up to 4+.
- **Hurdle is immune by architecture:** its binary stage predicts P(conflict)≈0 for the zero cells, so they never reach the raw positive-stage regressor. **DL is immune** because it compresses internally (asinh).

## What we did about it (this dossier)

1. **Built** the declared `target_transform` mechanism (views-stepshifter#52): transform once in `_process_data`, invert once at the `predict()` boundary; gate-enforced; HurdleModel/ShurfModel pinned to `identity` (deferred). 41 tests.
2. **Migrated** all 37 configs to `target_transform: "identity"` (views-models#113) — behaviour-preserving.
3. **EXP-01** (#54, #57–#61): validated through the real pipeline. identity reproduces 4.27; log1p/asinh drop MSLE to ~0.52 **but** the pre-registered **F4 falsifier fired** — compression wins headline MSLE by ~2.5× *worsening the escalation tail* and collapsing calibration. So compression is **necessary but not sufficient** — a salvage step, not an optimisation. (Reference frame: 60-mo average baseline 0.58, LOCF 0.81.)

## Corrections to the original investigation

- **Over-claim:** `FINDINGS.md` stated "all 19 `big_chungus` constituents are broken (raw)." The report's own per-constituent table shows only the **plain** constituents are broken; **Hurdle and DL were healthy.** The per-constituent table was present but not read; brokenness was inferred from "trains raw" rather than measured.
- **Consequence for the fix:** restoring `big_chungus` to health = **fixing the plain constituents** (the raw-target bug). Hurdle/Shurf are **not** the cause of *this* ensemble's failure.

## Hurdle robustness verdict (#67 — 2026-06-09)

**Verdict: the HurdleModel (identity) path is ROBUST.** All three DoD criteria met:
1. **Code audit (#63):** the two-stage path (`binary × positive`, **no inverse**) is sound under identity (forward is a no-op, output is raw); non-identity is correctly forbidden by the gate + `__init__` guard (it would otherwise emit silent log-space output); the older `_resolve_clf_model` refactor is scale-irrelevant. No new silent-correctness risk beyond the tracked D-24/D-25.
2. **Replication (#64):** three independent fresh cm Hurdle runs — fast_car **0.784**, twin_flame **1.081**, little_lies **1.017** — all sane and consistent with the June report (0.79–1.08). None insane. Health replicates across **3 models × 2 timepoints**.
3. **Tests (#65):** the identity path is locked by passing tests, including a new **real (non-mocked) end-to-end** fit→predict lock; full transform suite 42 passed.

**Conclusion:** Hurdle is *mediocre-but-not-broken, and not brittle* — salvage-complete for the Hurdle family. It was **never** the cause of the `big_chungus` breakage (that's the plain constituents). Enabling a *non-identity* transform on Hurdle (the two-stage inverse design) is **deferred** to the optimization regime (#66). **Next salvage target: ShurfModel** — the one place with a distinct, identified bug.

## Open threads (separate from this reconstruction)

- **ShurfModel `log_target=True` bug — CONFIRMED (D-27, Tier 1; diagnosed #68, 2026-06-09).** The FINDINGS-claimed site (`shurf_model.py:210-212` ordering) is **dead code** (the mutated column is dropped before output). The real mechanism is a **training-space ↔ sampling-space mismatch**: `fit` trains the positive stage on the **raw** target (identity-pinned `_process_data`, `:38,65-68`), but the `log_target=True` sampler (`:156`) applies `expm1` to that raw prediction. A fresh `fourtieth_symphony` run logged `RuntimeWarning: overflow encountered in expm1` → `expm1` of a raw conflict count **overflows float64 → `inf`** predictions. Exposed model: `fourtieth_symphony` (the only `log_target=True` Shurf; **not** a `big_chungus` constituent, so not part of the ensemble breakage). Distinct from the plain bug (over-prediction) and from Hurdle (which was fine). Fix: #69 (fastest salvage = `log_target=False` onto the correct `:180` sampler). **Genuine bug — unlike the Hurdle over-claim, this one is real.**
- **Ensemble aggregation:** an equal-weight mean that includes broken plain members will be dragged bad; worth confirming how `big_chungus` weights constituents (would a robust/weighted aggregation have masked the plain breakage?).
- **wandb depth:** run `2689xjtl` (Dylan/smellycloud) holds the exact constituent configs + training timestamps. Not needed for this reconstruction (the local report sufficed), and wandb is **not authenticated** in this env — available for artifact-level confirmation on request.

## Evidence index

- Report: `reports/investigation_metrics_08062026/report_calibration_model_20260604_230304_lr_ged_sb.html` (per-constituent table; constituent list; owner/run metadata).
- Prior analysis (with the over-claim): `reports/investigation_metrics_08062026/FINDINGS.md`, `TASK_2_views-stepshifter.md`.
- Fresh production runs (2026-06-09): brown_cheese identity/log1p/asinh + cm baselines (#57–#62); fast_car cm Hurdle diagnosis (`/tmp/hurdle_diag_fast_car.log`).
- EXP-01 record: `07_experiment_log.md`; driver `exp01_brown_cheese.py`; readout `exp01_readout.py`; reconciliation `exp01_reconcile.py`.
- Risk register: D-17 (root cause, production-validated), D-22 (MSLE selection, production-validated), D-26 (Hurdle/Shurf deferred).
