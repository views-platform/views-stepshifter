# Stepshifter target-compression fix — team report

**Date:** 2026-06-10 · **Shipped:** views-stepshifter **1.3.0** (on PyPI) · **Author:** Simon (+ Claude Code)
**Evidence trail:** this dossier (`08_forensic_what_happened.md`, `07_experiment_log.md`, `09_chunky_bunny_rerun_log.md`), risk register `technical_risk_register.md`, ADR-003, PRs #74/#76, issue #75.

---

## TL;DR

The **plain** stepshifter cm models (XGB/XGBRF/LGBM) were silently broken: they trained on the **raw, ~90 %-zero conflict target with no compression**, so they massively over-predicted on the zero cells — cm MSLE **4–5**, *worse than predicting all-zeros* (2.147). That dragged the `big_chungus` ensemble to **2.519** (also worse than all-zeros).

We diagnosed it, built a declared **`target_transform`** mechanism, set the plain models to **`log1p`**, fixed two secondary bugs, and **released views-stepshifter 1.3.0 to PyPI**. The rebuilt ensemble (`chunky_bunny`) scores **cm MSLE 0.588** — sane.

> ⚠️ **This is a SALVAGE, not an optimization.** `log1p` restores sane *headline* MSLE, but our pre-registered tail test **fired**: the compressed models are **~2.5× worse on the escalation tail** (they under-predict the high-fatality cells — the operational product). The models are now **mediocre-but-bug-free, not good.** Making them good is the next phase, not this one.

---

## What broke, and what didn't

The breakage was **real but narrow** — model *type* predicts it exactly (per the 2026-06-04 report's own per-constituent leaderboard):

| Constituent type | cm MSLE | Status |
|---|---|---|
| **Deep-learning** (compresses internally, asinh) | 0.43–0.57 | ✅ healthy all along |
| **Hurdle** (binary gate × positive stage) | 0.79–1.08 | ✅ healthy all along |
| *all-zeros baseline* | 2.147 | — |
| **`big_chungus` ensemble** | **2.519** | ⚠️ dragged down by the plain members |
| **Plain** (raw target, no compression) | 1.9–5.2 | ❌ **broken** |

The original investigation over-claimed "all 19 constituents broken." The report's own table contradicts that: **only the plain models broke.** Hurdle and DL were fine (confirmed by independent re-runs).

## Root cause (plain models only)

- **`5fcfe43`** (2025-11-20) added an internal `log1p`/`expm1` to stepshifter; the queryset's compensating `.ln()` was disabled at the same time.
- **`08ee2eb`** (2026-04-11) reverted `5fcfe43` — **but the queryset `.ln()` was never re-enabled.**
- → **144 days with zero target compression anywhere.** A squared-log-error regressor on a 90 %-zero target minimises loss by over-predicting onto the zeros (mean prediction ~16–60 where truth is 0) → MSLE blows up to 4+.
- **Hurdle is immune by architecture** (its binary stage zeroes out the zero cells before the raw regressor sees them); **DL is immune** because it compresses internally.

## What we did

1. **Built** a declared `target_transform` mechanism (#52): a closed registry `{identity, log1p, asinh}`, applied once at fit and inverted once at the `predict()` boundary, **enforced by the reproducibility gate**. Hurdle/Shurf are gate-pinned to `identity` (their non-identity path is deferred).
2. **Migrated** all 37 views-models stepshifter configs — plain → `log1p`, Hurdle/Shurf → `identity`.
3. **Validated empirically (EXP-01, on real `brown_cheese`, through the production pipeline):**

| arm | cm MSLE | tail MSLE (obs>100) | calibration | mean-pred on true-zeros |
|---|---|---|---|---|
| identity (= the bug) | 4.269 | **1.796** | 2.46× over | 16.04 |
| **log1p** (the fix) | **0.521** | 4.199 | 0.27× under | 0.55 |
| asinh | 0.515 | 4.646 | 0.24× under | 0.49 |

   `identity` reproduces production 4.27 exactly (diagnosis confirmed). `log1p` slashes headline MSLE 4.27 → 0.52.
4. **Fixed two secondary bugs:**
   - **ShurfModel `log_target=True`** (D-27): trained the positive stage on the raw target but applied `expm1` to it → float64 overflow → `inf`. Gate now rejects it.
   - **DL `revolving_door`/NHiTS** diverged on the +12-month partition (predictions → 10²¹). Resolved by the **r2darts2 library update + tuning (RevIN + SpotlightLossLogcosh)** — not a config tweak.
5. **Rebuilt the ensemble** (`chunky_bunny`, all 23 constituents fresh): **cm MSLE 0.588** (was 2.519). Reference frame: 60-mo-average baseline 0.58, LOCF 0.81.

## ⚠️ The caveat the team must internalise

Our pre-registered **F4 falsifier fired**: selecting on MSLE alone would crown a *worse* model. Compression swaps **over-prediction-on-zeros** for **tail-blindness + severe under-forecasting**:
- tail-conditional MSLE: 1.80 (raw) → **4.2–4.6 (compressed)** — ~2.5× worse;
- calibration: 2.46× **over** → ~0.25× **under** (predicts ~¼ of the actual mass).

MSLE on a 90 %-zero target rewards fitting the zeros; in log space the heavy tail is compressed, so the model predicts near-zero almost everywhere — great for the mass, **blind to escalation, which is the policy product.** So: **compression is necessary but not sufficient.** `0.588` means *bug-free*, **not** *good*. Do not read it as "the models now forecast escalation well" — they don't.

## What shipped

- **views-stepshifter 1.3.0** on PyPI (consumers pin `<2.0.0`, so they receive it).
- **Breaking change:** the gate now *requires* a `target_transform` key on stepshifter configs (`MissingHyperparameterError` otherwise). All views-models configs are migrated; **other consumers must add `target_transform` before upgrading.**

## Open follow-ups (tracked, none blocking)

- **EXP-02 — the actual optimization** (the real next step): a **tail/quantile-aware selection metric** (not MSLE alone, register D-22), the deferred Hurdle two-stage, and/or a count likelihood (Tweedie/NB, D-23). `log1p` > `asinh` if a transform is used at all (better tail of the two).
- **D-30** (Tier 2): no fail-loud guard on DL prediction magnitude — a diverged DL constituent can silently inflate an ensemble (caught manually this time).
- **#75:** replace the 4 xfail'd mock-brittle stepshifter/Hurdle tests with real behavioral tests (+ first-ever ShurfModel/darts coverage).
- **D-14:** no `poetry.lock` — CI installs unpinned.

## One-line summary for the standup

> *We found and fixed a 144-day-old bug where plain stepshifters trained on the raw conflict target with no compression (MSLE 4+, ensemble 2.5 — worse than all-zeros). Shipped `target_transform`/`log1p` as views-stepshifter 1.3.0; ensemble back to 0.588. Caveat: this restores sane values but **not** tail/escalation skill — that's the next phase (EXP-02), and MSLE alone must not be the selection metric.*
