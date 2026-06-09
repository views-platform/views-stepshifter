# 07 — Experiment Log (append-only)

**Date opened:** 2026-06-08. Append-only Popperian ledger. Every entry links its pre-registration (`05` / a `preregister` artifact) and states its verdict against the **pre-committed falsifiers**. Negatives are recorded with the same prominence as wins (use the postmortem template). Newest at the bottom; never edit past entries except to add a cross-link.

Entry template (from the skill):
```
### EXP-NN · <title> · <date> · <SUCCESS|FALSIFIED|INCONCLUSIVE>
- Plan (pre-reg): <link>
- Variable: <the one thing changed>
- Driver / artifact / results: <script · artifact ts · location>
- Readout: <fast probe> → <full metrics vs locked baseline>
- Verdict vs falsifiers (plan §5): <which fired / none> ⇒ <verdict>
- Decision: <next step per plan §7>
```

---

## EXP-01 · target compression through the real declared mechanism · 2026-06-08 · MIXED — diagnosis **CONFIRMED**, naive-transform fix **FALSIFIED** (F4 fired)

- **Plan (pre-reg):** [05_preanalysis_exp01.md](05_preanalysis_exp01.md) (amended: arms {identity, log1p, asinh}; plain only; no Tweedie).
- **Prerequisite met:** ran through the **real centralized mechanism** (views-stepshifter#52 — `target_transform` applied *inside* `StepshifterModel`: forward in `_process_data`, inverse at the `predict()` boundary), **not** the boundary hack. The earlier smoke/boundary run does **not** count (per the amendment).
- **Variable:** `target_transform` ∈ {identity (control), log1p, asinh}; all else held constant (brown_cheese, XGBRFRegressor, calibration train[121,444]/test[445,492], steps 1–36, n_estimators 300, n=89,388).
- **Driver / measurement / results:** `exp01_brown_cheese.py` · readout via the unit-tested `exp01_readout.py` (10 tests green) · raw log `/tmp/exp01_full.log` · τ=100.
- **Readout (full, n=89,388):**

  | arm | MSLE | tail MSLE (obs>100) | mean-pred on true-zeros | calibration ratio | mean pred |
  |---|---|---|---|---|---|
  | identity | 4.269 | **1.796** | 16.04 | 2.463 (over) | 60.7 |
  | log1p | 0.521 | 4.199 | 0.55 | 0.270 (under) | 6.65 |
  | asinh | **0.515** | 4.646 | 0.49 | 0.237 (under) | 5.85 |

  (all-zeros baseline MSLE = 2.147; mean actual = 24.6)

- **Verdict vs falsifiers (plan §5):**
  - **F3 (parity/control) — did NOT fire ✅:** identity = **4.269 ≈ production 4.27**, and reproduces the **mean-16-on-true-zeros** signature (16.04). Harness validated; the diagnosis (production target trained **raw/identity**) is **confirmed on real data through the real mechanism**.
  - **F1 (compression ineffective) — did NOT fire ✅:** log1p/asinh both beat all-zeros (0.52/0.51 ≪ 2.15). Compression slashes headline MSLE 4.27 → ~0.52.
  - **F2 (works-but-degenerate) — PARTIALLY supported ⚠️:** the MSLE win is largely a near-zero-everywhere effect — calibration collapses to 0.24–0.27 (predicts ~¼ of actual mass), mean-on-true-zeros ~0.5.
  - **F4 (MSLE misleads) — FIRED 🔴:** the MSLE winner (asinh) is **worst** on the escalation tail (4.646 vs identity 1.796); **both** compression arms are **~2.5× worse on the tail** than identity. Per §7, the tail metric overrides MSLE.
- **Decision (plan §7):** F4 fired → **compression is NECESSARY but NOT SUFFICIENT.** It swaps over-prediction-on-zeros for tail-blindness + severe under-forecasting. **Do not naively flip plain production models to log1p/asinh** — this re-scopes views-models#114. Among the MSLE-passers the tail-better arm is **log1p** (4.199 < 4.646) — so *if* a transform is used, **log1p over asinh**. **Next:** a design review / EXP-02 on **tail-aware** fixes (a tail/quantile-aware selection metric — not MSLE alone, D-22; the deferred Hurdle two-stage; or a count likelihood Tweedie/NB, D-23) **before** any production flip. Register: D-17 confirmed via the real mechanism; D-22 strengthened.

### Postmortem — F4 fired (the naive-transform fix is operationally misleading)
- **Predicted vs saw:** the skepticism ledger anticipated this exactly ("compression may help zeros but hurt the escalation tail — the policy product"). It did: tail-conditional MSLE went 1.80 (identity) → 4.2–4.6 (compressed), while calibration flipped from 2.46× **over** to ~0.25× **under**. MSLE-alone would have crowned asinh a triumphant fix.
- **Why:** MSLE on a 90%-zero target rewards fitting the zeros; in log space the heavy tail is compressed, so the regressor minimises loss by predicting near-zero almost everywhere — great for the mass, blind to escalation (the cells that matter operationally).
- **Ad hoc rescue resisted:** we do **not** downgrade F4 to an edge case or re-select on MSLE. The pre-committed rule stands — the tail metric overrides.
- **What this changes:** the program's centre of gravity moves from "pick the transform" to "fix the tail/calibration." The transform mechanism (#52/#113) remains correct, needed infrastructure (identity parity proves it), but it is **not, by itself, the fix.**
- **Threats to validity (still open):** single model (brown_cheese), single partition, single metric family; τ=100 is one tail definition. EXP-02 (≥1 model per family + the Hurdle arm + a tail/quantile metric) is needed before any cohort decision.

### Production validation (#57–#61 · 2026-06-08)
The custom-driver result above was independently re-run through the **trusted production pipeline** — `python ./main.py -r calibration -t -e -sa`, one calibration run per arm, via `StepshifterManager` → views-evaluation (not the custom harness). Findings:
- **Production matched the custom driver to 3 decimals on EVERY metric** — MSLE (identity 4.2687, log1p 0.5205, asinh 0.5154 vs driver 4.269/0.521/0.515), tail-conditional MSLE (1.796/4.199/4.646), calibration (2.463/0.270/0.237), mean-on-true-zeros (16.04/0.55/0.49).
- The driver's **hand-asserted partition** (`train[121,444]/test[445,492]`) and **feature set** were **confirmed correct** against the pipeline's actual run.
- **F4 fires on production data too** (MSLE winner asinh 0.515; tail winner identity 1.796).
- ⇒ The EXP-01 conclusion **stands on trusted data**; the custom harness was faithful (not "counting unicorns"). Evidence: production artifacts `models/brown_cheese/data/generated/predictions_calibration_{233114,233628,234112}_*.parquet`; reconciliation `exp01_reconcile.py`; tracking issue #61 (closed). This closes the harness-fidelity threat above.

### Baseline reference frame (#62 · 2026-06-08)
Two standard naive baselines run through the same pipeline (cm, `lr_ged_sb`, calibration, views-evaluation `MSLE_mean`, same partition):

| model | cm MSLE_mean | mean pred |
|---|---|---|
| asinh (stepshifter) | 0.515 | 5.8 |
| log1p (stepshifter) | 0.521 | 6.6 |
| **`average_cmbaseline` (60-mo rolling avg)** | **0.581** | 41.8 |
| `locf_cmbaseline` (persistence / LOCF) | 0.807 | 105.6 |
| `zero_cmbaseline` (all-zeros) | 2.147 | 0 |
| identity (stepshifter, raw) | 4.269 | 60.7 |
| fatalities002 gold standard | ~0.835 *(comparability unverified)* | — |

**Reading:** the compression arms (~0.52) only *marginally* beat a dumb **60-month average (0.58)** on MSLE — the "fix" is barely better than naively averaging the past, while being *worse* on the escalation tail. Further evidence (with D-22 / F4) that **MSLE is a weak selection metric** and the compression arms are not a compelling fix on their own. *Caveat:* the gold-standard 0.835 is not yet confirmed comparable (likely a different partition/aggregation); a same-pipeline check would be needed before claiming a 60-mo average beats it.

> Note: the synthetic reproduction in `reports/investigation_metrics_08062026/FINDINGS.md` (raw 4.79 vs log1p 1.66 vs zeros 2.16) is **diagnostic evidence**, not a pre-registered experiment of this dossier. It motivates EXP-01; it does not substitute for it (synthetic data, not real models vs the gold standard).

> Note: the synthetic reproduction in `reports/investigation_metrics_08062026/FINDINGS.md` (raw 4.79 vs log1p 1.66 vs zeros 2.16) is **diagnostic evidence**, not a pre-registered experiment of this dossier. It motivates EXP-01; it does not substitute for it (synthetic data, not real models vs the gold standard).
