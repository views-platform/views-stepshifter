# Negative Result — the HurdleModel hard gate DOES drop escalations (H FALSIFIED)

**Status:** Hypothesis **FALSIFIED**. The identity HurdleModel's hard `{0,1}` gate **materially
drops escalations** relative to the plain reference; its aggregate magnitude honesty (MCR≈1) was a
coincidence that **does not** extend to the operational tail. **So-what:** Option 0 (keep + document)
is rejected; recommend **Option 1** (probability gate) via a *deferred* ADR (salvage phase — do not
implement now).

**Pre-registration:** `05_analysis_plan.md` (EXP-01). **Driver:** `exp01_tail_readout.py`
(retrain-free; run from the views-models root). **Window:** months 457–492, n=74,490 pooled
prediction instances/model, mean obs ≈ 26.7.

## 1. What we tested (and pre-registered)
**H:** the hard gate does **not** materially drop escalations — tail-acceptable, Option 0 holds.
**Falsifier F1 (pre-committed):** if the Hurdle family's tail missed-escalation **and** tail error
are materially worse than the references (car_radio log1p + all-zeros), Option 0 is rejected → Option 1.

## 2. What happened (the data)

**Missed-escalation rate = `mean(pred < 0.5 | obs > τ)`** (fraction of real escalation cells the model predicts ≈ 0):

| model (type) | τ=1 | τ=10 | τ=25 | τ=100 | tail MSLE (obs>10) |
|---|---|---|---|---|---|
| fast_car (XGB) | 0.230 | 0.157 | 0.123 | 0.085 | 3.76 |
| fluorescent_adolescent (XGB) | 0.224 | 0.162 | 0.125 | 0.085 | 3.49 |
| green_squirrel (XGBRF) | 0.098 | 0.070 | 0.049 | 0.034 | 2.19 |
| high_hopes (LGBM) | 0.400 | 0.316 | 0.278 | 0.198 | 6.60 |
| little_lies (LGBM) | 0.345 | 0.270 | 0.241 | 0.182 | 5.96 |
| twin_flame (LGBM) | 0.277 | 0.197 | 0.155 | 0.111 | 4.94 |
| **car_radio (plain log1p, REF)** | **0.057** | **0.022** | **0.013** | **0.017** | **2.44** |
| all-zeros (REF) | 1.000 | 1.000 | 1.000 | 1.000 | 19.50 |

Every Hurdle misses **3–14×** more escalations than the plain reference. car_radio **never** predicts
exactly 0 (`pred==0%` = 0.000 at every τ); the Hurdle hard gate zeros 7–32% of `obs>10` cells.

**Calibration (mean pred vs mean obs by observed bucket) — the F3 evidence:**

| obs bucket | mean obs | car_radio pred | twin_flame pred | little_lies pred |
|---|---|---|---|---|
| 0 | 0.00 | 0.35 | 6.12 | 1.75 |
| (0,1] | 1.00 | 3.83 | 39.3 | 18.9 |
| (1,10] | 4.66 | 6.68 | 62.1 | 27.3 |
| (10,100] | 39.45 | 34.5 | 167.8 | 93.9 |
| (100,∞] | 772.7 | 213.3 | 399.9 | 471.0 |

The Hurdles grossly **over-predict small/zero cells** and **under-predict the extreme tail** — the
two errors cancel into a benign aggregate MCR while per-cell calibration is wrecked.

## 3. Decisive evidence
The cleanest, least-confounded comparison: **car_radio vs every Hurdle on missed-escalation**. car_radio
is the *timid* model by aggregate MCR (0.36) yet misses the **fewest** escalations (2.2% at τ=10),
because a continuous regressor always places some mass on a cell. The hard gate, by emitting a `{0,1}`
**label**, sets the entire prediction to 0 wherever it rounds the event probability down — and it does
so on 7–32% of cells with real, often large, conflict. This is the EWS-dangerous failure mode, and it
is the *direct* mechanical consequence of D-33 (label gate, probability discarded).

## 4. Biases weighed (intellectual-honesty audit)
- **Confound (F2) NOT fully ruled out but does not rescue:** LGBM hurdles miss more than XGB (tuning,
  D-23 — real), but the best-tuned XGB/XGBRF hurdles still miss 7–16% vs car_radio's 2.2%. The gate
  effect is present across *all* learners → it is the estimator, not just tuning.
- **Pooling/alignment caveat (skepticism #5):** this retrain-free readout pools all 13 rolling
  sequences and inner-joins to actuals; absolute MCRs run ~10% higher than the official views-evaluation
  numbers (e.g. car_radio 0.362 here vs 0.327 in the note). The **relative** Hurdle-vs-reference tail
  comparison — the basis of the decision — is robust to this (same n, same join, same window for all).
- **`pred≈0 := pred<0.5`:** results are identical at `pred==0` exactly (the `pred==0%` column matches
  the `<0.5` column for the Hurdles), so the threshold choice is not load-bearing.
- **Confirmation bias check:** the result *contradicts* the prior (Story A) leaning ("aggregate-benign,
  probably keep"). It was pre-registered before any tail number was seen.

## 5. What is / isn't established
- **Established:** the hard gate drops a large fraction of escalations vs a continuous reference (F1
  fired); aggregate MCR masks per-cell miscalibration (F3 fired); the effect is not merely a tuning
  artefact (F2 partial). These are on the retrain-free calibration readout, n≈74k/model.
- **Not established:** the exact official-eval magnitudes (this is an approximation); the *fix's*
  efficacy (Option 1 is recommended, not tested); behaviour on the forecasting (vs calibration) partition.

## 6. Disposition
- **Abandon Option 0** (keep + document the gate as benign). The "document as a deliberate property"
  framing the method panel warned against is now empirically wrong.
- **Recommend Option 1** (binary stage emits `P(Y>0)` + clip, so the product ≈ `E[Y]` and never hard-zeros
  an escalation cell) — via a **deferred** ADR/issue linking #66. **Do not implement** (salvage phase).
- **Re-escalate D-33** (the hard-gate label) — its Tier-3 de-escalation rested on "aggregate magnitude
  benign," which the tail readout refutes. Candidate Tier 2 (operationally-harmful missed escalations).
- **Track the tuning gap** (untuned LGBM hurdles, D-23) separately.
- This redirects Story C (estimator decision) from "keep + document" to "recommend Option 1, deferred."
