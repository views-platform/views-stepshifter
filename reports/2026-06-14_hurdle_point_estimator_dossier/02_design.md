# 02 — Design: the Hurdle point-estimator decision space

**Date:** 2026-06-14. **Status:** draft — framed, awaiting `expert-method-review` before pre-registration.
**Graduates to:** a proposed ADR carrying the estimator decision (Story C / #79).

> This is not a build spec — it is the **decision** the dossier must reach: given a confirmed-but-
> aggregate-benign mis-specification, what do we do? The salvage phase means the *near-term* answer is
> very likely "(0) keep + document + verify the tail", with (1)/(2)/(3) deferred to a post-salvage ADR.

## 1. The problem (established, Story A)

The Hurdle point forecast is `gate × magnitude` where:
- **gate** = the binary stage's **hard `{0,1}` class label** (D-33) — the available probability is
  discarded; no `clip`. So it is not `P(Y>0)`.
- **magnitude** = a regressor trained on **whole zero-heavy series** (D-37) — `E[Y|ever-positive]`,
  not `E[Y|Y>0]`.

So the product is **not** the textbook hurdle mean `E[Y]=P(Y>0)·E[Y|Y>0]` and elicits no clean
functional consistent with the MSLE it's scored on (D-38). **Yet** on real data it is
magnitude-honest in aggregate (MCR 0.75–1.29) — the two errors offset. The open question is the
**tail**: does the hard gate silently zero escalation cells? (Unmeasured → harness C-1.)

## 2. Options (the decision)

| # | Option | What changes | Cost | Salvage-phase status |
|---|--------|--------------|------|----------------------|
| 0 | **Keep + document + verify tail** | nothing in code; document the gate as a deliberate estimator; build the tail readout (C-1); pin with tests (C-3) | low | **in scope now** |
| 1 | Probability gate | binary stage emits `P(Y>0)` (clip [0,1]) instead of a label → product ≈ `E[Y]` | medium | deferred (flag c1) |
| 2 | Observation-level positive stage | condition the magnitude on `Y>0`, not series-ever-positive | medium–high (Darts continuous-timestamp constraint, see README §Hurdle) | deferred (flag c2) |
| 3 | Count likelihood | replace two-stage with Tweedie/NB single model | high | deferred (flag c3) |

## 3. Decision criteria (what a good answer must show)

1. **Tail accuracy** (the EWS product): tail-conditional error + missed-escalation rate must not be
   worse than the current identity Hurdle (and ideally better). *Primary.*
2. **Aggregate calibration** preserved (MCR stays ≈1; A.3 invariant).
3. **Honest attribution** — transform/estimator effect separated from the **untuned-LGBM** confound (C-4).
4. **Proper-scoring coherence** — if a point functional is claimed (mean vs median), the training loss
   must elicit it (Gneiting; D-38).
5. **Salvage compliance** — code changes (1/2/3) only via a promoted ADR lifting D-26; not unilaterally.

## 4. Current recommendation (provisional, pre-method-review)

**Option 0 now.** The estimator is not breaking aggregate magnitude; the genuine unknown is the tail.
Build C-1, verify the tail, document the gate as an explicit (if non-textbook) estimator, and pin it.
If C-1 shows the hard gate is silently dropping escalations, that is the evidence that would justify
opening a *deferred* ADR for option 1 (probability gate — the cheapest principled fix). Options 2/3 are
research-grade and squarely post-salvage. This recommendation is what `expert-method-review` should
attack before anything is pre-registered.

## 5. Open method questions (for expert-method-review)
- Is "magnitude-honest in aggregate via offsetting errors" acceptable, or a fragile coincidence that a
  data-regime shift breaks? (A.3 vs §2 option 1.)
- Right tail threshold(s) τ and tail metric (MSLE|obs>τ vs MAE vs a proper weighted score, Lerch2017)?
- Does the untuned-LGBM confound (C-4) materially distort the 6-model comparison?
