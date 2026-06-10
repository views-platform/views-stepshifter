# 11 — Post-mortem: the stepshifter target-compression incident

**Date:** 2026-06-10 · **Severity:** high (silent model-output corruption in a production forecasting ensemble) · **Status:** resolved (views-stepshifter 1.3.0 released) · **Author:** Simon (+ Claude Code)

> **Convention:** this post-mortem follows the dossier's discipline — every claim is backed by a named artifact (commit, run, report cell, register entry), and **negatives are first-class**: the process failures below are recorded with the same prominence as the wins, and the falsified result (F4) is not softened. Blameless: the focus is the system that let a silent error live for 144 days, not the people.

---

## 1. Summary

For ~144 days the **plain** stepshifter cm models trained on the **raw, ~90 %-zero conflict target with no compression anywhere**, silently over-predicting the zero cells (cm MSLE 4–5, *worse than predicting all-zeros*). This dragged the `big_chungus` forecasting ensemble to **MSLE 2.519** — worse than the all-zeros baseline (2.147). No error was ever raised. We diagnosed the root cause, built and shipped a declared `target_transform` mechanism (views-stepshifter **1.3.0**, on PyPI), fixed two secondary silent-inflation bugs, and rebuilt the ensemble to **0.588**. **The fix restores sane values, not forecast skill** (see §8).

## 2. Impact

- The 2026-06-04 production ensemble report (`big_chungus`, run `2689xjtl`) was **worse than all-zeros** — operationally useless and, worse, *not obviously broken* to a casual reader of the headline number.
- Scope was **narrow**: only the plain constituents (XGB/XGBRF/LGBM). Hurdle (0.79–1.08) and deep-learning (0.43–0.57) constituents were healthy throughout — but the equal-weight mean dragged the whole ensemble below the zero baseline.
- Duration: **2026-04-11 → 2026-06-09** (~144 days) with no compression and no alarm.

## 3. Timeline

| Date | Event | Artifact |
|---|---|---|
| 2025-11-20 | Internal `log1p`/`expm1` added to stepshifter; queryset `.ln()` disabled simultaneously (to avoid double-compression) | `5fcfe43`, `831d3ebc` |
| 2026-04-11 | `5fcfe43` reverted — **but the queryset `.ln()` was never re-enabled** → compression gap opens | `08ee2eb` |
| 2026-06-04 | `big_chungus` ensemble runs at **MSLE 2.519** (> all-zeros 2.147) | report `20260604_230304` |
| 2026-06-08 | Diagnosis: dossier created; EXP-01 pre-registered + executed on real `brown_cheese` | dossier `05`/`07` |
| 2026-06-08 | `identity` arm reproduces production **4.269 ≈ 4.27** → root cause **confirmed**; F4 **fires** (tail) | `07_experiment_log.md` |
| 2026-06-09 | Forensic reconstruction; Shurf `log_target` bug confirmed (D-27); Hurdle verdict ROBUST | `08`, #68 |
| 2026-06-09→10 | `chunky_bunny` rebuilt; NHiTS divergence found + fixed (r2darts2); ensemble **0.588** | `09` |
| 2026-06-10 | Released **views-stepshifter 1.3.0** → PyPI (PRs #74, #76) | release `v1.3.0` |

## 4. Root cause

A **revert that restored only half of a coupled change.** `5fcfe43` had paired an internal log-transform with disabling the queryset `.ln()`. The revert (`08ee2eb`) undid the internal transform but **left `.ln()` disabled** — so the compensating compression existed in *neither* place. A squared-log-error regressor on a 90 %-zero target then minimised loss by over-predicting onto the zeros (mean prediction ~16–60 where truth is 0). Hurdle was immune by architecture (binary gate zeroes the zero cells first); DL was immune (internal asinh).

## 5. Detection

**Slow and lucky, not systemic.** The only signal was a human noticing the *ensemble* MSLE was worse than all-zeros in a report read 8 weeks after the gap opened. There was **no automated alarm** — not at training (raw target is "valid"), not at prediction (raw predictions are "valid"), not at report generation. The per-constituent leaderboard that localises the breakage was *in the report the whole time*, but the first investigation didn't read it (see §7).

## 6. Resolution

1. Declared **`target_transform`** registry `{identity, log1p, asinh}` (#52), gate-enforced; plain → `log1p`, Hurdle/Shurf pinned `identity`.
2. **Shurf `log_target=True`** fix (D-27): was `expm1`-ing a raw prediction → float64 overflow → `inf`; gate now rejects it.
3. **NHiTS divergence** (10²¹ on the +12-month partition) fixed by the r2darts2 update (RevIN + SpotlightLossLogcosh), not a config tweak.
4. Rebuilt `chunky_bunny` (23 fresh constituents) → **0.588**; released as **1.3.0** to PyPI.

## 7. What went wrong — contributing factors (negatives first-class)

1. **Silent for 144 days.** The whole failure mode produced *no error at any stage*. A bug that corrupts model output without a signal is the worst kind. → **fail-loud gap.**
2. **Output is not gated, only input.** The fail-loud philosophy caught a bad *input* this session (viewser drift-detection rejected `yellow_submarine`'s absent IMF data), but a *10²¹ output* from a diverged NHiTS sailed straight through into a clean report (**D-30**). Input is guarded; output is not.
3. **Diagnosis by inference, not measurement.** The first investigation (`FINDINGS.md`) over-claimed "all 19 constituents broken" by reasoning from "trains raw" — when the report's own per-constituent table showed **Hurdle and DL were fine**. The data to correct it was present and unread.
4. **A near-botched release.** `pyproject` declared `1.2.0`, but `1.2.0` was *already a published tag* pointing at code **without** the feature (**D-14**). Only the release ritual's version check caught it before we shipped a corrupt 1.2.0.
5. **Tests that can't catch this class of bug.** The model tests (`test_fit`/`test_predict`) mock `ProcessPoolExecutor`/the model classes and assert *call-wiring*, not numerical behavior — they would have stayed green through the entire incident (**D-13**). They were also env-brittle enough to block CI (now xfail'd; real tests tracked in **#75**).
6. **Operational fragility.** The diverging NHiTS run corrupted the GPU's CUDA context (`torch.cuda.is_available()` → `False`), recoverable only by reloading the `nvidia_uvm` module. A divergent model shouldn't be able to wedge the box.
7. **Dev path ≠ production path (D-28).** The ensemble's `run.sh` route installs the *published* views-stepshifter (which, until 1.3.0, lacked the fix) into per-model envs — so the config fix could be **silently ignored at a real ensemble run**. Validation ran in an editable install. Closed by the 1.3.0 release; the gap itself is a structural risk.

## 8. The honest caveat (the falsified result, not softened)

The pre-registered **F4 falsifier fired.** `log1p` restores headline MSLE (4.27 → 0.52) but the compressed models are **~2.5× worse on the escalation tail** (tail MSLE 1.80 → 4.2–4.6) and calibration flips from 2.46× over to ~0.25× under — i.e. they now **under-forecast the high-fatality cells, which are the operational product.** Selecting on MSLE alone would crown a *worse* model. **Compression is necessary but not sufficient; `0.588` means bug-free, not good.** The optimization (tail/calibration) is a future phase, not this one.

## 9. What went well

- **Pre-registration + falsifiers worked exactly as intended.** F4 was anticipated in the skepticism ledger and fired on schedule, stopping a naive "log1p is the fix" conclusion. The dossier's honesty held under pressure.
- **The release rituals earned their keep** — the version-collision (D-14) would have shipped a corrupt 1.2.0 without them.
- **Diagnosis was validated on the trusted production pipeline** (not just a custom harness): identity reproduced 4.27 to 3 decimals.
- **The fix is principled** (declared, gate-enforced, reversible) rather than a patch.

## 10. Lessons

1. **A revert must restore the full coupled state.** The `.ln()` half was the silent landmine. Coupled changes need a coupled revert checklist.
2. **Guard outputs, not just inputs.** Add a fail-loud magnitude/range check on predictions before they're persisted/aggregated (D-30).
3. **Measure per-unit; don't infer brokenness from a global property.** Read the leaderboard.
4. **Tests must exercise real numerical behavior** — mock-wiring tests give false confidence (#75).
5. **Release hygiene is a safety control, not bureaucracy.**

## 11. Action items

| Item | Owner | Tracking |
|---|---|---|
| EXP-02 — tail/quantile-aware selection metric + count-likelihood arm (the *real* optimization) | research | dossier `04`; register D-22/D-23 |
| Fail-loud guard on DL/constituent prediction magnitude | platform | **D-30** |
| Replace xfail'd mock-brittle tests with real behavioral tests (+ Shurf/darts coverage) | stepshifter | **#75** |
| Cut a `poetry.lock` (CI installs unpinned) | stepshifter | **D-14** (lockfile half) |
| ~~Release the fix so production stops training raw~~ | — | ✅ done (1.3.0; closes D-28) |

## 12. Evidence index

Forensic detail `08_forensic_what_happened.md`; EXP-01 ledger + F4 postmortem `07_experiment_log.md`; ensemble rebuild `09_chunky_bunny_rerun_log.md`; team summary `10_team_report.md`; risk register `technical_risk_register.md` (D-13/14/17/22/27/28/30); ADR-003; commits `5fcfe43`/`08ee2eb`; release `v1.3.0` (PRs #74, #76).
