# 09 — chunky_bunny re-run log + findings (2026-06-09)

Running log for the dev-mode re-run of the **chunky_bunny** ensemble (exact clone of the original 2026-06-04 `big_chungus`: 23 constituents = 13 plain + 6 Hurdle + 4 DL). Material for the meta report. Append-only.

## The run recipe (validated)
Production deploy needs the views-stepshifter release (1.2.0); but to produce correct numbers **now** we run in the editable dev env:
1. **Online** (NOT `WANDB_MODE=offline`) + `-re`. The evaluation report reads metric values back from the **wandb run summary** (a public-API `Run` object), so an offline run yields "not calculated" for every metric even though they were computed. Online → syncs → report fills.
2. **Everything fresh** (clean slate per model). The report's partition-consistency check (`evaluation.py` "Partition metadata mismatch between models") compares every model's *latest* run; the +12-month partition bump means all old runs are stale, so all 23 + the 3 baselines must be re-run before any report.
3. **Ensemble report runs LAST** — after all constituents + baselines have fresh, same-partition runs.

## FINDING — fail-loud caught structurally-absent data (yellow_submarine)
**What:** `yellow_submarine` failed its viewser fetch (before training): `RuntimeError: Input is not a df or a df index` → `DataFetchException`. Unique to yellow_submarine among the 23.

**Cause:** its distinctive columns are `lr_imfweo_ngdp_rpch` (+ `_tlag12/24/36`) — IMF WEO **annual** GDP growth (`from_loa='country_year'`). The **+12-month partition bump** (target window now months 457–504) outruns the IMF annual data's coverage, so viewser returns a **non-DataFrame** for those columns.

**Was it fail-loud, or did it just break? → Caught by a deliberate fail-loud guard.** The raise is from viewser's **drift-detection `InputGate`** (the "input warning machinery"), via a type-check in `views_tensor_utilities/mappings.py::get_index` (`else: raise RuntimeError('Input is not a df or a df index')`). It is the **platform's** fail-loud gate, not the views-stepshifter `target_transform` work — but it is a deliberate guard, and it did its job: it refused to hand structurally-absent data to training.

**Why we "used to run with bad data":** the queryset's `.transform.missing.replace_na(0)` **silently masks missing values** — partial gaps get filled with 0 and the model trains regardless (the silent-garbage-run pattern). That masking only works when a DataFrame comes back. Here the data was **entirely absent** for the window → not a df at all → `replace_na` had nothing to operate on → the drift-detection type-guard tripped. **Boundary: silent masking handles partial gaps; total absence trips the loud gate.**

**Verdict (for the meta report):**
- *Good:* the platform caught structurally-broken input loudly instead of training on garbage — aligned with fail-loud.
- *Latent fragility worth flagging:* `replace_na(0)` **silently hides partial data-quality gaps**; only total absence is caught. A model can still train on heavily zero-filled (degraded) features without any signal. The episode is a concrete example of masking-vs-fail-loud tension exposed by an unrelated change (the partition bump).

**Disposition:** commented out the 4 `imfweo` `.with_column(...)` blocks in `models/yellow_submarine/configs/config_queryset.py` (salvage workaround — drops those features); fetch then succeeded. Tracked in **views-models#125**. Proper fix (confirm/refresh IMF WEO coverage in viewser, then restore the columns) deferred to the team.

## Per-constituent calibration MSLE (time-series-wise, cm, lr_ged_sb; new +12-month partition)
| constituent | family | MSLE |
|---|---|---|
| bittersweet_symphony | plain | 0.520 |
| brown_cheese | plain | 0.587 |
| car_radio | plain | 0.467 |
| counting_stars | plain | 0.515 |
| demon_days | plain | 0.491 |
| good_riddance | plain | 0.530 |
| heavy_rotation | plain | 0.514 |
| national_anthem | plain | 0.574 |
| ominous_ox | plain | 0.520 |
| plastic_beach | plain | 0.587 *(≡ brown_cheese — D-21: identical configs)* |
| popular_monster | plain | 0.496 |
| teen_spirit | plain | 0.550 |
| yellow_submarine | plain | *(imfweo removed; see #125)* |
| fast_car | Hurdle | 0.885 |
| (remaining Hurdle + 4 DL) | — | *(in progress)* |

All plain constituents are sane (~0.47–0.59), beating the cm baselines (zero 2.15, average ~0.58–0.72, locf ~0.65–0.81) — the salvage target. The `log1p` under-prediction trait (MCR_point ~0.28) persists, deferred to optimization.

## FINDING — silent prediction-inflation NOT caught (revolving_door / NHiTS)
**What:** `revolving_door` (NHiTS, DL) produced **`y_hat_bar = 1.0e+21`** (mean prediction ~10^21 vs actual ~25), cm time-series-wise **MSLE 67.03** — catastrophically inflated. The model trained, produced 13 prediction files, and **wrote a clean report**; the only failure signal was a human noticing the MSLE/y_hat_bar.

**Mechanism (suspected):** NHiTS diverged on the new +12-month window; the DL models' internal **AsinhTransform** inverse (`sinh`) on a diverged latent value explodes (sinh grows exponentially) → ~10^21 predictions. Directly analogous to the D-27 ShurfModel `expm1`-of-raw inflation, but in the deep-learning (r2darts2) stack. `revolving_door` was healthy (0.43–0.57) in the 2026-06-04 big_chungus, so the +12-month partition is the likely trigger. Other DL (TSMixer 0.92, NBEATS 0.545, TiDE 0.41) were sane.

**Fail-loud contrast (for the meta report):**
- `yellow_submarine`: bad **input** (absent IMF data) → **caught loud** by viewser drift-detection InputGate.
- `revolving_door`: bad **output** (10^21 predictions) → **NOT caught** — silently produced a report with insane numbers. There is **no guard on prediction magnitude / inverse-transform overflow** in the DL path. This is a concrete fail-loud GAP: input is gated, output is not.

**Impact:** a constituent at 10^21 would obliterate the chunky_bunny **mean** aggregation (ensemble mean ≈ 4e19). It must NOT be included as-is. Un-ticked on the run tracker (views-models#117).

**Next:** re-run once to distinguish stochastic divergence (DL training variance → may converge) from systematic (new partition genuinely breaks NHiTS → team/data fix, or exclude from the ensemble).

**Update — confirmed SYSTEMATIC (not stochastic):** re-run #2 gave the opposite degenerate failure — `y_hat_bar = 0.00212` (near-zero collapse), MSLE 2.19 (≈ the all-zeros baseline 2.15). So NHiTS swings between **overflow (10^21)** and **collapse (~0)** across runs — it does not converge on the +12-month window. The other 3 DL (TSMixer/NBEATS/TiDE) are sane and stable. **revolving_door (NHiTS) is broken on the new partition** (was healthy 0.43–0.57 in the 2026-06-04 big_chungus → the partition bump is the trigger). Not a quick salvage fix — it's DL training instability (LR/gradient-clip/seed territory, owned by the r2darts2/DL team). Decision pending: exclude from chunky_bunny (22/23) vs. team investigation.

**Update 2 — grad_clip fix FAILED.** Tightening `gradient_clip_val` 20→5 (to match NBEATS) did NOT stabilise NHiTS — it diverged again to `y_hat_bar = 1.2e23`, MSLE 1647. Four runs total, all broken (10^21, ~0 collapse, 10^23). Gradient clipping is not the lever; the instability is deeper (likely the model + AsinhTransform inverse on this window). Also surfaced a GPU side-effect: the diverging run triggered a CUDA "unspecified launch failure" that corrupted the CUDA context (`torch.cuda.is_available()=False`); recovered without reboot via `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`. **Conclusion: revolving_door/NHiTS is not salvageable by config tweak here — it needs the r2darts2/DL team (LR, architecture, or an output clamp on the inverse-transform). Decision: exclude from chunky_bunny (run 22/23) to deliver the ensemble, and track NHiTS separately.**

## OUTCOME — chunky_bunny ensemble complete (2026-06-10 02:50)
All 23 constituents ran sane on the new partition. **revolving_door/NHiTS was fixed not by config tweak but by the r2darts2 update + the r7/r8 DL tuning (RevIN + SpotlightLossLogcosh)** — y_hat_bar back to ~31 (was 10^21/10^23). DL final: TSMixer 0.814, NBEATS 0.497, NHiTS 0.555, TiDE 0.503. **chunky_bunny ensemble cm MSLE = 0.588 (time-series-wise)** — vs the original broken big_chungus 2.519 and all-zeros 2.147. Salvage objective met: a sane, bug-free ensemble (mediocre on the tail/log1p under-prediction, deferred to optimization). Side-finding: the diverging NHiTS run corrupted the CUDA context (recovered via nvidia_uvm module reload, no reboot).
