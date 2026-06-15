"""EXP-01 — HurdleModel tail/distributional readout (C-1). Pre-registered in 05_analysis_plan.md.

Retrain-free. For each model: pool the 13 rolling calibration prediction sequences, join to that
model's observed actuals (lr_ged_sb) on (month_id, country_id) over months 457-492, and compute:
  - missed-escalation rate = mean(pred < 0.5 | obs > tau) for tau in {1,10,25,100} (+ pred==0 exactly)
  - tail error: MSLE & MAE on obs > tau
  - calibration by observed magnitude bucket
Controls: car_radio (plain log1p) + an all-zeros baseline; matched-feature XGB-vs-LGBM pairs.

Run:
  cd .../views-models && \
  /home/simon/anaconda3/envs/views_pipeline/bin/python \
      ../views-stepshifter/reports/2026-06-14_hurdle_point_estimator_dossier/exp01_tail_readout.py
"""
import glob
import numpy as np
import pandas as pd

MODELS_DIR = "models"  # run from the views-models repo root
WIN = (457, 492)
TAUS = [1, 10, 25, 100]
BUCKETS = [(-0.1, 0.0), (0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, np.inf)]

# (name, kind) — kind is descriptive only
HURDLES = ["fast_car", "fluorescent_adolescent", "green_squirrel",
           "high_hopes", "little_lies", "twin_flame"]
REFERENCE = "car_radio"  # plain log1p


def load_pairs(name):
    """Pooled (pred, obs) over the 13 sequences for one model; obs from its own raw df."""
    pred_files = sorted(glob.glob(f"{MODELS_DIR}/{name}/data/generated/predictions_calibration_*_*.parquet"))
    pred_files = [f for f in pred_files if "/predictions_calibration_" in f]
    if not pred_files:
        return None
    preds = pd.concat([pd.read_parquet(f) for f in pred_files])
    pcol = [c for c in preds.columns if c.startswith("pred_")][0]
    preds = preds.rename(columns={pcol: "pred"})[["pred"]]
    raw = pd.read_parquet(f"{MODELS_DIR}/{name}/data/raw/calibration_viewser_df.parquet")
    obs = raw[["lr_ged_sb"]].rename(columns={"lr_ged_sb": "obs"})
    df = preds.join(obs, how="inner")
    mi = df.index.get_level_values("month_id")
    df = df[(mi >= WIN[0]) & (mi <= WIN[1])].copy()
    df["pred"] = df["pred"].clip(lower=0.0)  # match the pipeline neg->0 standardize
    return df


def msle(pred, obs):
    return float(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2)) if len(pred) else float("nan")


def report(name, df):
    n = len(df)
    pred, obs = df["pred"].to_numpy(), df["obs"].to_numpy()
    mcr = pred.mean() / obs.mean() if obs.mean() else float("nan")
    print(f"\n### {name}  (n={n}, mean_obs={obs.mean():.2f}, mean_pred={pred.mean():.2f}, "
          f"MCR={mcr:.3f}, overall_MSLE={msle(pred, obs):.3f})")
    # missed-escalation + tail error per tau
    print("  tau | n_tail | missed%(pred<0.5) | pred==0% | tailMSLE | tailMAE")
    for tau in TAUS:
        m = obs > tau
        nt = int(m.sum())
        if nt == 0:
            print(f"  {tau:>4}| {nt:>6} | (no cells)")
            continue
        miss = float((pred[m] < 0.5).mean())
        zero = float((pred[m] == 0).mean())
        tmsle = msle(pred[m], obs[m])
        tmae = float(np.mean(np.abs(pred[m] - obs[m])))
        print(f"  {tau:>4}| {nt:>6} | {miss:>15.3f} | {zero:>7.3f} | {tmsle:>8.3f} | {tmae:>8.1f}")
    # calibration by observed bucket
    print("  bucket(obs]      | n     | mean_obs | mean_pred")
    for lo, hi in BUCKETS:
        m = (obs > lo) & (obs <= hi)
        if m.sum() == 0:
            continue
        print(f"  ({lo:>5},{hi:>6}] | {int(m.sum()):>5} | {obs[m].mean():>8.2f} | {pred[m].mean():>8.2f}")


def main():
    print("=" * 78)
    print("EXP-01 tail readout — window months", WIN, "| pred clipped >=0 | pred~0 := pred<0.5")
    print("=" * 78)
    # Hurdles
    for name in HURDLES:
        df = load_pairs(name)
        if df is None:
            print(f"\n### {name}: NO PREDICTION FILES")
            continue
        report(name, df)
    # Reference: plain log1p
    dref = load_pairs(REFERENCE)
    if dref is not None:
        report(f"{REFERENCE} [REFERENCE: plain log1p]", dref)
        # All-zeros baseline on the same actuals
        dz = dref.copy()
        dz["pred"] = 0.0
        report("all-zeros [REFERENCE baseline]", dz)


if __name__ == "__main__":
    main()
