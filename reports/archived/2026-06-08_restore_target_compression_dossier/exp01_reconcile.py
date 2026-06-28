"""
EXP-01 reconciliation (#60) — compute the multi-component readout on the **production**
prediction outputs (from `main.py -r calibration -t -e -sa`) and compare to the
custom-driver EXP-01 numbers. The tail-conditional MSLE (the metric F4 rests on) is
NOT produced by views-evaluation, so we compute it here on the production predictions
via the unit-tested `exp01_readout` — i.e. production predictions + tested measurement.

Run in the dossier dir: `conda run -n views_pipeline python exp01_reconcile.py`
"""
import glob

import pandas as pd

from exp01_readout import readout

BC = "/home/simon/Documents/scripts/views_platform/views-models/models/brown_cheese"
GEN = f"{BC}/data/generated"
RAW = f"{BC}/data/raw/calibration_viewser_df.parquet"
TARGET = "lr_ged_sb"
TAU = 100.0

# Production prediction-artifact timestamps (one calibration run per arm).
ARMS = {
    "identity": "20260608_233114",
    "log1p": "20260608_233628",
    "asinh": "20260608_234112",
}
# Custom-driver numbers (07_experiment_log.md EXP-01) for side-by-side.
DRIVER = {
    "identity": dict(msle=4.269, tail=1.796, calib=2.463, zeros=16.04),
    "log1p": dict(msle=0.521, tail=4.199, calib=0.270, zeros=0.55),
    "asinh": dict(msle=0.515, tail=4.646, calib=0.237, zeros=0.49),
}
# views-evaluation headline (step/ts MSLE_mean) from each production run.
PROD_EVAL_MSLE = {"identity": 4.2687, "log1p": 0.5205, "asinh": 0.5154}

actual = pd.read_parquet(RAW)[TARGET].astype(float)


def pool_production_predictions(ts):
    files = sorted(glob.glob(f"{GEN}/predictions_calibration_{ts}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no production predictions for ts {ts}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df[f"pred_{TARGET}"].clip(lower=0), len(files)


def main():
    print(f"EXP-01 reconciliation — production predictions vs custom driver (tau={TAU})\n")
    hdr = f"{'arm':9s} {'source':7s} {'MSLE':>8s} {'tail_MSLE':>9s} {'calib':>7s} {'mean@zero':>9s}"
    print(hdr)
    print("-" * len(hdr))
    prod = {}
    for arm, ts in ARMS.items():
        pred, nseq = pool_production_predictions(ts)
        a = actual.reindex(pred.index)
        m = a.notna()
        r = readout(a[m].to_numpy(), pred[m].to_numpy(), tau=TAU)
        prod[arm] = r
        d = DRIVER[arm]
        print(f"{arm:9s} {'PROD':7s} {r['msle']:8.3f} {r['tail_msle']:9.3f} "
              f"{r['calibration_ratio']:7.3f} {r['mean_pred_on_true_zeros']:9.3f}   "
              f"(n={r['n']}, {nseq} seq, views-eval MSLE_mean={PROD_EVAL_MSLE[arm]})")
        print(f"{'':9s} {'driver':7s} {d['msle']:8.3f} {d['tail']:9.3f} "
              f"{d['calib']:7.3f} {d['zeros']:9.3f}")

    msle_winner = min(prod, key=lambda a: prod[a]["msle"])
    tail_winner = min(prod, key=lambda a: prod[a]["tail_msle"])
    fires = msle_winner != tail_winner
    print(f"\nF4 verdict on PRODUCTION data: MSLE winner = {msle_winner} "
          f"({prod[msle_winner]['msle']:.3f}); tail winner = {tail_winner} "
          f"({prod[tail_winner]['tail_msle']:.3f}) -> "
          f"{'F4 FIRES (confirmed on production)' if fires else 'does NOT fire'}")
    return prod


if __name__ == "__main__":
    main()
