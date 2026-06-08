"""
EXP-01 driver — compression/likelihood pre-screen on real brown_cheese data.
Pre-registration: 05_preanalysis_exp01.md. Run in the `views_pipeline` conda env.

Arms (the ONE variable): identity (control) / log1p / asinh. (Tweedie is out of
scope — deferred per the scope decision, register D-23/D-26.)
Everything else held constant (data, features, partition, XGBRF family, the stepshift).
Readout (multi-component, per the plan): cm MSLE vs all-zeros (2.15) + production
identity parity (~4.27); mean prediction on true-zero cells; tail-conditional MSLE
(observed > tau); calibration-in-the-large (mean pred vs mean actual).

NOTE: this is a SMOKE / boundary-hack driver (it applies the transform in the
script). It is SUPERSEDED by the real declared-mechanism run in issue #54, which
applies the transform inside the model (forward in _process_data, inverse at the
predict() boundary). The DATA path below is machine-specific.

Usage:
  python exp01_brown_cheese.py            # full run (steps 1-36, n_estimators 300; minutes)
  python exp01_brown_cheese.py --smoke    # plumbing check (steps 1-2, n_estimators 20; ~1 min)
"""
import sys
import numpy as np
import pandas as pd
from views_stepshifter.models.stepshifter import StepshifterModel

SMOKE = "--smoke" in sys.argv
DATA = "/home/simon/Documents/scripts/views_platform/views-models/models/brown_cheese/data/raw/calibration_viewser_df.parquet"
TARGET = "lr_ged_sb"
PARTS = {"train": [121, 444], "test": [445, 492]}
STEPS = [1, 2] if SMOKE else list(range(1, 37))
PARAMS = {"n_estimators": 20 if SMOKE else 300, "n_jobs": 12}
TAU = 100.0  # escalation threshold for the tail-conditional readout

df0 = pd.read_parquet(DATA)
actual = df0[TARGET].astype(float).copy()


def run_arm(fwd, inv, model_reg, extra_params=None):
    df = df0.copy()
    df[TARGET] = fwd(df[TARGET].to_numpy(dtype=float))
    cfg = {
        "steps": STEPS,
        "targets": [TARGET],
        "parameters": {**PARAMS, **(extra_params or {})},
        "model_reg": model_reg,
        "sweep": False,
    }
    m = StepshifterModel(cfg, PARTS)
    m.fit(df)
    preds = pd.concat(m.predict("calibration"))  # list[DataFrame] -> pooled
    p = pd.Series(inv(preds[f"pred_{TARGET}"].to_numpy(dtype=float)), index=preds.index)
    return p.clip(lower=0)


def msle(y, yhat):
    return float(np.mean((np.log1p(yhat) - np.log1p(y)) ** 2))


def score(p):
    a = actual.reindex(p.index)
    mask = a.notna()
    a, p = a[mask], p[mask]
    zero = a == 0
    tail = a > TAU
    return {
        "n": int(len(a)),
        "MSLE": round(msle(a, p), 4),
        "MSLE_zeros_base": round(msle(a, np.zeros(len(a))), 4),
        "mean_on_true_zero": round(float(p[zero].mean()), 3),
        "mean_pred": round(float(p.mean()), 2),
        "mean_actual": round(float(a.mean()), 2),
        "tail_MSLE(obs>%g)" % TAU: round(float(np.mean((np.log1p(p[tail]) - np.log1p(a[tail])) ** 2)), 4) if tail.any() else None,
    }


ARMS = {
    "identity": (lambda x: x, lambda x: x, "XGBRFRegressor", None),
    "log1p":    (np.log1p, np.expm1, "XGBRFRegressor", None),
    "asinh":    (np.arcsinh, np.sinh, "XGBRFRegressor", None),
}

print(f"EXP-01 brown_cheese | SMOKE={SMOKE} | steps={len(STEPS)} | n_estimators={PARAMS['n_estimators']}")
print("all-zeros baseline MSLE target = 2.15 ; production identity parity = ~4.27\n")
results = {}
for name, (fwd, inv, mr, ep) in ARMS.items():
    try:
        s = score(run_arm(fwd, inv, mr, ep))
        results[name] = s
        print(f"{name:9s} {s}")
    except Exception as e:
        print(f"{name:9s} ERROR: {type(e).__name__}: {e}")

print("\n--- verdict checks (per 05_preanalysis_exp01.md) ---")
if "identity" in results:
    idp = results["identity"]["MSLE"]
    print(f"F3 parity: identity MSLE {idp} vs production ~4.27 -> {'OK' if abs(idp-4.27)<0.5 else 'MISMATCH (harness suspect)' if not SMOKE else 'n/a (smoke)'}")
for arm in ("log1p", "asinh"):
    if arm in results:
        m = results[arm]["MSLE"]
        print(f"{arm}: MSLE {m} vs zeros 2.15 -> {'BEATS baseline' if m < 2.15 else 'does NOT beat (F1 candidate)'}")
