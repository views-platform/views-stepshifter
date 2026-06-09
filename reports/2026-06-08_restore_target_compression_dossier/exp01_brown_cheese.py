"""
EXP-01 driver — target-compression pre-screen on real brown_cheese data, run
THROUGH THE REAL declared mechanism (issue #54). The transform is declared via the
config ``target_transform`` key and applied *inside* ``StepshifterModel`` (forward in
``_process_data``, inverse at the ``predict()`` boundary). No boundary hacks: the
driver hands the model a RAW dataframe and receives RAW-space predictions.

Pre-registration: ``05_preanalysis_exp01.md``. Arms (the ONE variable): identity
(control) / log1p / asinh. Measurement is delegated to ``exp01_readout`` (unit-tested
in ``test_exp01_readout.py``) — the driver only orchestrates the fits.

Run in the ``views_pipeline`` conda env with the mechanism installed editable
(``feature/target_transform``). The DATA path is machine-specific.

Usage:
  python exp01_brown_cheese.py            # full (steps 1-36, n_estimators 300; minutes)
  python exp01_brown_cheese.py --smoke    # plumbing (steps 1-2, n_estimators 20; ~1 min)
"""
import math
import sys

import pandas as pd

from exp01_readout import readout
from views_stepshifter.models.stepshifter import StepshifterModel

SMOKE = "--smoke" in sys.argv
DATA = (
    "/home/simon/Documents/scripts/views_platform/views-models/"
    "models/brown_cheese/data/raw/calibration_viewser_df.parquet"
)
TARGET = "lr_ged_sb"
PARTS = {"train": [121, 444], "test": [445, 492]}
STEPS = [1, 2] if SMOKE else list(range(1, 37))
PARAMS = {"n_estimators": 20 if SMOKE else 300, "n_jobs": 12}
TAU = 100.0  # escalation threshold for the tail-conditional readout
ARMS = ["identity", "log1p", "asinh"]
ALLZEROS_BASELINE = 2.15  # known cm MSLE of the all-zeros forecast
PROD_IDENTITY = 4.27  # known production identity MSLE (F3 parity target)

df0 = pd.read_parquet(DATA)
actual = df0[TARGET].astype(float)


def run_arm(target_transform):
    """Fit + calibration-predict one arm through the real declared mechanism.

    The model applies the forward (in fit) and the inverse (at predict) itself,
    so we pass the RAW dataframe and get RAW-space predictions back.
    """
    cfg = {
        "steps": STEPS,
        "targets": [TARGET],
        "parameters": PARAMS,
        "model_reg": "XGBRFRegressor",
        "sweep": False,
        "target_transform": target_transform,
    }
    model = StepshifterModel(cfg, PARTS)
    model.fit(df0)
    preds = pd.concat(model.predict("calibration"))
    # Mirror the production manager clamp (negatives -> 0, register D-16).
    return preds[f"pred_{TARGET}"].clip(lower=0)


def score_arm(pred):
    """Align predictions to the observed target and delegate to the tested readout."""
    a = actual.reindex(pred.index)
    mask = a.notna()
    return readout(a[mask].to_numpy(), pred[mask].to_numpy(), tau=TAU)


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _verdicts(results):
    print("\n--- pre-registered falsifier checks (05_preanalysis_exp01.md) ---")
    if "identity" in results:
        idp = results["identity"]["msle"]
        if SMOKE:
            verdict = "n/a (smoke — reduced steps/estimators)"
        else:
            verdict = "OK" if abs(idp - PROD_IDENTITY) < 0.5 else "F3 FIRES — harness/parity suspect"
        print(f"F3 (identity parity): identity MSLE {idp:.3f} vs prod ~{PROD_IDENTITY} -> {verdict}")
    for arm in ("log1p", "asinh"):
        if arm in results:
            m = results[arm]["msle"]
            beats = m < ALLZEROS_BASELINE
            print(
                f"F1 ({arm} vs all-zeros): MSLE {m:.3f} vs {ALLZEROS_BASELINE} -> "
                f"{'beats baseline' if beats else 'F1 FIRES — does NOT beat'}"
            )
    present = {a: results[a] for a in ARMS if a in results}
    valid_tails = {
        a: r["tail_msle"] for a, r in present.items() if not math.isnan(r["tail_msle"])
    }
    if present and valid_tails:
        msle_winner = min(present, key=lambda a: present[a]["msle"])
        tail_winner = min(valid_tails, key=lambda a: valid_tails[a])
        fires = msle_winner != tail_winner
        print(
            f"F4 (MSLE vs escalation tail): MSLE winner={msle_winner}, "
            f"tail winner={tail_winner} -> "
            f"{'F4 FIRES — MSLE winner loses the tail' if fires else 'consistent'}"
        )


def main():
    print(
        f"EXP-01 brown_cheese | REAL mechanism | SMOKE={SMOKE} | "
        f"steps={len(STEPS)} | n_estimators={PARAMS['n_estimators']}"
    )
    print(f"baselines: all-zeros MSLE={ALLZEROS_BASELINE} | production identity ~{PROD_IDENTITY}\n")
    results = {}
    for arm in ARMS:
        try:
            results[arm] = score_arm(run_arm(arm))
            print(f"[{arm:8s}] " + "  ".join(f"{k}={_fmt(v)}" for k, v in results[arm].items()))
        except Exception as e:
            print(f"[{arm:8s}] ERROR: {type(e).__name__}: {e}")
    _verdicts(results)
    return results


if __name__ == "__main__":
    main()
