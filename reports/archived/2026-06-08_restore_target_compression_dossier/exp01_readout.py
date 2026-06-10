"""Pure readout / metric functions for EXP-01 (the target-compression pre-screen).

Separated from the experiment *driver* so the measurement — on which the entire
experiment's validity rests — is unit-tested independently of the (slow, real)
model fits (single-responsibility: this module measures, the driver orchestrates).

Every function takes aligned array-likes of non-negative counts. ``msle`` matches
the platform definition (mean of (log1p(pred) - log1p(true))**2, applied
symmetrically), i.e. ``views_evaluation ... calculate_msle_native``.
"""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def msle(y_true, y_pred) -> float:
    """Mean squared log error: mean((log1p(pred) - log1p(true))**2)."""
    return float(np.mean((np.log1p(_arr(y_pred)) - np.log1p(_arr(y_true))) ** 2))


def tail_conditional_msle(y_true, y_pred, tau) -> float:
    """MSLE restricted to the escalation tail (cells with observed > tau).

    Returns nan when no cell exceeds tau. This is the operationally critical
    readout the method review flagged: an MSLE winner can still lose here.
    """
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    mask = y_true > tau
    if not mask.any():
        return float("nan")
    return msle(y_true[mask], y_pred[mask])


def mean_pred_on_true_zeros(y_true, y_pred) -> float:
    """Mean prediction on cells whose observed value is exactly zero — the
    over-prediction signature (production identity sits at ~16). nan if none."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    mask = y_true == 0
    if not mask.any():
        return float("nan")
    return float(np.mean(y_pred[mask]))


def calibration_ratio(y_true, y_pred) -> float:
    """Calibration-in-the-large: mean(pred) / mean(true). 1.0 is perfect;
    >1 over-forecasts, <1 under-forecasts. nan if mean(true) == 0."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = float(np.mean(y_true))
    if denom == 0.0:
        return float("nan")
    return float(np.mean(y_pred)) / denom


def readout(y_true, y_pred, tau) -> dict:
    """The full multi-component readout (per 05_preanalysis_exp01.md) as a dict.

    NOT MSLE alone: pairs the headline MSLE with the all-zeros baseline, the
    escalation-tail MSLE, the over-prediction signature, and calibration — so a
    transform that wins MSLE by compressing the tail is visible, not hidden.
    """
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return {
        "n": int(y_true.size),
        "msle": msle(y_true, y_pred),
        "msle_allzeros_baseline": msle(y_true, np.zeros_like(y_true)),
        "tail_msle": tail_conditional_msle(y_true, y_pred, tau),
        "mean_pred_on_true_zeros": mean_pred_on_true_zeros(y_true, y_pred),
        "calibration_ratio": calibration_ratio(y_true, y_pred),
        "mean_pred": float(np.mean(y_pred)),
        "mean_actual": float(np.mean(y_true)),
    }
