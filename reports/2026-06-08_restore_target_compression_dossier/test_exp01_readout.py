"""Unit tests for the EXP-01 readout/metrics (exp01_readout.py).

The experiment's entire validity rests on the measurement, so it is tested
independently of the (slow, real) model fits. Run:
    conda run -n views_pipeline pytest reports/2026-06-08_restore_target_compression_dossier/test_exp01_readout.py -q
"""
import numpy as np
import pytest

from exp01_readout import (
    calibration_ratio,
    mean_pred_on_true_zeros,
    msle,
    readout,
    tail_conditional_msle,
)

E = np.e


# ---- msle (matches the platform log1p-both-sides definition) ----

def test_msle_zero_when_equal():
    assert msle([0, 1, 100], [0, 1, 100]) == 0.0


def test_msle_known_value():
    # log1p(e-1) = ln(e) = 1 ; log1p(0) = 0 ; squared error = 1 per cell
    assert msle([0, 0], [E - 1, E - 1]) == pytest.approx(1.0)


def test_msle_symmetric_in_log_space():
    a, b = [0, 5, 100], [2, 0, 80]
    assert msle(a, b) == pytest.approx(msle(b, a))


# ---- tail-conditional msle (escalation tail) ----

def test_tail_conditional_only_uses_cells_above_tau():
    # tau=100: only the 200-cell qualifies; pred there is exact -> 0
    assert tail_conditional_msle([0, 200], [999, 200], tau=100) == pytest.approx(0.0)


def test_tail_conditional_nan_when_no_tail():
    assert np.isnan(tail_conditional_msle([0, 1, 50], [0, 1, 50], tau=100))


# ---- over-prediction signature ----

def test_mean_pred_on_true_zeros():
    # true zeros at idx 0,1 -> mean of preds [3,7] = 5.0 ; the 5-cell ignored
    assert mean_pred_on_true_zeros([0, 0, 5], [3, 7, 100]) == pytest.approx(5.0)


def test_mean_pred_on_true_zeros_nan_when_none():
    assert np.isnan(mean_pred_on_true_zeros([1, 2], [1, 2]))


# ---- calibration-in-the-large ----

def test_calibration_ratio_over_forecast():
    # mean pred = 2, mean true = 1 -> 2.0
    assert calibration_ratio([0, 2], [1, 3]) == pytest.approx(2.0)


def test_calibration_ratio_nan_when_true_all_zero():
    assert np.isnan(calibration_ratio([0, 0], [1, 2]))


# ---- full readout ----

def test_readout_keys_and_baseline():
    r = readout([0, 0, 200], [1, 1, 150], tau=100)
    assert set(r) >= {
        "n", "msle", "msle_allzeros_baseline", "tail_msle",
        "mean_pred_on_true_zeros", "calibration_ratio", "mean_pred", "mean_actual",
    }
    assert r["n"] == 3
    assert r["msle_allzeros_baseline"] == pytest.approx(msle([0, 0, 200], [0, 0, 0]))
    assert r["mean_pred_on_true_zeros"] == pytest.approx(1.0)
    assert r["mean_actual"] == pytest.approx(200 / 3)
