"""Tests for monitoring metrics."""

import numpy as np

from drift_lab.metrics import (
    expected_calibration_error,
    population_stability_index,
    rolling_accuracy,
)


def test_psi_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=500)
    psi = population_stability_index(x, x)
    assert psi < 0.01


def test_psi_positive_for_shifted_samples():
    ref = np.random.default_rng(0).normal(size=500)
    cur = np.random.default_rng(1).normal(loc=2.0, size=500)
    psi = population_stability_index(ref, cur)
    assert psi > 0.1


def test_rolling_accuracy_bounds():
    y = np.array([0, 1, 0, 1, 0, 1] * 100)
    pred = y.copy()
    df = rolling_accuracy(y, pred, window=50)
    assert df["accuracy"].dropna().between(0, 1).all()


def test_ece_perfect_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    ece = expected_calibration_error(y, p, n_bins=2)
    assert ece < 0.15
