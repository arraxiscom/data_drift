"""Monitoring metrics: PSI, KS, rolling accuracy, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between two numeric or binary samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    uniq = np.unique(expected)
    if len(uniq) <= 2:
        exp_pct = np.array([1.0 - expected.mean(), expected.mean()]) + epsilon
        act_pct = np.array([1.0 - actual.mean(), actual.mean()]) + epsilon
        exp_pct = exp_pct / exp_pct.sum()
        act_pct = act_pct / act_pct.sum()
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    quantiles = np.linspace(0, 1, n_bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)
    exp_pct = exp_counts / max(exp_counts.sum(), 1) + epsilon
    act_pct = act_counts / max(act_counts.sum(), 1) + epsilon
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def two_sample_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and p-value."""
    stat, pval = stats.ks_2samp(reference, current)
    return float(stat), float(pval)


def rolling_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 500,
) -> pd.DataFrame:
    """Rolling accuracy over a prediction stream."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    series = pd.Series(correct).rolling(window, min_periods=max(50, window // 5)).mean()
    return pd.DataFrame({"index": np.arange(len(series)), "accuracy": series.values})


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for binary probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)
