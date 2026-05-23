"""High-level analysis helpers for notebooks and monitoring demos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from drift_lab.constants import LEDGER_ROUTE_FEATURES
from drift_lab.metrics import (
    expected_calibration_error,
    population_stability_index,
    two_sample_ks,
)


def attach_predictions(
    model: ClassifierMixin,
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of df with prob_review and pred_label columns."""
    cols = feature_cols or LEDGER_ROUTE_FEATURES
    out = df.copy()
    probs = model.predict_proba(out[cols].values)[:, 1]
    out["prob_review"] = probs
    out["pred_label"] = (probs >= 0.5).astype(int)
    return out


def accuracy_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    pred_col: str = "pred_label",
) -> pd.DataFrame:
    """Accuracy and row counts per group value."""
    grouped = df.groupby(group_col, observed=True)
    rows = []
    for key, chunk in grouped:
        acc = (chunk[pred_col] == chunk[label_col]).mean()
        rows.append({group_col: key, "accuracy": float(acc), "n": len(chunk)})
    return pd.DataFrame(rows).sort_values(group_col)


def daily_label_rate(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """Mean label rate per day index."""
    return df.groupby("day")[label_col].mean()


def daily_channel_mix(df: pd.DataFrame) -> pd.Series:
    """Mean online-channel indicator per day."""
    return df.groupby("day")["channel_online"].mean()


def compare_windows(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    """PSI and KS for one feature between two windows."""
    ref = reference[feature].values
    cur = current[feature].values
    ks_stat, ks_p = two_sample_ks(ref, cur)
    return {
        "feature": feature,
        "psi": population_stability_index(ref, cur),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
    }


def outcome_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Accuracy and calibration in one dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "ece": expected_calibration_error(y_true, y_prob),
        "positive_rate": float(y_true.mean()),
        "mean_score": float(np.mean(y_prob)),
    }


def simulate_label_delay(
    df: pd.DataFrame,
    latency_days: int = 4,
) -> pd.DataFrame:
    """Add label_available_day for delayed ground-truth joins."""
    out = df.copy()
    out["label_available_day"] = out["day"] + latency_days
    return out


def joinable_at_day(df: pd.DataFrame, day: int) -> pd.DataFrame:
    """Rows whose labels are available on or before `day`."""
    if "label_available_day" not in df.columns:
        df = simulate_label_delay(df)
    return df[df["label_available_day"] <= day]
