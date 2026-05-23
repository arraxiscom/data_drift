"""Inline figures for notebooks (returns matplotlib Figure objects)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_lab.plots import apply_style


def channel_mix_figure(df: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    """Stacked offline/online share over days."""
    apply_style()
    mix = df.groupby("day")["channel_online"].mean()
    offline = 1.0 - mix
    fig, ax = plt.subplots()
    days = mix.index.values
    ax.fill_between(days, 0, offline.values, label="Offline", alpha=0.75)
    ax.fill_between(days, offline.values, 1.0, label="Online", alpha=0.75)
    ax.set_xlabel("Day")
    ax.set_ylabel("Share of traffic")
    ax.set_title("Sales channel mix over time")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    meta = {
        "day_start_online_share": float(mix.iloc[0]),
        "day_end_online_share": float(mix.iloc[-1]),
    }
    return fig, meta


def rolling_accuracy_figure(
    acc_df: pd.DataFrame,
    title: str = "Rolling accuracy",
) -> tuple[Any, dict[str, float]]:
    apply_style()
    fig, ax = plt.subplots()
    valid = acc_df.dropna()
    ax.plot(valid["index"], valid["accuracy"], color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Rolling accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, {
        "final_rolling_accuracy": float(valid["accuracy"].iloc[-1]),
        "min_rolling_accuracy": float(valid["accuracy"].min()),
    }


def psi_series_figure(psi_df: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(psi_df["start"], psi_df["psi"], color="#dc2626", linewidth=1.5)
    ax.axhline(0.1, color="#6b7280", linestyle="--", label="Review threshold (0.1)")
    ax.axhline(0.25, color="#9ca3af", linestyle=":", label="Strong shift (0.25)")
    ax.set_xlabel("Window start (sample index)")
    ax.set_ylabel("PSI")
    ax.set_title("Windowed PSI vs reference batch")
    ax.legend()
    fig.tight_layout()
    return fig, {
        "max_psi": float(psi_df["psi"].max()),
        "mean_psi": float(psi_df["psi"].mean()),
    }


def prediction_hist_figure(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
) -> tuple[Any, dict[str, float]]:
    from scipy import stats

    apply_style()
    fig, ax = plt.subplots()
    ax.hist(ref_probs, bins=30, alpha=0.6, label="Reference window", density=True)
    ax.hist(cur_probs, bins=30, alpha=0.6, label="Current window", density=True)
    ax.set_xlabel("Predicted P(manual review)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction distribution shift")
    ax.legend()
    fig.tight_layout()
    stat, pval = stats.ks_2samp(ref_probs, cur_probs)
    return fig, {"ks_statistic": float(stat), "ks_pvalue": float(pval)}


def calibration_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[Any, dict[str, float]]:
    from drift_lab.metrics import expected_calibration_error

    apply_style()
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        confs.append(y_prob[mask].mean())
        accs.append(y_true[mask].mean())
        counts.append(mask.sum())
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="#6b7280", label="Perfect calibration")
    ax.bar(confs, accs, width=0.08, alpha=0.7, label="Empirical bins")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return fig, {"ece": float(ece)}


def shift_overlay_figure(series: dict[str, np.ndarray]) -> Any:
    apply_style()
    fig, ax = plt.subplots()
    for name, values in series.items():
        ax.plot(values, label=name, linewidth=1.2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized monitoring statistic")
    ax.set_title("Gradual drift, abrupt shift, and seasonal noise")
    ax.legend()
    fig.tight_layout()
    return fig


def embedding_distance_figure(distances: np.ndarray) -> tuple[Any, dict[str, float]]:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(distances, color="#7c3aed", linewidth=1.2)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Mean cosine distance to reference")
    ax.set_title("Embedding drift surrogate")
    fig.tight_layout()
    return fig, {
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }
