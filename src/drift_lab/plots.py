"""Matplotlib helpers with a consistent style for story exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "figure.figsize": (8, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_channel_mix(df: pd.DataFrame, path: Path) -> dict:
    """Area-style channel mix over days."""
    apply_style()
    mix = df.groupby("day")["channel_online"].mean()
    offline = 1.0 - mix
    fig, ax = plt.subplots()
    days = mix.index.values
    ax.fill_between(days, 0, offline.values, label="Offline", alpha=0.75)
    ax.fill_between(days, offline.values, 1.0, label="Online", alpha=0.75)
    ax.set_xlabel("Day")
    ax.set_ylabel("Share of traffic")
    ax.set_title("LedgerRoute: sales channel mix over time")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    save_figure(path)
    return {
        "day_start_online_share": float(mix.iloc[0]),
        "day_end_online_share": float(mix.iloc[-1]),
    }


def plot_rolling_accuracy(acc_df: pd.DataFrame, path: Path, title: str) -> dict:
    apply_style()
    fig, ax = plt.subplots()
    valid = acc_df.dropna()
    ax.plot(valid["index"], valid["accuracy"], color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Rolling accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    save_figure(path)
    return {
        "final_rolling_accuracy": float(valid["accuracy"].iloc[-1]),
        "min_rolling_accuracy": float(valid["accuracy"].min()),
    }


def plot_psi_series(psi_df: pd.DataFrame, path: Path) -> dict:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(psi_df["start"], psi_df["psi"], color="#dc2626", linewidth=1.5)
    ax.axhline(0.1, color="#6b7280", linestyle="--", label="Common review threshold (0.1)")
    ax.axhline(0.25, color="#9ca3af", linestyle=":", label="Strong shift (0.25)")
    ax.set_xlabel("Window start (sample index)")
    ax.set_ylabel("PSI (log_amount)")
    ax.set_title("Feature drift: PSI vs reference window")
    ax.legend()
    save_figure(path)
    return {
        "max_psi": float(psi_df["psi"].max()),
        "mean_psi": float(psi_df["psi"].mean()),
    }


def plot_shift_comparison(
    series: dict[str, np.ndarray],
    path: Path,
) -> dict:
    """Overlay gradual drift, abrupt shift, and noisy baseline."""
    apply_style()
    fig, ax = plt.subplots()
    for name, values in series.items():
        ax.plot(values, label=name, linewidth=1.2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Monitoring statistic (normalized)")
    ax.set_title("Gradual drift, abrupt shift, and seasonal noise")
    ax.legend()
    save_figure(path)
    return {k: {"final": float(v[-1]), "max": float(v.max())} for k, v in series.items()}


def plot_prediction_histogram(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
    path: Path,
) -> dict:
    apply_style()
    fig, ax = plt.subplots()
    ax.hist(ref_probs, bins=30, alpha=0.6, label="Reference week", density=True)
    ax.hist(cur_probs, bins=30, alpha=0.6, label="Current week", density=True)
    ax.set_xlabel("Predicted P(manual review)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction distribution drift")
    ax.legend()
    save_figure(path)
    stat, pval = __import__("scipy.stats", fromlist=["stats"]).stats.ks_2samp(ref_probs, cur_probs)
    return {"ks_statistic": float(stat), "ks_pvalue": float(pval)}


def plot_embedding_distance(distances: np.ndarray, path: Path) -> dict:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(distances, color="#7c3aed", linewidth=1.2)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Mean cosine distance to reference")
    ax.set_title("Embedding drift surrogate (provider swap)")
    save_figure(path)
    return {
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }
