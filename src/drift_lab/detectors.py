"""Lightweight drift and changepoint detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_detect(
    values: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> int | None:
    """Return index of first CUSUM alarm, or None if none."""
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return None
    mean = values[: max(20, len(values) // 5)].mean()
    pos, neg = 0.0, 0.0
    for i, x in enumerate(values):
        pos = max(0.0, pos + x - mean - drift)
        neg = min(0.0, neg + x - mean + drift)
        if pos > threshold or abs(neg) > threshold:
            return i
    return None


def windowed_psi_series(
    reference: np.ndarray,
    stream: np.ndarray,
    window: int,
    step: int,
) -> pd.DataFrame:
    """Compute PSI in sliding windows along a stream."""
    from drift_lab.metrics import population_stability_index

    reference = np.asarray(reference, dtype=float)
    stream = np.asarray(stream, dtype=float)
    rows = []
    for start in range(0, len(stream) - window + 1, step):
        chunk = stream[start : start + window]
        psi = population_stability_index(reference, chunk)
        rows.append({"start": start, "psi": psi})
    return pd.DataFrame(rows)
