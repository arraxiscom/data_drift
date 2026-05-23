"""Synthetic LedgerRoute expense-routing streams with controlled drift types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DriftKind = Literal[
    "stable",
    "covariate_gradual",
    "concept_abrupt",
    "prior_shift",
    "noise_only",
]


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for reproducible synthetic production streams."""

    n_days: int = 120
    samples_per_day: int = 200
    seed: int = 42
    drift_start_day: int = 40
    concept_shift_day: int = 70


def _feature_matrix(
    rng: np.random.Generator,
    n: int,
    online_fraction: float,
    amount_scale: float = 1.0,
) -> pd.DataFrame:
    channel = rng.binomial(1, online_fraction, size=n)
    amount = rng.lognormal(mean=3.2 + 0.15 * channel, sigma=0.45, size=n)
    amount *= amount_scale
    mcc = rng.integers(0, 8, size=n)
    foreign_flag = rng.binomial(1, 0.08 + 0.04 * channel, size=n)
    weekend = rng.binomial(1, 0.28, size=n)
    return pd.DataFrame(
        {
            "channel_online": channel,
            "log_amount": np.log1p(amount),
            "mcc_bucket": mcc,
            "foreign_flag": foreign_flag,
            "weekend": weekend,
        }
    )


def _label_from_features(
    rng: np.random.Generator,
    df: pd.DataFrame,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    logits = df.values @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, probs).astype(int)


def generate_stream(
    kind: DriftKind,
    config: StreamConfig | None = None,
) -> pd.DataFrame:
    """Generate a day-indexed stream with features, labels, and metadata."""
    cfg = config or StreamConfig()
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []

    base_weights = np.array([0.9, 0.55, 0.12, 0.35, -0.25])
    shifted_weights = np.array([0.35, 0.62, 0.18, 0.40, -0.20])

    for day in range(cfg.n_days):
        if kind == "stable":
            online_frac = 0.25
            weights, bias = base_weights, -1.1
        elif kind == "covariate_gradual":
            progress = max(0.0, (day - cfg.drift_start_day) / max(1, cfg.n_days - cfg.drift_start_day))
            online_frac = 0.25 + 0.55 * min(1.0, progress)
            weights, bias = base_weights, -1.1
        elif kind == "concept_abrupt":
            online_frac = 0.30 if day < cfg.concept_shift_day else 0.32
            if day < cfg.concept_shift_day:
                weights, bias = base_weights, -1.1
            else:
                weights, bias = shifted_weights, -0.35
        elif kind == "prior_shift":
            online_frac = 0.28
            weights, bias = base_weights, -1.1 if day < cfg.concept_shift_day else -0.55
        elif kind == "noise_only":
            online_frac = 0.27 + 0.03 * np.sin(2 * np.pi * day / 14)
            weights, bias = base_weights, -1.1
        else:
            raise ValueError(f"Unknown stream kind: {kind}")

        x_df = _feature_matrix(rng, cfg.samples_per_day, online_frac)
        y = _label_from_features(rng, x_df, weights, bias)
        for i in range(cfg.samples_per_day):
            row = x_df.iloc[i].to_dict()
            row["label"] = int(y[i])
            row["day"] = day
            row["stream_kind"] = kind
            rows.append(row)

    return pd.DataFrame(rows)


def train_reference_model(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> Pipeline:
    """Fit a simple classifier on the first portion of a stream."""
    cols = feature_cols or [
        "channel_online",
        "log_amount",
        "mcc_bucket",
        "foreign_flag",
        "weekend",
    ]
    x_train = df[cols].values
    y_train = df["label"].values
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def build_ledger_route_model(config: StreamConfig | None = None) -> Pipeline:
    """Train on an early stable window (days 0–29) for saga demos."""
    cfg = config or StreamConfig()
    stable = generate_stream("stable", cfg)
    early = stable[stable["day"] < 30]
    return train_reference_model(early)
