"""Tests for analysis helpers."""

import pandas as pd

from drift_lab.analysis import (
    accuracy_by_group,
    attach_predictions,
    compare_windows,
    outcome_summary,
    simulate_label_delay,
)
from drift_lab.streams import StreamConfig, generate_stream, train_reference_model


def test_attach_predictions_adds_columns():
    cfg = StreamConfig(n_days=5, samples_per_day=50)
    df = generate_stream("stable", cfg)
    model = train_reference_model(df)
    scored = attach_predictions(model, df)
    assert "prob_review" in scored.columns
    assert "pred_label" in scored.columns


def test_accuracy_by_group():
    cfg = StreamConfig(n_days=10, samples_per_day=100)
    df = generate_stream("stable", cfg)
    model = train_reference_model(df[df["day"] < 5])
    scored = attach_predictions(model, df)
    out = accuracy_by_group(scored, "channel_online")
    assert set(out.columns) == {"channel_online", "accuracy", "n"}


def test_compare_windows_psi_nonnegative():
    cfg = StreamConfig(n_days=30, samples_per_day=100)
    ref = generate_stream("stable", cfg)
    cur = generate_stream("covariate_gradual", cfg)
    m = compare_windows(ref[ref["day"] < 10], cur[cur["day"] > 20], "log_amount")
    assert m["psi"] >= 0.0


def test_simulate_label_delay():
    cfg = StreamConfig(n_days=3, samples_per_day=10)
    df = generate_stream("stable", cfg)
    delayed = simulate_label_delay(df, latency_days=4)
    assert (delayed["label_available_day"] == delayed["day"] + 4).all()


def test_outcome_summary_keys():
    cfg = StreamConfig(n_days=5, samples_per_day=40)
    df = generate_stream("stable", cfg)
    model = train_reference_model(df)
    scored = attach_predictions(model, df)
    s = outcome_summary(scored["label"], scored["pred_label"], scored["prob_review"])
    assert {"accuracy", "ece", "positive_rate", "mean_score"} <= s.keys()
