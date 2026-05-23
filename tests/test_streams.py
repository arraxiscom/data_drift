"""Tests for synthetic streams."""

import pandas as pd

from drift_lab.streams import StreamConfig, generate_stream, train_reference_model


def test_generate_stream_stable_has_expected_columns():
    df = generate_stream("stable", StreamConfig(n_days=5, samples_per_day=10))
    assert len(df) == 50
    assert "channel_online" in df.columns
    assert "label" in df.columns
    assert df["stream_kind"].iloc[0] == "stable"


def test_covariate_drift_increases_online_share():
    cfg = StreamConfig(n_days=80, samples_per_day=50, drift_start_day=20)
    df = generate_stream("covariate_gradual", cfg)
    early = df[df["day"] < 25]["channel_online"].mean()
    late = df[df["day"] > 60]["channel_online"].mean()
    assert late > early + 0.2


def test_train_reference_model_predicts():
    df = generate_stream("stable", StreamConfig(n_days=10, samples_per_day=20))
    model = train_reference_model(df)
    cols = ["channel_online", "log_amount", "mcc_bucket", "foreign_flag", "weekend"]
    preds = model.predict(df[cols].iloc[:5])
    assert len(preds) == 5
