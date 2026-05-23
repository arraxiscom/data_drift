"""Smoke tests mirroring notebook workflows (no Jupyter required)."""

from drift_lab import StreamConfig, build_ledger_route_model, generate_stream
from drift_lab.analysis import attach_predictions, compare_windows, simulate_label_delay
from drift_lab.detectors import cusum_detect, windowed_psi_series
from drift_lab.metrics import rolling_accuracy


def test_notebook02_covariate_workflow():
    cfg = StreamConfig()
    ref = generate_stream("stable", cfg)
    cov = generate_stream("covariate_gradual", cfg)
    model = build_ledger_route_model(cfg)
    scored = attach_predictions(model, cov)
    late = cov[cov["day"] >= 90]
    ref_win = ref[ref["day"] < 30]
    ch = compare_windows(ref_win, late, "channel_online")
    assert ch["psi"] >= 0.0
    acc_online = scored[scored["channel_online"] == 1]
    assert len(acc_online) > 100
    psi_df = windowed_psi_series(
        ref_win["log_amount"].values,
        cov["log_amount"].values,
        window=2000,
        step=400,
    )
    assert not psi_df.empty


def test_notebook03_concept_ece_rises():
    cfg = StreamConfig()
    model = build_ledger_route_model(cfg)
    scored = attach_predictions(model, generate_stream("concept_abrupt", cfg))
    pre = scored["day"] < cfg.concept_shift_day
    from drift_lab.analysis import outcome_summary

    pre_m = outcome_summary(
        scored.loc[pre, "label"],
        scored.loc[pre, "pred_label"],
        scored.loc[pre, "prob_review"],
    )
    post_m = outcome_summary(
        scored.loc[~pre, "label"],
        scored.loc[~pre, "pred_label"],
        scored.loc[~pre, "prob_review"],
    )
    assert post_m["ece"] > pre_m["ece"]


def test_notebook04_cusum_gradual_only():
    cfg = StreamConfig()
    cov = generate_stream("covariate_gradual", cfg)
    mix = cov.groupby("day")["channel_online"].mean().values
    mix = (mix - mix.mean()) / mix.std()
    assert cusum_detect(mix, threshold=4.0) is not None


def test_notebook05_delayed_join():
    cfg = StreamConfig()
    scored = attach_predictions(
        build_ledger_route_model(cfg),
        generate_stream("covariate_gradual", cfg),
    )
    delayed = simulate_label_delay(scored, latency_days=4)
    joinable = delayed[delayed["label_available_day"] <= delayed["day"].max()]
    acc = rolling_accuracy(
        joinable["label"].values,
        joinable["pred_label"].values,
        window=800,
    )
    assert acc["accuracy"].dropna().between(0, 1).all()
