"""Export PNG + JSON artifacts for the Arraxis living-model saga."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_lab.constants import LEDGER_ROUTE_FEATURES
from drift_lab.detectors import cusum_detect, windowed_psi_series
from drift_lab.metrics import (
    expected_calibration_error,
    rolling_accuracy,
    two_sample_ks,
)
from drift_lab.plots import (
    plot_channel_mix,
    plot_embedding_distance,
    plot_prediction_histogram,
    plot_psi_series,
    plot_rolling_accuracy,
    plot_shift_comparison,
)
from drift_lab.streams import StreamConfig, build_ledger_route_model, generate_stream

FEATURE_COLS = LEDGER_ROUTE_FEATURES


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _predict(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(df[FEATURE_COLS].values)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def export_all(artifacts_dir: Path) -> dict[str, dict]:
    """Generate all story figures and return metrics keyed by basename."""
    cfg = StreamConfig()
    model = build_ledger_route_model(cfg)
    ref_stable = generate_stream("stable", cfg)
    ref_window = ref_stable[ref_stable["day"] < 30]

    results: dict[str, dict] = {}

    # Story 1: silent scores — accuracy on covariate stream
    cov_df = generate_stream("covariate_gradual", cfg)
    cov_preds, cov_probs = _predict(model, cov_df)
    acc_df = rolling_accuracy(cov_df["label"].values, cov_preds, window=800)
    m1 = plot_rolling_accuracy(
        acc_df,
        artifacts_dir / "living_model_silent_scores_accuracy.png",
        "LedgerRoute: rolling accuracy under covariate drift",
    )
    m1["reference_train_days"] = 30
    m1["stream_kind"] = "covariate_gradual"
    _write_json(artifacts_dir / "living_model_silent_scores_accuracy.json", m1)
    results["living_model_silent_scores_accuracy"] = m1

    # Story 2: channel mix + PSI
    m2a = plot_channel_mix(
        cov_df, artifacts_dir / "living_model_covariate_channel_mix.png"
    )
    _write_json(artifacts_dir / "living_model_covariate_channel_mix.json", m2a)
    results["living_model_covariate_channel_mix"] = m2a

    ref_channel = ref_window["log_amount"].values
    stream_channel = cov_df["log_amount"].values
    psi_df = windowed_psi_series(ref_channel, stream_channel, window=2000, step=400)
    m2b = plot_psi_series(psi_df, artifacts_dir / "living_model_covariate_psi.png")
    ks_stat, ks_p = two_sample_ks(ref_channel, stream_channel[-5000:])
    m2b["ks_statistic"] = ks_stat
    m2b["ks_pvalue"] = ks_p
    _write_json(artifacts_dir / "living_model_covariate_psi.json", m2b)
    results["living_model_covariate_psi"] = m2b

    # Story 3: concept drift — calibration + accuracy
    concept_df = generate_stream("concept_abrupt", cfg)
    c_preds, c_probs = _predict(model, concept_df)
    pre = concept_df["day"] < cfg.concept_shift_day
    post = concept_df["day"] >= cfg.concept_shift_day
    m3 = {
        "accuracy_pre_shift": float(
            (c_preds[pre.values] == concept_df.loc[pre, "label"].values).mean()
        ),
        "accuracy_post_shift": float(
            (c_preds[post.values] == concept_df.loc[post, "label"].values).mean()
        ),
        "ece_pre_shift": expected_calibration_error(
            concept_df.loc[pre, "label"].values, c_probs[pre.values]
        ),
        "ece_post_shift": expected_calibration_error(
            concept_df.loc[post, "label"].values, c_probs[post.values]
        ),
        "concept_shift_day": cfg.concept_shift_day,
    }
    acc_concept = rolling_accuracy(concept_df["label"].values, c_preds, window=800)
    m3.update(
        plot_rolling_accuracy(
            acc_concept,
            artifacts_dir / "living_model_concept_regime_accuracy.png",
            "Accuracy drop after label semantics change",
        )
    )
    _write_json(artifacts_dir / "living_model_concept_regime_accuracy.json", m3)
    results["living_model_concept_regime_accuracy"] = m3

    # Story 4: shift vs drift vs noise
    gradual = cov_df.groupby("day")["channel_online"].mean().values
    abrupt = concept_df.groupby("day")["label"].mean().values
    noise = generate_stream("noise_only", cfg).groupby("day")["label"].mean().values

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-8)

    m4 = plot_shift_comparison(
        {
            "Gradual (channel mix)": _norm(gradual),
            "Abrupt (label rate)": _norm(abrupt),
            "Seasonal noise": _norm(noise),
        },
        artifacts_dir / "living_model_shift_noise_compare.png",
    )
    alarm = cusum_detect(_norm(gradual), threshold=4.0)
    m4["cusum_alarm_day_gradual"] = int(alarm) if alarm is not None else None
    _write_json(artifacts_dir / "living_model_shift_noise_compare.json", m4)
    results["living_model_shift_noise_compare"] = m4

    # Story 5: prediction drift (delayed labels proxy)
    late = cov_df[cov_df["day"] >= 80]
    early = cov_df[(cov_df["day"] >= 30) & (cov_df["day"] < 50)]
    _, early_probs = _predict(model, early)
    _, late_probs = _predict(model, late)
    m5 = plot_prediction_histogram(
        early_probs,
        late_probs,
        artifacts_dir / "living_model_prediction_drift.png",
    )
    m5["median_label_latency_days"] = 4
    _write_json(artifacts_dir / "living_model_prediction_drift.json", m5)
    results["living_model_prediction_drift"] = m5

    # Story 6: embedding surrogate
    rng = np.random.default_rng(7)
    n_batches = 60
    ref_emb = rng.normal(size=(200, 32))
    ref_emb /= np.linalg.norm(ref_emb, axis=1, keepdims=True)
    distances = []
    for t in range(n_batches):
        drift_scale = 0.02 * max(0, t - 25)
        batch = ref_emb + rng.normal(scale=0.05 + drift_scale, size=ref_emb.shape)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        cos = 1.0 - (batch @ ref_emb.T).max(axis=1)
        distances.append(float(cos.mean()))
    dist_arr = np.array(distances)
    m6 = plot_embedding_distance(
        dist_arr, artifacts_dir / "living_model_embedding_drift.png"
    )
    m6["note"] = "Synthetic surrogate; not a live LLM endpoint."
    _write_json(artifacts_dir / "living_model_embedding_drift.json", m6)
    results["living_model_embedding_drift"] = m6

    # Hub summary
    hub = {
        "model": "LogisticRegression on LedgerRoute synthetic features",
        "stable_accuracy_on_train_window": float(
            (model.predict(ref_window[FEATURE_COLS].values) == ref_window["label"].values).mean()
        ),
        "covariate_final_rolling_accuracy": m1["final_rolling_accuracy"],
        "max_channel_psi": m2b["max_psi"],
        "accuracy_drop_after_concept_shift": m3["accuracy_pre_shift"] - m3["accuracy_post_shift"],
    }
    _write_json(artifacts_dir / "living_model_hub_summary.json", hub)
    results["living_model_hub_summary"] = hub

    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "artifacts" / "story_assets"
    export_all(artifacts)
    print(f"Wrote figures to {artifacts}")


if __name__ == "__main__":
    main()
