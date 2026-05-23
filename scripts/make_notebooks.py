#!/usr/bin/env python3
"""Generate minimal runnable notebooks for the public repo."""

from pathlib import Path

NOTEBOOKS = [
    (
        "01_baseline_stream.ipynb",
        "from drift_lab.streams import StreamConfig, build_ledger_route_model, generate_stream\n\n"
        "cfg = StreamConfig(n_days=40, samples_per_day=100)\n"
        "stable = generate_stream('stable', cfg)\n"
        "model = build_ledger_route_model(cfg)\n"
        "acc = (model.predict(stable.iloc[:500, :5]) == stable.iloc[:500]['label']).mean()\n"
        "print('Train-window accuracy on stable stream:', round(float(acc), 4))",
    ),
    (
        "02_covariate_drift.ipynb",
        "from drift_lab.streams import StreamConfig, build_ledger_route_model, generate_stream\n"
        "from drift_lab.metrics import population_stability_index, two_sample_ks\n\n"
        "cfg = StreamConfig()\n"
        "ref = generate_stream('stable', cfg)\n"
        "cov = generate_stream('covariate_gradual', cfg)\n"
        "ref_ch = ref[ref['day'] < 30]['channel_online'].values\n"
        "late_ch = cov[cov['day'] > 80]['channel_online'].values\n"
        "print('PSI channel_online:', round(population_stability_index(ref_ch, late_ch), 4))\n"
        "print('KS:', two_sample_ks(ref_ch, late_ch))",
    ),
    (
        "03_concept_drift.ipynb",
        "from drift_lab.streams import StreamConfig, build_ledger_route_model, generate_stream\n"
        "from drift_lab.metrics import expected_calibration_error\n\n"
        "cfg = StreamConfig()\n"
        "model = build_ledger_route_model(cfg)\n"
        "df = generate_stream('concept_abrupt', cfg)\n"
        "cols = ['channel_online','log_amount','mcc_bucket','foreign_flag','weekend']\n"
        "probs = model.predict_proba(df[cols].values)[:, 1]\n"
        "pre = df['day'] < cfg.concept_shift_day\n"
        "print('ECE pre:', round(expected_calibration_error(df.loc[pre,'label'], probs[pre.values]), 4))\n"
        "print('ECE post:', round(expected_calibration_error(df.loc[~pre,'label'], probs[(~pre).values]), 4))",
    ),
    (
        "04_shift_vs_drift_noise.ipynb",
        "from drift_lab.streams import StreamConfig, generate_stream\n"
        "from drift_lab.detectors import cusum_detect\n\n"
        "cfg = StreamConfig()\n"
        "cov = generate_stream('covariate_gradual', cfg)\n"
        "mix = cov.groupby('day')['channel_online'].mean().values\n"
        "mix = (mix - mix.mean()) / mix.std()\n"
        "print('CUSUM alarm day (gradual mix):', cusum_detect(mix))",
    ),
    (
        "05_prediction_and_delayed_labels.ipynb",
        "from drift_lab.streams import StreamConfig, build_ledger_route_model, generate_stream\n"
        "from drift_lab.metrics import two_sample_ks\n\n"
        "cfg = StreamConfig()\n"
        "model = build_ledger_route_model(cfg)\n"
        "df = generate_stream('covariate_gradual', cfg)\n"
        "cols = ['channel_online','log_amount','mcc_bucket','foreign_flag','weekend']\n"
        "early = df[(df['day'] >= 30) & (df['day'] < 50)]\n"
        "late = df[df['day'] >= 90]\n"
        "p0 = model.predict_proba(early[cols].values)[:, 1]\n"
        "p1 = model.predict_proba(late[cols].values)[:, 1]\n"
        "print('KS on predicted scores:', two_sample_ks(p0, p1))",
    ),
    (
        "06_llm_embedding_surrogate.ipynb",
        "import numpy as np\n\n"
        "rng = np.random.default_rng(0)\n"
        "ref = rng.normal(size=(100, 16))\n"
        "ref /= np.linalg.norm(ref, axis=1, keepdims=True)\n"
        "batch = ref + rng.normal(scale=0.2, size=ref.shape)\n"
        "batch /= np.linalg.norm(batch, axis=1, keepdims=True)\n"
        "dist = 1.0 - (batch @ ref.T).max(axis=1).mean()\n"
        "print('Mean cosine distance after synthetic provider shift:', round(float(dist), 4))",
    ),
]


def _nbformat(cells_source: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["LedgerRoute synthetic demo. See https://arraxis.com/living-model/"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in cells_source.split("\n")],
            },
        ],
    }


def main() -> None:
    import json

    out = Path(__file__).resolve().parents[1] / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    for name, code in NOTEBOOKS:
        path = out / name
        path.write_text(json.dumps(_nbformat(code), indent=1), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
