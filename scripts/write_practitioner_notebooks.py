#!/usr/bin/env python3
"""Write practitioner notebooks (source of truth for notebooks/*.ipynb)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
}


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    }


def code(text: str) -> dict:
    truncated = text.strip("\n")
    lines = truncated.split("\n")
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }


def notebook(*cells: dict) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": KERNEL,
        "cells": list(cells),
    }


ALL_NOTEBOOKS: list[tuple[str, dict]] = []


def _register(name: str, nb: dict) -> None:
    ALL_NOTEBOOKS.append((name, nb))


_register(
    "01_baseline_stream.ipynb",
    notebook(
        md(
            """# 01 — LedgerRoute baseline and reference window

**LedgerRoute** is a synthetic expense-routing classifier: each row is a corporate card transaction with tabular features, and the label is whether finance must manually review it.

Monitoring starts with a **reference window**—the period you treat as representative of training. We fit on days 0–29 of a stable stream and evaluate on days 30–39 before running drift scenarios.

Narrative context: [The living model](https://arraxis.com/living-model/) on Arraxis."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import pandas as pd

from drift_lab import LEDGER_ROUTE_FEATURES, StreamConfig, generate_stream
from drift_lab.analysis import attach_predictions, outcome_summary
from drift_lab.streams import train_reference_model

cfg = StreamConfig(n_days=40, samples_per_day=200, seed=42)
stable = generate_stream("stable", cfg)
print(f"{len(stable):,} rows, days 0–{stable['day'].max()}")
stable.head()"""
        ),
        md(
            """## Feature schema

| Column | Meaning |
|--------|---------|
| `channel_online` | 1 if submitted via online checkout |
| `log_amount` | log1p(expense amount) |
| `mcc_bucket` | merchant category bucket (0–7) |
| `foreign_flag` | cross-border indicator |
| `weekend` | transaction on Sat/Sun |
| `day` | synthetic day index |
| `label` | 1 = needs manual review |

Labels are drawn from a logistic model on the feature vector (see `drift_lab.streams`)."""
        ),
        code("stable[LEDGER_ROUTE_FEATURES + ['label']].describe().round(3).T"),
        md("## Fit on days 0–29, evaluate on 30–39"),
        code(
            """train = stable[stable["day"] < 30]
holdout = stable[stable["day"] >= 30]

model = train_reference_model(train)
train_scored = attach_predictions(model, train)
holdout_scored = attach_predictions(model, holdout)

for name, part in [("Train", train_scored), ("Holdout", holdout_scored)]:
    m = outcome_summary(part["label"], part["pred_label"], part["prob_review"])
    print(name, {k: round(v, 4) for k, v in m.items()})"""
        ),
        md("## Label rate by day (stable stream)"),
        code(
            """ax = stable.groupby("day")["label"].mean().plot(figsize=(8, 3), title="Review rate by day (stable)")
ax.set_ylabel("P(manual review)")
ax.figure.tight_layout()"""
        ),
        md(
            """## Takeaway

1. Freeze **which days** define your reference window and persist them in config.
2. Store **model version + feature schema** with every prediction row before drift tests.
3. Hold-out on adjacent stable days is a sanity check—not a substitute for production monitoring."""
        ),
    ),
)

_register(
    "02_covariate_drift.ipynb",
    notebook(
        md(
            """# 02 — Covariate shift (when P(X) moves)

**Covariate shift** means the input distribution changes while the conditional label rule P(Y|X) stays fixed. LedgerRoute simulates marketing pushing spend from offline procurement to online checkout (`channel_online` share rises from ~25% toward ~80%).

Symptoms in production:
- Global accuracy may look flat early on.
- Accuracy on the **growing segment** degrades first.
- PSI on one feature can stay low while the **mix plot** already tells the story."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import pandas as pd

from drift_lab import StreamConfig, build_ledger_route_model, generate_stream
from drift_lab.analysis import accuracy_by_group, attach_predictions, compare_windows
from drift_lab.detectors import windowed_psi_series
from drift_lab.viz import channel_mix_figure, psi_series_figure

cfg = StreamConfig()
ref = generate_stream("stable", cfg)
cov = generate_stream("covariate_gradual", cfg)
model = build_ledger_route_model(cfg)  # trained on stable days 0–29

ref_win = ref[ref["day"] < 30]
cov_scored = attach_predictions(model, cov)
print("Reference rows:", len(ref_win), "| Covariate stream rows:", len(cov))"""
        ),
        md("## Visual: channel mix over time"),
        code(
            """fig, mix_meta = channel_mix_figure(cov)
mix_meta"""
        ),
        md("## PSI and KS: reference vs late traffic"),
        code(
            """late = cov[cov["day"] >= 90]
rows = []
for feat in ["channel_online", "log_amount", "foreign_flag"]:
    rows.append(compare_windows(ref_win, late, feat))
pd.DataFrame(rows).round(4)"""
        ),
        md("## Segment accuracy (frozen model)"),
        code(
            """acc = accuracy_by_group(cov_scored, "channel_online")
acc.assign(accuracy=acc["accuracy"].round(4))"""
        ),
        md("## Windowed PSI on `log_amount` along the stream"),
        code(
            """psi_df = windowed_psi_series(
    ref_win["log_amount"].values,
    cov["log_amount"].values,
    window=2000,
    step=400,
)
fig, psi_meta = psi_series_figure(psi_df)
psi_meta"""
        ),
        md(
            """## Takeaway

- Plot **mix and segments** before trusting a single PSI number.
- PSI below 0.1 is not a clean bill of health if conditional accuracy on the online tail is wrong.
- Response options: importance weighting, explicit channel feature, or scheduled retrain—after segment eval, not on PSI alone."""
        ),
    ),
)

_register(
    "03_concept_drift.ipynb",
    notebook(
        md(
            """# 03 — Concept drift (when P(Y|X) changes)

On **day 70** the synthetic finance policy tightens: categories that auto-cleared now require review. Features look similar; **the correct label** changes. That is **concept drift**.

Watch **calibration** (ECE) and reviewer queue volume—not accuracy alone. In this generator, accuracy can move in a misleading direction while probabilities stop meaning what they did under the old policy."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import pandas as pd

from drift_lab import StreamConfig, build_ledger_route_model, generate_stream
from drift_lab.analysis import attach_predictions, outcome_summary
from drift_lab.metrics import rolling_accuracy
from drift_lab.viz import calibration_figure, rolling_accuracy_figure

cfg = StreamConfig()
model = build_ledger_route_model(cfg)
df = generate_stream("concept_abrupt", cfg)
scored = attach_predictions(model, df)

shift = cfg.concept_shift_day
pre = scored["day"] < shift
post = scored["day"] >= shift
print("Concept shift day:", shift)"""
        ),
        md("## Accuracy and ECE pre/post shock"),
        code(
            """summary = pd.DataFrame(
    {
        "window": ["pre-shift", "post-shift"],
        **{
            k: [
                outcome_summary(
                    scored.loc[pre, "label"],
                    scored.loc[pre, "pred_label"],
                    scored.loc[pre, "prob_review"],
                )[k],
                outcome_summary(
                    scored.loc[~pre, "label"],
                    scored.loc[~pre, "pred_label"],
                    scored.loc[~pre, "prob_review"],
                )[k],
            ]
            for k in ["accuracy", "ece", "positive_rate", "mean_score"]
        },
    }
)
summary.round(4)"""
        ),
        md("## Reliability diagrams"),
        code(
            """fig_pre, _ = calibration_figure(scored.loc[pre, "label"], scored.loc[pre, "prob_review"])
fig_pre.suptitle("Pre-shift calibration", y=1.02)
fig_post, _ = calibration_figure(scored.loc[~pre, "label"], scored.loc[~pre, "prob_review"])
fig_post.suptitle("Post-shift calibration", y=1.02)"""
        ),
        md("## Rolling accuracy with frozen weights"),
        code(
            """acc_df = rolling_accuracy(scored["label"].values, scored["pred_label"].values, window=800)
fig, roll_meta = rolling_accuracy_figure(
    acc_df,
    title="Rolling accuracy after label semantics change (weights frozen)",
)
roll_meta"""
        ),
        md(
            """## Takeaway

1. Version **`label_policy`** with every label row—treat policy releases like schema migrations.
2. Page on **calibration / queue depth** when accuracy still looks acceptable.
3. Retrain on **post-shock labels only**, or weight by policy version; do not mix incompatible outcomes in one loss."""
        ),
    ),
)

_register(
    "04_shift_vs_drift_noise.ipynb",
    notebook(
        md(
            """# 04 — Gradual drift, abrupt shift, and seasonal noise

Teams say "drift" for three different mechanisms:

| Mechanism | Example in LedgerRoute | Typical response |
|-----------|------------------------|------------------|
| Gradual covariate drift | Rising online channel share | Segment eval, scheduled retrain |
| Abrupt concept/policy shift | Label rule change on day 70 | Rollback, threshold freeze, human gate |
| Seasonal noise | Sinusoidal label-rate fluctuation | Longer windows, same-week-last-year baseline |

This notebook compares **detectors and runbooks**, not a single alarm type."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import numpy as np
import pandas as pd

from drift_lab import StreamConfig, generate_stream
from drift_lab.analysis import daily_channel_mix, daily_label_rate
from drift_lab.detectors import cusum_detect
from drift_lab.viz import shift_overlay_figure

cfg = StreamConfig()
cov = generate_stream("covariate_gradual", cfg)
concept = generate_stream("concept_abrupt", cfg)
noise = generate_stream("noise_only", cfg)


def normalize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)


gradual = normalize(daily_channel_mix(cov).values)
abrupt = normalize(daily_label_rate(concept).values)
seasonal = normalize(daily_label_rate(noise).values)"""
        ),
        md("## Overlay normalized daily signals"),
        code(
            """fig = shift_overlay_figure(
    {
        "Gradual (channel mix)": gradual,
        "Abrupt (label rate)": abrupt,
        "Seasonal noise": seasonal,
    }
)"""
        ),
        md("## CUSUM alarms (same threshold, different physics)"),
        code(
            """rows = []
for name, series in [
    ("gradual mix", gradual),
    ("abrupt label rate", abrupt),
    ("seasonal noise", seasonal),
]:
    alarm = cusum_detect(series, threshold=4.0)
    rows.append({"signal": name, "first_alarm_day": alarm})
pd.DataFrame(rows)"""
        ),
        md("## Suggested runbook mapping"),
        code(
            """pd.DataFrame(
    [
        ("Gradual covariate", "Segment accuracy + mix plot", "Scheduled retrain / reweight"),
        ("Abrupt concept", "ECE + queue volume", "Policy rollback, threshold freeze"),
        ("Seasonal noise", "Same ISO week last year", "Extend window; no retrain"),
    ],
    columns=["mechanism", "confirm with", "first response"],
)"""
        ),
        md(
            """## Takeaway

- **Do not share one on-call runbook** for all alarm types.
- Control **multiplicity** when testing many features hourly—digest sub-threshold moves.
- **Replay** a past week offline before retraining on noise."""
        ),
    ),
)

_register(
    "05_prediction_and_delayed_labels.ipynb",
    notebook(
        md(
            """# 05 — Prediction drift and delayed labels

When ground truth arrives days later, **score distributions** are legitimate early warnings. This notebook:

1. Compares predicted probabilities between an early and a late window on a covariate stream (frozen model).
2. Simulates **label latency** and shows how rolling outcome metrics appear only after joins catch up."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import pandas as pd

from drift_lab import StreamConfig, build_ledger_route_model, generate_stream
from drift_lab.analysis import attach_predictions, simulate_label_delay
from drift_lab.metrics import rolling_accuracy, two_sample_ks
from drift_lab.viz import prediction_hist_figure, rolling_accuracy_figure

cfg = StreamConfig()
model = build_ledger_route_model(cfg)
df = generate_stream("covariate_gradual", cfg)
scored = attach_predictions(model, df)

early = scored[(scored["day"] >= 30) & (scored["day"] < 50)]
late = scored[scored["day"] >= 90]
ks_stat, ks_p = two_sample_ks(early["prob_review"], late["prob_review"])
print(f"KS on scores: stat={ks_stat:.4f}, p={ks_p:.2e}")"""
        ),
        md("## Prediction distribution shift"),
        code(
            """fig, hist_meta = prediction_hist_figure(
    early["prob_review"].values,
    late["prob_review"].values,
)
hist_meta"""
        ),
        md("## Delayed label join (synthetic 4-day latency)"),
        code(
            """delayed = simulate_label_delay(scored, latency_days=4)
# Labels for transactions on day d become available on day d+4
joinable = delayed[delayed["label_available_day"] <= delayed["day"].max()]
print(f"Joinable rows: {len(joinable):,} / {len(delayed):,}")

acc_df = rolling_accuracy(
    joinable["label"].values,
    joinable["pred_label"].values,
    window=800,
)
fig, roll_meta = rolling_accuracy_figure(
    acc_df,
    title="Rolling accuracy after label join (4-day latency)",
)
roll_meta"""
        ),
        md("## Outcome accuracy: early vs late window"),
        code(
            """def window_acc(part):
    return (part["pred_label"] == part["label"]).mean()

print("Early days 30–49:", round(window_acc(early), 4))
print("Late days 90+:", round(window_acc(late), 4))"""
        ),
        md(
            """## Takeaway

- Log **`expense_id`, model_version, feature_schema, score, timestamp** on every prediction.
- Use score drift for **triage**, not automatic retrain, when labels lag.
- Pair quantile shift with a small **audit batch** or shadow threshold before paging."""
        ),
    ),
)

_register(
    "06_llm_embedding_surrogate.ipynb",
    notebook(
        md(
            """# 06 — Embedding drift without retraining the LLM

Vendor embedding APIs change when the provider ships a new encoder. You cannot "retrain" that model inside your account. Practitioners instead:

1. **Pin** the provider model ID in config.
2. Keep a **reference batch** of embeddings (hashed IDs, not necessarily raw text).
3. Track **cosine distance** or retrieval hit-rate degradation.
4. Refresh indexes / RAG corpora and run **eval gates** before traffic moves.

This notebook simulates a provider swap with normalized random vectors—no GPU LLM required."""
        ),
        code(
            """# From repo root: pip install -e ".[dev]"
%matplotlib inline

import numpy as np
import pandas as pd

from drift_lab.viz import embedding_distance_figure

rng = np.random.default_rng(7)
n_batches = 60
dim = 32
n_refs = 200

ref = rng.normal(size=(n_refs, dim))
ref /= np.linalg.norm(ref, axis=1, keepdims=True)

distances = []
for t in range(n_batches):
  # provider noise increases after batch 25 (synthetic "model ID bump")
    drift_scale = 0.02 * max(0, t - 25)
    batch = ref + rng.normal(scale=0.05 + drift_scale, size=ref.shape)
    batch /= np.linalg.norm(batch, axis=1, keepdims=True)
    cos_dist = 1.0 - (batch @ ref.T).max(axis=1)
    distances.append(float(cos_dist.mean()))

dist_arr = np.array(distances)
print("Batches:", len(dist_arr), "| dim:", dim)"""
        ),
        md("## Mean cosine distance to reference batch"),
        code(
            """fig, meta = embedding_distance_figure(dist_arr)
meta"""
        ),
        md("## When to act (example thresholds on surrogate distance)"),
        code(
            """THRESHOLD = 0.08
breach = int(np.argmax(dist_arr > THRESHOLD)) if (dist_arr > THRESHOLD).any() else None
print("First batch above", THRESHOLD, ":", breach)
print("Action: pin model ID in CI, refresh vector index, run pinned eval set before traffic switch")"""
        ),
        md("## Governance levers (tabular vs embedding vs generative)"),
        code(
            """pd.DataFrame(
    [
        ("Tabular head", "Retrain / reweight on fresh labels", "Rolling accuracy, PSI"),
        ("Embedding API", "Pin model ID; refresh vector index", "Reference-batch cosine distance"),
        ("Prompt + RAG", "Hash prompts; version corpus", "Pinned eval set regression"),
    ],
    columns=["layer", "maintenance action", "monitor"],
)"""
        ),
        md(
            """## Takeaway

Full weight retrain is one tool among many. Match the **maintenance action to the layer** that actually moved."""
        ),
    ),
)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, nb in ALL_NOTEBOOKS:
        path = out_dir / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"wrote {path.name} ({n_md} markdown, {n_code} code cells)")


if __name__ == "__main__":
    main()
