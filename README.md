# data_drift

Runnable **LedgerRoute** playground for production drift monitoring: synthetic expense-routing streams, PSI/KS/ECE metrics, changepoint detectors, and six notebooks you can run top-to-bottom.

Companion to [The living model](https://arraxis.com/living-model/) on Arraxis. The library is `drift_lab`; notebooks show how practitioners wire monitoring layers without waiting on real production data.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"
pytest
jupyter lab notebooks/
```

Requires Python 3.10+. Core deps: NumPy, pandas, SciPy, scikit-learn, Matplotlib.

## What you get

| Piece | Purpose |
|-------|---------|
| `src/drift_lab/streams.py` | Reproducible stable / covariate / concept / noise streams |
| `src/drift_lab/metrics.py` | PSI, KS, rolling accuracy, ECE |
| `src/drift_lab/detectors.py` | CUSUM, windowed PSI series |
| `src/drift_lab/analysis.py` | `attach_predictions`, segment accuracy, delayed-label join |
| `src/drift_lab/viz.py` | Inline Matplotlib figures for notebooks |
| `notebooks/*.ipynb` | Practitioner walkthroughs (see below) |
| `docs/taxonomy.md` | Covariate vs concept vs prior; drift vs noise |
| `docs/open_questions.md` | FAQ with concrete answers |

## Notebooks

Run in order or jump to the failure mode you care about.

| Notebook | You will |
|----------|----------|
| `01_baseline_stream.ipynb` | Define the reference training window and hold-out sanity check |
| `02_covariate_drift.ipynb` | Plot channel mix, PSI/KS, segment accuracy under P(X) shift |
| `03_concept_drift.ipynb` | Separate accuracy from calibration after a policy shock |
| `04_shift_vs_drift_noise.ipynb` | Compare gradual drift, abrupt shift, and seasonal noise + CUSUM |
| `05_prediction_and_delayed_labels.ipynb` | Score-distribution drift and 4-day label latency |
| `06_llm_embedding_surrogate.ipynb` | Embedding distance monitor without LLM retrain |

Notebooks are maintained via `scripts/write_practitioner_notebooks.py` when library APIs change.

## Minimal API example

```python
from drift_lab import StreamConfig, build_ledger_route_model, generate_stream
from drift_lab.analysis import attach_predictions, compare_windows

cfg = StreamConfig()
model = build_ledger_route_model(cfg)
cov = generate_stream("covariate_gradual", cfg)
scored = attach_predictions(model, cov)

ref = cov[cov["day"] < 30]
late = cov[cov["day"] >= 90]
print(compare_windows(ref, late, "channel_online"))
```

## Site figure export (maintainers)

PNG assets under `artifacts/story_assets/` feed arraxis.com story pages. Regenerate after changing plots or stream defaults:

```bash
python -m drift_lab.export_figures
# or: export-story-figures
```

This is optional for notebook users.

## License

MIT
