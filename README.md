# data_drift

Runnable examples for [The living model](https://arraxis.com/living-model/) on Arraxis: synthetic expense-routing traffic (**LedgerRoute**), drift metrics, and notebooks that reproduce the saga figures.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"
pytest
python scripts/export_story_figures.py
```

After `export_story_figures.py`, plots and metric sidecars are under `artifacts/story_assets/`.

## Notebooks

| File | Topic |
|------|--------|
| `notebooks/01_baseline_stream.ipynb` | Reference model on a stable window |
| `notebooks/02_covariate_drift.ipynb` | Channel mix, PSI |
| `notebooks/03_concept_drift.ipynb` | Change in P(Y given X) |
| `notebooks/04_shift_vs_drift_noise.ipynb` | Gradual vs abrupt vs seasonal noise |
| `notebooks/05_prediction_and_delayed_labels.ipynb` | Score distribution drift |
| `notebooks/06_llm_embedding_surrogate.ipynb` | Embedding distance without LLM retrain |

## Docs

- [taxonomy.md](docs/taxonomy.md) — covariate, prior, and concept shift; drift vs noise
- [open_questions.md](docs/open_questions.md) — monitoring questions (contributions via GitHub issues)

## License

MIT
