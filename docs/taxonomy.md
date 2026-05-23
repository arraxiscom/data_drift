# Drift and shift taxonomy

Working definitions used in this repository and on the Arraxis **living model** saga. Terminology overlaps in the literature; we pick one map and stick to it.

## Dataset shift (umbrella)

A deployed model was trained on distribution \(P_{\text{train}}(X, Y)\). Production draws from \(P_{\text{prod}}(X, Y)\). When the two differ, we are under **dataset shift** (see Quiñonero-Candela et al., *Dataset Shift in Machine Learning*, MIT Press).

Common factors:

| Name | What changes | Model symptom |
|------|----------------|---------------|
| **Covariate shift** (feature / data drift) | \(P(X)\) | Good on old regions of \(X\), weak on new regions |
| **Prior shift** | \(P(Y)\) | Class rates change; thresholds may be wrong |
| **Concept shift** (concept drift) | \(P(Y \mid X)\) | Same \(X\) can imply a different correct label |

## Drift vs shift vs noise (operational)

- **Gradual drift:** slow change across weeks; windowed PSI/KS and rolling performance often suffice.
- **Abrupt shift / changepoint:** regime break (policy change, fraud ring adaptation, macro shock); needs faster alarms and explicit rollback paths.
- **Sampling noise:** finite batches fluctuate even when \(P(X,Y)\) is stable; short windows and many simultaneous tests create false alarms.

## Monitoring signals (2025–2026 practice)

1. **Input features** — cheap, can be misleading alone (covariate shift without immediate label impact).
2. **Predictions** — distribution of scores changes when upstream or semantics move; useful when labels are delayed.
3. **Outcomes vs delayed labels** — ground truth joins days or weeks later; lowest false-positive rate but slow.
4. **Calibration** — probability outputs can drift before accuracy moves; see recent anytime-valid calibration monitors (e.g. PITMonitor, arXiv:2603.13156).

## Modalities

| Modality | Typical reference | Retrain? |
|----------|-------------------|----------|
| Tabular sklearn / GBDT | Feature batch from training window | Often yes, on schedule or alarm |
| Embeddings from an API | Version-pinned model + stored reference vectors | Refresh index / RAG, not full LLM |
| Generative LLM | Prompt version, tool schema, eval set | Guardrails, routing, eval gates; full weight updates are rare |

## References

- Quiñonero-Candela, Sugiyama, Schwaighofer, Lawrence (eds.), *Dataset Shift in Machine Learning*.
- Moreno-Torres et al., "A unifying view on dataset shift in classification" ([PDF](https://www.uco.es/~inforjos/docs/dataset_shift_JMLR.pdf)).
- Evidently AI, "Machine Learning Monitoring, Part 5: Data and Concept Drift" ([blog](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)).
- Gama et al., "A Survey on Concept Drift Adaptation" ([ACM](https://doi.org/10.1145/2699984)).
