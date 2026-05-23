# Open questions

Community-facing list mirrored on the Arraxis saga. Answers belong in issues and PRs on this repo.

1. **Shift vs drift** — Should abrupt changepoints use the same detectors and runbooks as gradual covariate drift?
2. **Noise vs drift** — How do you set window length and control false alarms when monitoring dozens of features?
3. **Feature drift without metric drop** — Do you still retrain, reweight, or only log?
4. **Concept drift with stable marginals** — When \(P(X)\) looks fine but \(P(Y \mid X)\) moved, which features or labels do you add?
5. **Delayed labels** — How long can prediction-drift proxies stand in for ground truth?
6. **LLM stack** — Separate monitors for weights, embedding API version, prompt hash, and RAG corpus age?
7. **When to pause** — Shadow mode, automatic rollback, or human gate: what triggers each?
8. **Adversarial drift** — Fraud and abuse adapt faster than monthly retrains; what is the minimum response time?
9. **Prior vs concept** — Class balance shifts without semantic change: threshold tuning or full retrain?
10. **Unstructured data** — Images, text, graphs: what is the reference window when you cannot store raw production data?

Send evidence (link, metric, or minimal repro) via a GitHub issue on this repository.
