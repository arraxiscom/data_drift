# Drift FAQ (answered)

Short answers for practitioners. The [living model saga](https://arraxis.com/living-model/) walks through LedgerRoute examples; notebooks in this repo reproduce the figures.

1. **Shift vs drift** — Use gradual runbooks (segment eval, scheduled retrain) for slow covariate change. Use abrupt runbooks (policy rollback, threshold freeze, human gate) for changepoints on labels or queue volume. Same tooling can feed both; do not share one on-call playbook.

2. **Noise vs drift** — Set minimum monitoring windows to one business cycle (e.g. 28 days for monthly seasonality). Control multiplicity when testing many features; route sub-threshold moves to a digest, not pages.

3. **Feature drift without metric drop** — Log and segment first. Retrain or reweight only when conditional accuracy on the growing slice breaches your bar, not when PSI alone crosses 0.1.

4. **Concept drift with stable marginals** — Monitor calibration and queue depth by `label_policy` version. Add post-shock labels; do not mix policies in one training loss without weights.

5. **Delayed labels** — Prediction-drift and feature layers are legitimate triage while labels lag; pair quantile shift with a small audit batch or shadow threshold before auto-retrain. Rolling outcome metrics need a join key on every score row.

6. **LLM stack** — Separate series for weights (rare changes), embedding API model ID, prompt hash, and RAG corpus age. A green tabular PSI does not authorize a prompt deploy.

7. **When to pause** — Shadow when blast radius is unknown; automatic rollback only where harm is bounded; human sign-off for customer-facing payouts and compliance queues.

8. **Adversarial drift** — Minimum response is rules plus analyst loop at hours/days; batch retrain on labeled fraud weeks supplements, it does not replace.

9. **Prior vs concept** — Balance-only change: thresholds and class weights. Semantic label change: new policy version, calibration, and post-shock training data.

10. **Unstructured data** — Use hashed reference batches, embedding-distance surrogates, and periodic labeled audits when raw production storage is blocked.

Evidence and reproduction: open a GitHub issue with a link, metric export, or minimal notebook run.
