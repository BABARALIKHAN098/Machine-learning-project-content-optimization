# Model Card — Content Trend Classifier

## Status

Experimental. The leakage-safe grouped-holdout macro F1 is below the provisional acceptance threshold, so the artifact is not approved for production automation.

## Intended Use

Rank anonymized published content for human review based on predicted probability of a downward trend. SEO specialists and content strategists may use the ranking as one input to refresh planning.

## Prohibited Use

- Do not automatically edit, unpublish, or republish content.
- Do not use the output to evaluate individuals or infer sensitive traits.
- Do not treat predicted trends as proof that a refresh will cause improvement.
- Do not use the experimental artifact on schemas or time windows that differ from the documented feature contract.

## Data and Validation

- Training source: 30,000 anonymized content rows across 32 clients.
- Target: `trend_direction` (`down`, `stable`, `up`, `new`, `flat`).
- Validation strategy: deterministic client-grouped train/validation/test partitions.
- Identifiers: `content_id` and `client_id` are excluded from estimator inputs.
- Temporal leakage controls: current 30-day, overlapping 90-day, target-derived rates/tiers, and `trend_pct` are excluded.
- `provider_used` is excluded because 71.46% of values are missing.

## Results

The authoritative machine-readable values are in `reports/metrics/model_metrics.json`.

- Selected model: random forest.
- Test macro F1: 0.4215.
- Test recall for `down`: 0.6603.
- Provisional macro-F1 threshold: 0.45 (not met).

## Limitations

The data is a single retrospective snapshot. The labels describe observed trends and do not demonstrate that model-directed refreshes improve outcomes. A production decision requires timestamped feature snapshots, labels from a later outcome window, and a new prospective grouped evaluation. Minority classes, particularly `flat` and `new`, have less support and should be examined in the detailed metrics and confusion matrix.

## Human Oversight

Every prediction carries a human-review warning. Reviewers should consider business importance, content quality, seasonality, and editorial context before acting.
