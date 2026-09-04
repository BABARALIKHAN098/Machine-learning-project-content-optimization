# Implementation Plan — Content Trend Classification

**Source specification:** `specs/SPEC-00-problem-definition.md`  
**Plan status:** Implemented — experimental model, performance gate not met  
**Owner:** Babar Ali Khan  
**Target:** `trend_direction`  
**Task:** Five-class classification (`down`, `stable`, `up`, `new`, `flat`)

## Implementation Result (2026-09-03)

- The data contract, reproducible audit, centralized leakage denylist, deterministic client-grouped splitting, train-only preprocessing, baselines, two candidate models, sealed-test evaluation, artifact packaging, batch inference, tests, CI, and model documentation are implemented.
- The random forest was selected using validation macro F1 and then evaluated on the untouched test partition.
- Test macro F1 is 0.4215, below the provisional 0.45 threshold; `down` recall is 0.6603.
- The artifact is therefore marked experimental and must not be used for production automation.
- The next substantive step is to create timestamped feature snapshots and future-window labels, then repeat grouped validation. Post-hoc tuning against the current test result is intentionally prohibited.

## 1. Objective

Build a reproducible batch machine-learning workflow that estimates a content item's next-period trend direction from information available at the prediction cutoff. The output will help content teams prioritize human review; it will not automatically modify or publish content.

The first release is successful when it:

1. prevents target, temporal, client, and preprocessing leakage;
2. beats most-frequent and stratified-random baselines on macro F1;
3. reaches the provisional macro-F1 threshold of 0.45 on a client-grouped holdout set;
4. reports recall for the `down` class and per-class errors;
5. scores 30,000 rows reproducibly on a standard CPU;
6. produces versioned model, preprocessor, configuration, metrics, and dataset-fingerprint artifacts.

## 2. Decisions Required Before Model Implementation

| Decision | Proposed choice | Approval condition |
| --- | --- | --- |
| Prediction target | `trend_direction` | Stakeholder confirms all five classes are operationally meaningful. |
| Prediction cutoff | End of the previous 30-day period | Every input must be known at this point. |
| Primary metric | Macro F1 | Approved because the target is imbalanced and minority classes matter. |
| Minimum performance | Macro F1 >= 0.45 | Retain as provisional until baseline and label quality are reviewed. |
| Validation unit | `client_id` group | No client may occur in more than one data partition. |
| Decision use | Human review prioritization | Predictions must not trigger automatic content changes. |

If the supplied CSV is a single current snapshot rather than a time-shifted training table, the model can demonstrate association but not true forward-looking performance. Before production use, generate labels from a later outcome window and retain feature values as they existed at the cutoff.

## 3. Feature Availability and Leakage Contract

### 3.1 Allowed identifiers

- Retain `content_id` only for joining predictions back to source records.
- Retain `client_id` only for grouped splitting, controlled reporting, and authorized joins.
- Never pass either identifier into the feature transformer or estimator.

### 3.2 Candidate model features

Use only fields available by the prediction cutoff:

- Search attributes: `search_volume`, `competition`, `competition_level`, `cpc`.
- Content attributes: `content_type`, `main_intent`, `word_count`, `char_count`.
- Generation metadata: `provider_used`, `model_used`, subject to missingness review.
- Previous-period behavior: `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`.
- Age and freshness: `content_age_days`, `age_tier`, `age_tier_order`, `days_since_last_update`, `freshness_tier`.
- Length bands: `word_count_tier`, `char_count_tier`.

### 3.3 Fields excluded from training

Exclude outcome-window values and their derivatives:

- all `*_last_30d` fields;
- all `*_90d` fields when the 90-day window overlaps the outcome period;
- `days_with_impressions` and `days_with_sessions` when calculated through the outcome date;
- `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, and `ai_traffic_pct` when calculated through the outcome date;
- `impression_tier`, `position_tier`, and `trend_pct`;
- the target `trend_direction` itself;
- `content_id` and `client_id` as estimator inputs.

Implement one centralized feature-selection function and assert this exclusion contract in tests. Do not rely only on configuration comments.

## 4. Delivery Phases

### Phase 0 — Approve the problem contract

**Files:** `specs/SPEC-00-problem-definition.md`, this plan.

Tasks:

- Confirm the intended user, decision, target classes, and prediction cutoff.
- Confirm whether `new` is a predictable future class or a lifecycle state that should be handled with a rule.
- Confirm whether `flat` and `stable` are meaningfully distinct for the business decision.
- Agree on the cost tradeoff between false positives and false negatives for `down`.
- Approve macro F1 as the selection metric and `down` recall as a guardrail.
- Record the stakeholder and approval date in SPEC-00.

Exit criteria:

- All unchecked stakeholder items in SPEC-00 are resolved or explicitly deferred.
- Target semantics and prediction timing have written approval.

### Phase 1 — Reproducible ingestion and dataset audit

**Files:** `configs/data.yaml`, `src/machine_learning_project/data/ingestion.py`, `src/machine_learning_project/data/validation.py`, `src/machine_learning_project/data/profiling.py`, `scripts/validate_data.py`, `reports/`.

Tasks:

- Load `data/raw/dataset.csv` without modifying it.
- Calculate and store its SHA-256 fingerprint.
- Validate the exact 44-column schema, target existence, target values, and identifier uniqueness.
- Confirm 30,000 rows, 30,000 unique `content_id` values, and 32 clients for the current source version.
- Profile data types, missingness, cardinality, duplicate rows, invalid ranges, and constant fields.
- Check impossible values, including negative counts, rates outside their valid ranges, and inconsistent tier/value pairs.
- Review `provider_used`, whose observed missingness exceeds the configured 60% threshold; either drop it with rationale or explicitly approve an exception.
- Save a machine-readable profile and a human-readable audit report under `reports/`.

Tests:

- Valid source loads with the expected shape and fingerprint.
- Missing, empty, malformed, or schema-incompatible sources fail with actionable messages.
- Unknown target labels fail validation.
- Duplicate `content_id` values fail validation.
- The raw source hash is unchanged after the pipeline runs.

Exit criteria:

- Every column has one unambiguous role: target, ID/group, feature, or excluded.
- Data-quality exceptions are documented and approved.

### Phase 2 — Leakage-safe partitioning

**Files:** `configs/training.yaml`, `src/machine_learning_project/data/splitting.py`, `tests/unit/test_splitting.py`, `specs/SPEC-03-data-splitting-and-leakage-control.md`.

Tasks:

- Split by `client_id` using deterministic group-aware logic.
- Reserve approximately 20% of clients for final test and 20% of the remaining development data for validation.
- Optimize group assignment to preserve usable class coverage without allowing group overlap.
- Store row counts, client counts, class distributions, and split seed in metadata.
- If any class is absent from a partition, stop and revise the split strategy rather than silently continuing.
- Keep the test partition sealed until the final candidate is selected.

Tests:

- Train, validation, and test client sets are pairwise disjoint.
- Repeated runs with the same seed produce identical assignments.
- Every source row appears in exactly one partition.
- Target and identifiers never appear in the feature matrix.

Exit criteria:

- Split manifest is reproducible and passes all leakage assertions.

### Phase 3 — Preprocessing pipeline

**Files:** `configs/preprocessing.yaml`, `src/machine_learning_project/features/preprocessing.py`, `src/machine_learning_project/features/engineering.py`, `tests/unit/test_preprocessing.py`.

Tasks:

- Parse configured numeric fields as numeric and report coercion failures.
- Median-impute numeric fields using values learned from training data only.
- Impute categorical missing values with `__MISSING__`.
- One-hot encode categoricals with unknown-category handling enabled.
- Scale numeric features only when required by the estimator.
- Remove redundant representations if experiments show instability; for example, retain either an exact age value or its tier when appropriate.
- Fit transformers only on the training partition and serialize the fitted pipeline with the model.
- Expose transformed feature names for model inspection and debugging.

Tests:

- No unexpected nulls remain after transformation.
- An unseen category at inference does not fail.
- Input data is not mutated.
- Excluded and ID columns cannot enter transformed output.
- Validation/test statistics do not affect fitted imputers or encoders.

Exit criteria:

- A single fitted preprocessing object can transform training and unseen inference records consistently.

### Phase 4 — Baselines and feasibility gate

**Files:** `src/machine_learning_project/models/baseline.py`, `pipelines/training_pipeline.py`, `scripts/train_model.py`, `reports/metrics/`.

Tasks:

- Train a most-frequent baseline.
- Train a stratified-random baseline with a fixed seed.
- Optionally add a transparent rule baseline based only on previous-period signals available at cutoff.
- Evaluate baselines on the validation partition using macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1, and confusion matrices.
- Record training and batch-inference duration.
- Review whether the feature set has signal beyond class prevalence.

Exit criteria:

- Baseline metrics and artifacts are reproducible.
- The team decides whether the 0.45 macro-F1 threshold is realistic before tuning complex models.

### Phase 5 — Candidate model training

**Files:** `configs/training.yaml`, `src/machine_learning_project/models/train.py`, `src/machine_learning_project/models/tune.py`, `pipelines/training_pipeline.py`.

Tasks:

- Start with multinomial logistic regression as an interpretable linear candidate.
- Add a tree-based candidate suitable for mixed tabular data, such as histogram gradient boosting or random forest.
- Use class weighting where supported; compare it with unweighted training.
- Tune only a small, declared search space using group-aware cross-validation on development data.
- Select the candidate by validation macro F1, subject to the `down`-recall guardrail and runtime constraint.
- Fix all random seeds and record library versions, configuration, feature list, and source fingerprint.
- Do not use final test results for model or threshold selection.

Tests:

- Training is deterministic within documented numerical tolerances.
- Unsupported task types or target labels fail clearly.
- Serialized and in-memory models return equivalent predictions.
- Training metadata includes configuration, feature contract, fingerprint, and versions.

Exit criteria:

- One candidate is selected without consulting the final holdout result.

### Phase 6 — Final evaluation and error analysis

**Files:** `src/machine_learning_project/models/evaluate.py`, `scripts/evaluate_model.py`, `reports/metrics/`, `reports/figures/`, `specs/SPEC-07-evaluation-and-error-analysis.md`.

Tasks:

- Run the selected frozen pipeline once on the client-grouped test set.
- Report macro F1 as the primary metric.
- Report weighted F1, balanced accuracy, per-class precision/recall/F1, and confusion matrix.
- Highlight recall and false-negative count for `down`.
- Break down performance by client, content type, intent, age tier, freshness tier, and missingness cohort where sample size permits.
- Report confidence intervals using group-aware bootstrap resampling by client.
- Inspect common confusions, especially `flat` versus `stable` and `new` versus other classes.
- Compare runtime with the 30,000-row CPU constraint.
- Document limitations and whether the provisional 0.45 threshold was met.

Exit criteria:

- Results are reproducible from a single command.
- Failures and subgroup weaknesses are documented even if the headline threshold is met.

### Phase 7 — Packaging and batch inference

**Files:** `src/machine_learning_project/inference/predictor.py`, `src/machine_learning_project/inference/validation.py`, `src/machine_learning_project/inference/schemas.py`, `pipelines/inference_pipeline.py`, `artifacts/`.

Tasks:

- Package preprocessing and the estimator as one versioned artifact.
- Store metadata containing model version, training time, source hash, feature schema, class order, metrics, and dependency versions.
- Validate required inputs, data types, ranges, and unknown fields before scoring.
- Return `content_id`, predicted class, per-class probabilities when supported, model version, and warnings.
- Keep `client_id` out of feature computation and restrict it to authorized joins/reporting.
- Produce a ranked review queue using predicted probability of `down`; clearly label this as prioritization support, not an automated action.
- Benchmark scoring 30,000 rows on CPU.

Tests:

- Batch and single-row predictions obey the same schema.
- Missing required fields and malformed values return actionable errors.
- Output probabilities sum to one within tolerance.
- Input row order and `content_id` association are preserved.
- The 30,000-row benchmark satisfies the approved runtime limit.

Exit criteria:

- A clean environment can load the artifact and reproduce predictions without refitting.

### Phase 8 — Quality gates and documentation

**Files:** `tests/`, `.github/workflows/`, `README.md`, relevant `specs/` documents.

Tasks:

- Run formatting, linting, type checking, unit tests, integration tests, and contract tests in CI.
- Add an end-to-end smoke test using a small non-sensitive fixture.
- Document setup, validation, training, evaluation, and inference commands.
- Add a model card describing intended use, prohibited use, data, performance, limitations, and ethical considerations.
- Record reproducibility instructions and artifact locations.
- Ensure logs do not expose raw identifiers or row contents unnecessarily.

Exit criteria:

- CI passes from a clean checkout.
- Another developer can reproduce validation, training, evaluation, and inference from the README.

## 5. File-Level Work Breakdown

| Area | Planned change |
| --- | --- |
| `configs/data.yaml` | Finalize column roles, leakage exclusions, allowed target labels, range checks, and missingness decisions. |
| `configs/preprocessing.yaml` | Define imputers, categorical handling, scaling behavior, and feature-contract version. |
| `configs/training.yaml` | Set macro F1, baselines, grouped split sizes, model candidates, tuning bounds, and seeds. |
| `data/ingestion.py` | Deterministic loading and SHA-256 fingerprinting. |
| `data/validation.py` | Schema, domain, target, uniqueness, missingness, and leakage validation. |
| `data/profiling.py` | Reproducible profile and quality summaries. |
| `data/splitting.py` | Client-grouped deterministic train/validation/test assignment. |
| `features/preprocessing.py` | Train-only imputation, encoding, scaling, and feature-name output. |
| `features/engineering.py` | Cutoff-safe engineered features only. |
| `models/baseline.py` | Most-frequent, stratified, and optional rule baselines. |
| `models/train.py` | Candidate fitting and reproducibility metadata. |
| `models/tune.py` | Bounded group-aware model selection. |
| `models/evaluate.py` | Global, per-class, grouped, and runtime evaluation. |
| `inference/` | Input contract, artifact loading, predictions, probabilities, and warnings. |
| `pipelines/` | Reproducible orchestration for data, training, and inference flows. |
| `tests/` | Unit, integration, leakage, reproducibility, and prediction-contract coverage. |
| `reports/` | Dataset audit, metrics, figures, error analysis, and model card. |
| `artifacts/` | Versioned model, transformer, metadata, split manifest, and feature schema. |

## 6. Requirements Traceability

| SPEC-00 requirement | Planned verification |
| --- | --- |
| Predict `trend_direction` | Schema validation and five-class prediction contract tests. |
| Support refresh prioritization | Ranked `down`-probability batch output reviewed by a human. |
| Macro F1 is primary | Training config and evaluation report assert macro F1 as selection metric. |
| Macro F1 >= 0.45 | Final grouped-holdout evaluation; threshold remains provisional until approved. |
| Beat simple baselines | Side-by-side baseline and candidate report on the same partitions. |
| Prevent temporal leakage | Central denylist, cutoff contract, and automated leakage tests. |
| Prevent client leakage | Pairwise-disjoint `client_id` assertions across partitions. |
| Protect identifiers | IDs excluded from features and limited in logs/API outputs. |
| Report false negatives | `down` recall and false-negative counts in evaluation report. |
| CPU batch constraint | Repeatable 30,000-row benchmark with environment metadata. |
| Human-in-the-loop limitation | Model card and inference output warning prohibit automatic publishing changes. |

## 7. Recommended Implementation Order

1. Approve unresolved SPEC-00 decisions.
2. Complete the dataset audit and column-role review.
3. Implement and test the centralized leakage contract.
4. Implement deterministic client-grouped splitting.
5. Build the train-only preprocessing pipeline.
6. Establish and review baselines.
7. Train and select a small set of candidate models.
8. Freeze the candidate and evaluate once on the test set.
9. Complete error analysis and decide whether the model is fit for intended use.
10. Package inference, benchmark runtime, and finish documentation/CI.

## 8. Definition of Done

Implementation is complete only when:

- stakeholder approval is recorded in SPEC-00;
- the raw dataset fingerprint and audit are stored;
- every field has a documented role and leakage status;
- client-grouped partitions are reproducible and disjoint;
- baselines and the selected model are evaluated on identical partitions;
- final test macro F1 and `down` recall are reported without test-driven tuning;
- the model, preprocessing, metadata, and feature schema load together;
- 30,000-row CPU scoring meets the approved runtime constraint;
- unit, integration, contract, and leakage tests pass in CI;
- the README and model card document intended use, limitations, and reproducible commands.

## 9. Known Open Items

- Stakeholder approval of the target, prediction cutoff, and 0.45 threshold.
- Exact operational batch-runtime limit in seconds or minutes.
- Minimum acceptable recall for the `down` class.
- Confirmation that the 90-day aggregates overlap the target outcome window.
- Decision on whether `provider_used` should be dropped because of high missingness.
- Decision on whether `new`, `flat`, and `stable` should remain separate model classes.
- Availability of timestamped snapshots needed for a true prospective evaluation.
