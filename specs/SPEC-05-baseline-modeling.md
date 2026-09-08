# SPEC-05 - Baseline Modeling

**Status:** Implemented; verification recorded in `reports/baselines/verification.json`.
**Owner:** Babar Ali Khan
**Protocol adopted:** 2026-09-08, under the request to implement the implementation plan.

## Purpose and scope

Produce a reproducible development benchmark for five-class trend_direction classification.
Fit most-frequent and stratified DummyClassifier baselines using training labels only;
evaluate the existing verified validation partition. No feature engineering, candidate search,
refit on validation, test evaluation, deployment or model promotion belongs to this command.
Full-source schema and split identity checks occur upstream; no test dataframe reaches the runner.

## Adopted protocol

Contract and metric versions are 1.0. The canonical seed is 42 and the fixed repeat schedule
is [42, 43, 44, 45, 46]; arbitrary seed searches are rejected. Fit one majority estimator and
five independent stratified estimators. Predict once per estimator on the complete validation
batch, using neutral arrays. Sort development rows by stable row key before prediction.
Training string-label ties resolve to the lexicographically first tied label.

Both main partitions must contain every configured label, including down. Reject empty,
null, unknown or mismatched labels. Explicit ordered labels govern macro/weighted F1,
per-class results and confusion matrices, with zero_division=0. Balanced accuracy averages
over supported truth classes in metric-only fixtures; macro F1 always retains configured labels.

Material improvement requires at least +0.01 absolute macro F1 over majority, seed-42
stratified and the fixed-repeat stratified mean. Report strict superiority over both canonical
baselines, material improvement, down recall >=0.50 and macro F1 >=0.45 independently.
A weak score is a valid result, not a command error. Candidate ranking and recall fallback
are explicit development statuses, never promotion decisions.

Compare only versioned evidence with identical source, split, ordered row fingerprints,
counts, target, label order, partition and metric contract. Reject legacy or incompatible
reports. Frozen artifacts require verified payload hashes and safe local paths. The manifest
is published last. No raw identifiers or row-level predictions are published.

## Requirements

| Requirement | Behavior | Verification |
| --- | --- | --- |
| SPEC-05-REQ-001 | Validate strategies, versions, reference seed, repeat seeds and comparison settings before fitting. | BASE-T-001/002 |
| SPEC-05-REQ-002 | Learn majority class and class frequencies exclusively from training labels. | BASE-T-003/004/009 |
| SPEC-05-REQ-003 | Make dummy predictions independent of feature values and unavailable outcomes. | BASE-T-005/006 |
| SPEC-05-REQ-004 | Reuse verified, disjoint development partitions; expose no test rows to the runner. | BASE-T-007/008 |
| SPEC-05-REQ-005 | Evaluate all baselines on identical ordered validation rows and a fixed five-label metric contract. | BASE-T-010/011/012 |
| SPEC-05-REQ-006 | Reproduce the canonical seeded baseline and report predetermined random-seed variation without choosing a favorable seed. | BASE-T-013/014 |
| SPEC-05-REQ-007 | Report fit/prediction timings separately from semantic benchmark scores. | BASE-T-015 |
| SPEC-05-REQ-008 | Store aggregate results, training priors, parameters and source/split/config/code provenance. | BASE-T-016/017 |
| SPEC-05-REQ-009 | Compare candidates only against compatible evidence and report separate improvement, recall and project-target flags. | BASE-T-018/019/020 |
| SPEC-05-REQ-010 | Share baseline execution between the standalone CLI and training pipeline. | BASE-T-021 |
| SPEC-05-REQ-011 | Preserve source/artifact integrity, row identity and privacy during generation and reuse. | BASE-T-008/017/022 |
| SPEC-05-REQ-012 | Document limitations and provide a frozen benchmark reference for SPEC-06. | BASE-T-023 |

## Tests

| Test ID | Given/when | Expected result and location |
| --- | --- | --- |
| BASE-T-001 | Valid strategy/version/seed settings resolve | Exactly the prescribed baselines and repeat schedule; `tests/unit/test_baselines.py`. |
| BASE-T-002 | Unknown/duplicate strategies, empty seeds, missing reference, bool/invalid seeds, unsupported versions, invalid margin or test partition | Actionable error before fitting; baseline/config unit tests. |
| BASE-T-003 | Known training class distribution, including a tied majority | Correct learned majority and explicit deterministic tie behavior. |
| BASE-T-004 | Validation labels have a different majority/distribution | Training priors and majority remain unchanged; validation changes scores only. |
| BASE-T-005 | Different raw feature values with the same labels and row identities | Baseline predictions are unchanged; neutral arrays convey row counts only. |
| BASE-T-006 | Candidate/engineering builders instrumented to fail | Standalone baseline execution succeeds without invoking them. |
| BASE-T-007 | Verified grouped partitions; test access paths instrumented | Only train/validation reach the runner; identities/groups remain disjoint. |
| BASE-T-008 | Missing or changed source/split/assignment artifacts | Fail before fitting/publication; do not regenerate split artifacts; integration tests. |
| BASE-T-009 | Empty/null/unknown labels or a missing main-partition class | Explicit precondition failure without silently dropping rows/classes. |
| BASE-T-010 | Majority predictions with known validation prevalence q | Accuracy, balanced accuracy, macro/weighted F1, down recall and matrix match analytic expectations; new `tests/unit/test_evaluation.py`. |
| BASE-T-011 | Classes absent from predictions, and metric-only fixtures with a missing truth class | Fixed-label macro/per-class/matrix behavior and documented balanced-accuracy behavior. |
| BASE-T-012 | Unequal prediction/truth lengths, unknown predictions, bad label list | Fail before computing or publishing misleading metrics. |
| BASE-T-013 | Same seed, canonical rows and environment twice | Identical stratified predictions and metrics; no global RNG dependence. |
| BASE-T-014 | Fixed repeats with known metric values | Correct reference and mean/std/min/max; no best-seed selection or requirement of exact sampled proportions. |
| BASE-T-015 | Timing instrumentation around fit, predict and metric computation | Fit/predict costs separated; timing cannot alter decisions or semantic identity. |
| BASE-T-016 | Valid benchmark publication | Required versions, labels, priors, source/split/config/code/environment and payload hashes exist. |
| BASE-T-017 | Sensitive identifiers and seeded predictions | Public outputs contain aggregates and safe fingerprints only; no raw identifiers/prediction rows. |
| BASE-T-018 | Candidates at/below/above improvement, recall and project-target boundaries | Independent comparison flags follow exact rules, including equality at the material-improvement threshold. |
| BASE-T-019 | Candidate beats one baseline but not the other, or misses recall despite higher F1 | Report the failure explicitly; do not equate candidate ranking with readiness. |
| BASE-T-020 | Candidate and baseline have different source, split, row order/count, partition or metric version | Comparison rejected, including legacy reports lacking sufficient provenance. |
| BASE-T-021 | Standalone and integrated execution with the same inputs | Canonical metrics match; old `baselines_validation` consumers remain compatible. |
| BASE-T-022 | Two real runs and an interrupted/failed write | Source/split/model bytes unchanged; normalized outputs reproduce; incomplete payloads cannot be accepted as a valid manifest. |
| BASE-T-023 | Completed report and SPEC-06 handoff review | Development/seed-variation/previous-holdout/timing limitations and benchmark reuse instructions are explicit. |

## Verification and handoff

Run `python scripts/verify_baselines.py` to run the focused/full tests, Ruff and two isolated
real-data benchmark runs, verify artifact integrity, compare semantic results and check
source/split/model/report immutability. Exact commands and outcomes are saved in
`reports/baselines/verification.json`. Semantic equality excludes runtime measurements and
hashes of timing-bearing files; each run still passes its own complete payload hash checks.

SPEC-06 consumes `reports/baselines/baseline_manifest.json` using `load_benchmark`, or
passes it to `scripts/train_model.py --baseline-manifest`. Training records its manifest hash
and evaluates candidates on the same canonical validation population. Feature provenance is
recorded separately; different candidate features need not equal the feature-free baseline.

## Limitations and acceptance

Validation has already informed feature confirmation, and historical test evaluation exists.
This is a development benchmark, not an independent generalization estimate. Seed variation
measures classifier randomness on fixed data, not statistical significance or future-client
confidence. Dummy fit/predict timing excludes metrics and says nothing about a full model's
30,000-row runtime. External historical cutoff evidence for learned candidates remains unresolved.

Completion requires passing focused/full tests and lint, two-run reproducibility, source and
artifact immutability, private aggregate outputs, and the published frozen manifest plus verification.
Candidate target achievement is separate. No normative decisions remain open for this protocol.
