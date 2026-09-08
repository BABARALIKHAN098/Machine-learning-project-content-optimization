# Implementation Plan - Baseline Modeling

**Source:** `specs/SPEC-05-baseline-modeling.md`

**Status:** Implemented and verified on 2026-09-08 under the subsequent implementation request

**Prepared:** 2026-09-07

**Owner:** Babar Ali Khan

**Outcome:** A simple, reproducible performance benchmark for five-class `trend_direction` classification.

## 1. Specification status and objective

SPEC-05 is a draft: its scope, requirements, tests, and decisions currently contain `TBD`. This document proposes a concrete contract based on SPEC-00 and the current SPEC-03/04 implementation. Proposed defaults are not approved requirements or completed implementation. Creating this plan does not change the specification, train models, regenerate reports, or alter existing code.

Establish how well a model can perform using only the training class distribution. Produce a standalone benchmark that later candidate models must exceed on the same evaluation rows, with the same labels and metric definitions. A completed benchmark may demonstrate that a candidate is inadequate; high baseline performance is not itself an implementation success criterion.

The authoritative baselines in SPEC-00 are the most-frequent-class classifier and stratified random classifier. Retain those two. Use the existing client-grouped development partitions, with fitting on train and evaluation on validation. Final-test evaluation remains outside this phase.

## 2. Current-state assessment

Package-relative paths below are under `src/machine_learning_project/`.

| Area | Observed implementation | Gap to address |
| --- | --- | --- |
| `models/baseline.py` | `build_baselines(seed)` constructs `DummyClassifier` instances for `most_frequent` and `stratified`. | Strategies are hard-coded; no configuration validation or standalone benchmark contract. |
| `configs/training.yaml` | Declares both baseline names, seed 42, macro-F1 target 0.45 and `down` recall guardrail 0.50. | The factory does not consume the configured baseline list; material improvement has no defined margin. |
| `pipelines/training_pipeline.py` | Fits both baselines on training inputs and stores validation metrics before fitting candidates. | Baseline work cannot run independently; fit/predict/evaluation time is combined; benchmark outputs lack their own provenance and lifecycle. |
| `models/evaluate.py` | Accuracy, balanced accuracy, macro/weighted F1, per-class metrics, confusion matrix and down false negatives. | Validate input lengths/labels; macro and weighted F1 currently omit the explicit configured label list. |
| `models/tune.py` | Chooses candidates using macro F1 and a recall preference; falls back to all candidates if none meets recall. | Candidate choice is not a baseline-improvement or release gate; fallback status must be explicit in downstream reporting. |
| `data/development.py` | Verifies source and persisted split identities; returns train, validation and provenance only. | Reuse this interface without creating a new split or handing test rows to benchmark code. |
| `features/artifacts.py` | Loads and verifies frozen estimator-specific feature choices from SPEC-04. | Dummy baselines need no feature engineering; optional candidate comparisons need compatible provenance. |
| Tests | Candidate recall selection and broader training/feature tests exist. | No dedicated baseline factory, analytic-metric, stochastic-repeat, or benchmark artifact tests. |
| `reports/metrics/model_metrics.json` | Contains historical baseline validation results. | Historical majority confusion-matrix support totals 5,281 rows; the current split has 5,857 validation rows. These results must not be reused as the current benchmark. |

Current persisted development split: 17,670 training rows across 18 clients and 5,857 validation rows across 7 clients. Read these values from verified metadata at runtime rather than hard-coding them. The training majority is `down`; the implementation must learn the majority from training labels rather than assuming that class name.

SPEC-04 has already used validation for feature confirmation, and historical test results already exist. Consequently, this phase produces a development benchmark, not an independent final generalization estimate. Do not compare the SPEC-04 grouped-fold mean directly against a validation-only baseline score.

## 3. Scope and dependencies

### In scope

- The two prescribed dummy baselines, with deterministic strategy/seed configuration.
- A standalone development-only baseline CLI and reusable runner.
- Training-derived class counts, proportions, majority class and tie behavior.
- A fixed-label evaluation contract and complete aggregate metrics.
- A small predetermined stochastic-repeat study and clear interpretation of its variability.
- Source/split/config/code/environment provenance and stable semantic outputs.
- Candidate-to-baseline comparison rules and SPEC-06 handoff.
- Integration with existing training without duplicated baseline logic.
- Unit, integration, contract, reproducibility and privacy checks.

### Out of scope

- New feature families, rerunning SPEC-04 selection, hyperparameter search or resampling.
- New logistic-regression, tree, heuristic, uniform or prior-probability baselines. Logistic regression remains a candidate/reference model in the existing workflow.
- Calibration, probability ranking quality, log loss, threshold optimization and subgroup error analysis.
- Changing labels, split ratios, assignments, raw data or the existing 0.45 project target.
- Training-plus-validation refitting, final-test access/auditing, deployment or model promotion.
- Statistical confidence claims about future clients or periods from random-seed repeats.

Prerequisites are the validated source and persisted SPEC-03 manifest/assignments. Dummy benchmarks can run without a SPEC-04 feature manifest. Candidate comparisons must verify the appropriate frozen feature/model evidence and identical evaluation population. If split artifacts are missing or incompatible, fail with an instruction to run the existing split workflow explicitly; never regenerate them silently inside the benchmark command.

## 4. Proposed requirements and traceability

These IDs expand the specification's placeholder requirement. Adopt them in SPEC-05 during implementation, after agreeing on the protocol.

| Requirement | Proposed behavior | Verification |
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

## 5. Baseline contract

### 5.1 Inputs and lifecycle

The orchestration layer calls `load_development(data_config, training_config)`, then supplies the reusable runner with training labels, validation labels and ordering/partition provenance. Dummy estimators receive neutral one-column arrays solely to convey row counts. They do not need raw numeric/categorical inputs, preprocessing, identifiers, engineered features or a saved model bundle.

Keep the loader's full-source schema and identity validation upstream. Distinguish that integrity access from model fitting/evaluation: the baseline runner accepts no test argument and cannot transform or score the held-out test partition.

Preconditions:

- Nonempty train and validation, with unambiguous stable row ordering.
- No null or unknown labels; configured labels are unique and contain `down`.
- Every configured class appears in both main partitions, matching the current split contract.
- Unique row identities and disjoint client groups verified by the loader.
- No sample weights, oversampling or class-weight adjustment in dummy fitting.

Canonicalize rows within each development partition by the stable row key before seeded random prediction. Record an ordered row-identity hash without publishing raw keys. This prevents harmless source row reordering from changing which random prediction is associated with which label. A changed source byte hash still requires explicit source/split provenance renewal; ordering stability is not permission to ignore source mismatches.

### 5.2 Most-frequent-class baseline

Fit the current `DummyClassifier(strategy="most_frequent")` on training labels and predict the training-majority class for every validation row. Record the learned majority and training class counts.

Define ties explicitly: retain deterministic estimator behavior for string labels, verify the class ordering under the installed supported library, and document the resulting tie rule. Proposed contract: lexicographically first tied label. Test it; if supported versions differ, normalize tie handling in the factory rather than allowing environment-dependent predictions. Configured report label order must not silently redefine the majority tie rule.

Use this analytic fixture to verify metrics. With K configured classes, majority-predicted class m, and validation prevalence q for m:

- Accuracy = q.
- F1 for m = `2q / (1 + q)`; F1 for every other class = zero.
- Macro F1 = `2q / (K * (1 + q))`.
- Balanced accuracy = `1/K` when all K classes have validation support.
- If m is `down`, down recall = one and down false negatives = zero, despite poor multiclass discrimination.

These identities are tests of metric correctness, not a claim that every dataset's majority class is `down`.

### 5.3 Stratified-random baseline

Fit `DummyClassifier(strategy="stratified", random_state=seed)` using training class proportions. Validation frequencies must not influence sampling probabilities.

Proposed fixed protocol:

- Canonical reference seed: 42, preserving the existing behavior.
- Repeat seeds: `[42, 43, 44, 45, 46]`, ordered and unique.
- Fit one most-frequent baseline and five separate stratified instances: six inexpensive dummy fits total.
- For each seeded instance, predict the full canonical validation batch once and reuse that prediction vector for every metric. Do not redraw predictions independently for different metrics or chunks.
- Record each seed's metrics plus mean, population standard deviation (`ddof=0`), minimum and maximum for macro F1 and down recall.
- Never select the highest/lowest scoring random seed as the benchmark. The reference seed remains explicit even if its score is atypical.

Random-seed variability measures randomness of the classifier on fixed data. It is not a confidence interval for deployment performance. Do not require exact sampled class proportions in finite validation batches or require every class to appear among random predictions.

Probability metrics are excluded from this phase. In particular, stratified dummy probability output must not be described as calibrated risk estimates or automatically added to the review queue.

## 6. Evaluation and candidate comparison

### 6.1 Fixed-label metrics

Reuse `evaluate_predictions` after hardening its contract:

- Require matching, nonzero lengths for truth and predictions; reject unknown/null labels.
- Require a valid explicit ordered label set.
- Pass the label set into macro and weighted F1, per-class metrics and confusion matrices consistently.
- Use `zero_division=0` for undefined precision/F1 and preserve zero-valued records for unpredicted classes.
- Include per-class precision, recall, F1 and support, plus accuracy, balanced accuracy, macro F1, weighted F1, down false negatives and the confusion matrix.
- Verify confusion-matrix totals and supports equal validation row counts.

The main benchmark requires all five truth classes. Standalone metric tests should still define behavior when a truth class is absent: macro F1 retains all configured labels; balanced accuracy retains the library's supported-truth-class convention and must not be relabeled as an all-configured-class average. Document this distinction and avoid using incomplete-class fixtures as release evidence.

### 6.2 Comparison rules

Baseline generation succeeds when it produces valid evidence, regardless of score. Do not filter dummy baselines out because they fail the recall guardrail, and do not require the baselines themselves to reach macro F1 0.45.

For each later candidate on the same validation rows, report:

- Absolute macro-F1 difference from the most-frequent result.
- Absolute difference from canonical seed-42 stratified performance.
- Absolute difference from mean stratified performance across the fixed repeats.
- `beats_both_baselines`: candidate exceeds both majority and canonical stratified scores.
- `material_improvement_met`: proposed minimum absolute improvement of 0.01 over majority, canonical stratified and repeat-mean stratified scores.
- `down_recall_guardrail_met`: candidate down recall >= 0.50, from current training config.
- `project_macro_f1_target_met`: candidate macro F1 >= 0.45, from current training config.

The 0.01 margin is a proposed operational default; agree on it before implementation comparisons. It is not a statistical-significance claim. Keep the margin and existing project target independent. Do not lower either automatically in response to scores.

Report failing flags explicitly. Preserve the current candidate-ranking API initially, but distinguish `selected_for_development` from passing all comparison flags. The existing fallback when no candidate meets recall must not silently produce a successful benchmark/readiness status. Model promotion remains outside this phase.

Before comparing saved results, verify source SHA-256, evaluation partition, assignment identity, ordered-row fingerprint, row count, target/label order and metric-contract version. Reject incompatible or legacy reports with insufficient provenance. Feature configurations may legitimately differ between candidates and feature-independent dummy baselines; record feature provenance for candidates without demanding identical feature sets.

SPEC-04 validation confirmations may supply contextual comparisons when their provenance is demonstrably compatible, but avoid building an automatic adapter to unversioned reports. The simplest initial integration evaluates candidates and baselines together through the shared training runner. SPEC-06 can later consume the frozen benchmark manifest.

## 7. Configuration and APIs

Add `configs/baselines.yaml` with a `baselines` mapping:

| Setting | Proposed default/policy |
| --- | --- |
| `baseline_contract_version` | `1.0` |
| `metric_contract_version` | `1.0` for the newly explicit metric behavior |
| `strategies` | `[most_frequent, stratified]`; both required for the approved initial benchmark |
| `reference_seed` | 42 |
| `repeat_seeds` | `[42, 43, 44, 45, 46]`, including the reference seed |
| `minimum_macro_f1_improvement` | 0.01 absolute, proposed |
| `evaluation_partition` | `validation`; other values rejected |
| `ordering` | Stable row-key sort |
| `output_directory` | `reports/baselines` |

Validate mapping/list types, versions, supported names, duplicates, finite margin bounds, seed integer ranges and booleans masquerading as integers. Reject arbitrary strategy or seed searches. Keep source, split, project target and recall settings in their existing configs.

Avoid two competing baseline strategy lists: migrate `training.baselines` to the new config, updating every caller and documentation. If a temporary compatibility path keeps that list, require exact agreement and raise on conflict. Retain the existing `build_baselines(seed)` behavior for legacy callers where practical; add a validated configured factory/runner for new orchestration.

Proposed interfaces:

- `validate_baseline_config(config)` in `utils/config.py`.
- A configured factory in `models/baseline.py`, preserving independent estimator instances and explicit seeds.
- `evaluate_baselines(train_y, validation_y, config, labels)` returning canonical results, repeats, summaries and learned training priors; no test or feature argument.
- `run_baselines(data_config, training_config, baseline_config, output_dir)` in `pipelines/baseline_pipeline.py` for provenance, loading, ordering and publication.
- A comparison helper consuming typed/versioned evaluation evidence rather than arbitrary dictionaries with only score fields.

Split model execution from file writing so training can reuse the runner in memory without creating or overwriting the standalone benchmark on every run.

## 8. Delivery phases and file-level work

### Phase 0 - Freeze the benchmark protocol

**Files:** SPEC-05 and this plan.

Confirm the two strategies, seed set, tie rule, label behavior and material-improvement margin. Adopt development-only scope and record existing validation/test exposure. Specify compatibility and exit-status rules: malformed inputs fail; a valid benchmark or candidate with weak scores is a reported scientific result, not an execution error.

**Exit:** SPEC-05 has concrete requirements/tests and measurable defaults; no normative `TBD` remains.

### Phase 1 - Configuration and baseline factory

**Files:** `configs/baselines.yaml`, `configs/training.yaml`, `utils/config.py`, `models/baseline.py`.

Implement validation, migrate baseline strategy ownership, preserve legacy factory compatibility, create independent seeded estimators and record effective parameters. Use existing dependencies only.

**Exit:** Invalid settings fail before fitting; the initial configured protocol yields one majority run and five stratified runs.

### Phase 2 - Metric contract and aggregate result objects

**Files:** `models/evaluate.py`, `models/baseline.py`; optionally `models/benchmark.py` if the result/comparison logic warrants separation.

Add metric preconditions and explicit label handling. Implement baseline execution, one prediction vector per run, class priors, repeat summaries and separate timing measurements. Add the analytic majority fixture and fixed-seed tests before integration.

**Exit:** Known predictions produce hand-verifiable metrics; repeat aggregation and canonical seed reporting are deterministic.

### Phase 3 - Development-only CLI

**Files:** new `pipelines/baseline_pipeline.py`, new `scripts/run_baselines.py`; reuse `data/development.py`.

Load verified partitions, canonicalize development ordering, create privacy-safe identity fingerprints, run the shared evaluator and recheck source bytes before publishing. Expose config and output-directory arguments. Do not import/call candidate training, feature selection or test evaluation from this command.

**Exit:** One command produces the benchmark without fitting logistic regression/random forest, writing a model bundle or changing split artifacts.

### Phase 4 - Artifact publication and reproducibility

**Files:** runner, `reports/baselines/`, optional small artifact utility using the existing atomic-write approach.

Write the outputs in Section 9. Publish the manifest last, after all payloads have been written and hashed. Verify safe local artifact paths when reading a benchmark back. Reuse source/split fingerprints and version code/config/environment provenance. Do not overwrite `reports/metrics/model_metrics.json`, feature-study outputs or existing model bundles.

**Exit:** Two isolated runs reproduce semantic artifacts and predictions/metrics; timing differences are treated explicitly.

### Phase 5 - Shared training integration and SPEC-06 handoff

**Files:** `pipelines/training_pipeline.py`, `scripts/train_model.py`, comparison helper, optionally `scripts/evaluate_model.py` for report display.

Thread baseline config into training and use the shared execution path. Preserve canonical `baselines_validation` entries for existing consumers, adding versioned repeat/summary/comparison sections. Bind candidate metric reports to split provenance and metric-contract version. Separate candidate-ranking output from baseline, recall and target flags. Add a benchmark-manifest reference to model metadata when a compatible frozen benchmark is reused.

Reject stale benchmark/candidate evidence before comparison. Adding reporting metadata must not rerun feature selection, change selected feature configs, refit on validation or access test data.

**Exit:** Standalone and training-integrated canonical baseline metrics agree for the same partition/order/config; downstream comparison cannot silently use the historical 5,281-row validation report.

### Phase 6 - Verification and documentation

**Files:** tests below, `readme.md`, SPEC-05, this plan, `reports/baselines/verification.json`.

Run focused tests, the complete regression suite and Ruff. Execute real-data generation twice in separate output directories; compare canonical semantic outputs, source integrity and privacy. Record baseline numbers only after running the implementation. Document the benchmark interpretation and frozen SPEC-06 handoff; update status only with linked evidence.

**Exit:** Requirements and tests are traceable, benchmark artifacts reproduce, and source/split/model artifacts remain intact.

## 9. Artifact contract

All proposed standalone outputs live together in `reports/baselines/` or an explicit isolated output directory.

| File | Required contents |
| --- | --- |
| `baseline_metrics.json` | Metric schema/version; evaluation partition; ordered labels; train/validation row and group counts; canonical majority/stratified metrics; learned training class counts/proportions; majority tie outcome. |
| `baseline_repeats.csv` | Strategy, seed, macro/weighted F1, accuracy, balanced accuracy, down recall, down false negatives and measured fit/prediction seconds for each run. |
| `baseline_summary.json` | Seed set, reference seed, repeat aggregation convention, mean/std/min/max and explicit stochastic-variation interpretation. |
| `baseline_report.md` | Protocol, benchmark table, per-class/confusion-matrix interpretation, timing scope, provisional comparison margin and limitations. |
| `baseline_manifest.json` | Baseline/metric versions; source SHA-256; split manifest/assignment hashes; evaluation/ordered-row identity; effective config hash; code revision, dirty state and source-tree hash; dependency versions; artifact hashes. |
| `verification.json` | Actual test/lint commands and results, reproducibility evidence, source immutability and no-test/model-training checks. |

Persist aggregate results, not raw client/content identifiers or row-level prediction matrices. Derive any ordering/prediction audit fingerprints with the existing project-scoped privacy-safe identity mechanism; never include raw keys in logs. Keep validation labels and predictions in memory for evaluation only.

Store fit and prediction runtime as observed diagnostics with row counts and machine/runtime context. Dummy-model timing is not evidence that a future full preprocessing/model pipeline meets the project's 30,000-row CPU limit. Do not load test rows to create a full-source timing batch.

Define reproducibility over learned priors, ordered identity, per-seed predictions/metrics, summaries and normalized metadata. Exclude timestamps, runtime measurements and hashes of timing-bearing payloads from byte-for-byte semantic comparison; still verify each run's own file hashes for integrity. No new model serialization is needed for a dummy benchmark.

## 10. Test matrix

| Test ID | Given/when | Expected result and planned location |
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

Use `tests/integration/test_baseline_pipeline.py` for orchestration and `tests/contract/test_baseline_contract.py` for artifacts/comparison identity. Extend existing training integration tests for shared execution. Fixtures should be small, synthetic and multi-client; do not use the real test partition as a fixture. Do not use arbitrary statistical tolerances to assert that random predictions exactly match class priors.

## 11. Planned commands

The baseline CLI/config files below are proposed deliverables and do not exist at planning time.

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --data-config configs/data.yaml --training-config configs/training.yaml --baseline-config configs/baselines.yaml --output-dir reports/baselines
.\.venv\Scripts\python.exe -m pytest tests/unit/test_baselines.py tests/unit/test_evaluation.py tests/integration/test_baseline_pipeline.py tests/contract/test_baseline_contract.py tests/integration/test_training_pipeline.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

Run the baseline CLI a second time with a separate output directory, compare the semantic artifacts, and record the exact commands/results in `verification.json`. Keep scratch output separate from an active pytest base directory. Candidate training need not run just to produce the benchmark; exercise integration on synthetic fixtures first.

## 12. Decisions and implementation risks

| Decision/risk | Proposed resolution |
| --- | --- |
| What counts as a baseline? | The two SPEC-00 dummy strategies only; avoid expanding this phase into model selection. |
| Meaning of material improvement | Propose +0.01 absolute macro F1 against majority, canonical stratified and stratified mean; agree before comparison. |
| Reference versus repeat seeds | Preserve seed 42 as canonical; report all five predetermined runs and their summary. |
| Majority tie behavior | Explicit lexicographic string-label tie rule, backed by an estimator compatibility test. |
| Dual configuration ownership | New baseline config owns strategies/repeats; migrate or strictly reconcile the existing training list. |
| Legacy baseline metrics | Mark historical evidence as incompatible until source/split/metric provenance can be verified; do not relabel old results as current. |
| Reused validation and previously evaluated test | Label this benchmark developmental; no new independent performance claim. |
| Stochastic reproducibility | Canonical row order and one full-batch prediction per seeded run; publish repeat dispersion rather than selecting seeds. |
| Shared metric changes | Version explicit label behavior and run existing feature/training regression tests; do not compare differently defined metrics. |
| No baseline improvement or recall pass | Produce valid evidence and failing comparison flags; leave deployment/model decisions to later phases. |
| External historical cutoff evidence | Remains necessary for learned candidates; dummy benchmark quality cannot resolve that uncertainty. |

## 13. Acceptance checklist and definition of done

- [x] SPEC-05 has agreed requirements, tests and defaults with no normative placeholders.
- [x] The prescribed strategies and effective seed schedule are validated and recorded.
- [x] Training labels alone determine majority class and sampling priors.
- [x] Every baseline evaluates the same canonical validation population.
- [x] Dummy execution invokes neither feature engineering, candidate training nor test evaluation.
- [x] Metrics have explicit label/zero-division/input-validation semantics and analytic tests.
- [x] Reference-seed and repeated stochastic metrics are reproducible and correctly summarized.
- [x] Candidate comparisons require compatible provenance and expose separate improvement/recall/target flags.
- [x] Standalone and integrated baseline behavior agrees without duplicated execution logic.
- [x] Versioned manifests bind aggregate results to source, split, configuration, code and environment.
- [x] Artifacts contain no raw sensitive identifiers or row-level predictions.
- [x] Source, split assignments, existing models and feature-study outputs remain unchanged.
- [x] Focused tests, full regression suite, lint and two-run real-data checks pass with recorded evidence.
- [x] SPEC-06 receives a frozen benchmark manifest and clear comparison instructions.
- [x] Prior holdout exposure, stochastic variability and performance/runtime limitations are documented.

The phase is complete when the benchmark can be generated and consumed reproducibly, all agreed checks pass, and its interpretation is documented. Candidate success against the benchmark, the 0.45 target, final-test readiness and deployment remain separate outcomes. This planning task itself generates no new baseline scores or test evidence.


## 14. Implementation evidence (2026-09-08)

The implementation request adopted the proposed protocol, including the two dummy strategies,
canonical seed 42, fixed repeats 42-46, lexicographic tie handling and +0.01 material improvement.
This completion record supersedes the planning-only authorization/status language above.
[SPEC-05](../../specs/SPEC-05-baseline-modeling.md) now contains the concrete contract.

Existing baseline execution was completed with fixed seed validation, effective estimator parameters,
stronger source/split identity and local-artifact checks, explicit missing-split recovery instructions,
and CLI options for configured/frozen training reuse. Dedicated analytic, stochastic, publication,
comparison and synthetic training tests now exercise the shared implementation.

Actual verification: **44 focused tests passed; 105 full-suite tests passed; Ruff passed**.
Three library warnings came from intentional single-truth-class metric fixtures.
`python scripts/verify_baselines.py` produced two real-data benchmarks with identical metrics,
training priors, per-seed prediction fingerprints, summaries and normalized provenance. Each run's
payload hashes were independently verified. Protected source, split, model, feature-study and
historical metric artifacts retained identical bytes. No real-data candidates were fitted and
no test evaluation was performed.

Validation macro F1: most-frequent **0.136794**, canonical stratified **0.208024**.
These are development benchmarks, not release criteria or independent holdout estimates.

Evidence and SPEC-06 handoff:

- [Benchmark report](../../reports/baselines/baseline_report.md)
- [Frozen manifest](../../reports/baselines/baseline_manifest.json)
- [Exact verification commands and results](../../reports/baselines/verification.json)

Use `scripts/train_model.py --baseline-config configs/baselines.yaml --baseline-manifest
reports/baselines/baseline_manifest.json` to reuse the frozen reference. Identity/config/hash
mismatches fail before candidate fitting. Ranking remains separate from improvement/recall/target flags.
