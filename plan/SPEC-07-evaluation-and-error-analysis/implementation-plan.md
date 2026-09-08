# Implementation Plan - Evaluation and Error Analysis

**Source:** `specs/SPEC-07-evaluation-and-error-analysis.md`
**Status:** Proposed; this request authorizes planning only
**Prepared:** 2026-09-08
**Owner:** Babar Ali Khan
**Outcome:** Explain the frozen finalists' validation errors, compare supported subgroups, and record a reproducible model recommendation or explicit rejection for SPEC-08.

## 1. Specification status and objective

SPEC-07 is a draft containing `TBD` scope, requirements, tests and decisions. This plan proposes a concrete evaluation contract grounded in SPEC-00 and implemented SPEC-03/04/05/06. Defaults below become normative only when adopted during authorized implementation. This planning request creates only this document; it does not score models, retrain, change specifications/configurations, regenerate artifacts or access the test partition.

Evaluate the two frozen SPEC-06 finalists on the same verified development validation rows. Reproduce their saved predictions and metrics before drawing new conclusions. Analyze class confusions, missed declines, false refresh alerts, model disagreements and predetermined subgroups using those in-memory prediction vectors. Produce a versioned decision that separates metric eligibility, development preference, unresolved review issues and production readiness.

A valid result may reject both candidates. Passing this phase means trustworthy, reproducible evidence and a clear decision record, not improved model scores or automatic model promotion. SPEC-08 may receive research artifacts even when no model qualifies for release.

## 2. Current-state assessment and evidence

Package paths below are relative to `src/machine_learning_project/`; orchestration lives in root `pipelines/`.

| Area | Observed behavior | Gap / planned use |
| --- | --- | --- |
| `models/evaluate.py::evaluate_predictions` | Validates nonempty matching inputs and explicit labels; returns fixed-label metrics, confusion matrix and down false negatives. | Reuse metric contract 1.0. Add error-derived quantities in a separate versioned analysis contract. |
| `models/evaluate.py::subgroup_metrics` | Default minimum 25 rows; computes macro F1 without explicit labels; silently skips absent columns/small groups; exposes raw category keys. | Do not use unchanged for authoritative analysis. Replace/wrap with explicit labels, denominators, suppression records, validated columns and client privacy policy. |
| `models/training_artifacts.py::load_training_run` | Checks completed SPEC-06 versions, exact payload lists and hashes before trusted local joblib loads; compares model metadata and evaluation identity. | Reuse loading, then bind the saved run to freshly verified source/split/config/selection identities. File integrity alone does not establish compatibility with current data. |
| `models/benchmark.py` | Verifies frozen baseline identity/config/hashes and computes independent comparison flags. | Reuse on the identical validation population; do not compare subgroup or CV scores with full-validation baseline metrics. |
| `pipelines/tuning_pipeline.py` | Freezes CV preference before validation; stores both train-fitted finalists, aggregate validation metrics and scoped prediction fingerprints. | Consume immutable evidence. New evaluation must never enter search/refit paths. |
| `scripts/evaluate_model.py` | Displays a legacy metrics JSON, described as a sealed-test report; falls back to fixed-training validation format. | Add an explicit analysis mode or dispatch to a new CLI while retaining legacy display. Correct the misleading sealed-test description and provenance wording. |
| Config/artifacts | No `configs/evaluation.yaml` or dedicated evaluation pipeline exists. | Add isolated evaluation configuration, artifacts and verification. |
| Tests | Existing fixed-label metrics, training artifact integrity and model round-trip tests. | Extend with error arithmetic, support-aware subgroup metrics, paired comparisons, no-fit instrumentation and decision cases. |

### Frozen SPEC-06 reference

Current reference paths:

- Reports: `reports/training/spec06-reference/`.
- Models: `artifacts/models/training/spec06-reference/`.
- Baselines: `reports/baselines/baseline_manifest.json`.
- Verified reproduction: `reports/training/spec06-reproduction/` and `reports/training/spec06-verification.json`.

Published results, read from the existing report rather than newly computed:

| Finalist | CV macro F1 | Validation macro F1 | Validation down recall | Material baseline improvement | Project macro F1 target |
| --- | ---: | ---: | ---: | --- | --- |
| Logistic regression | 0.411854 | 0.389920 | 0.768068 | Pass | Fail |
| Random forest | 0.443426 | 0.430224 | 0.861367 | Pass | Fail |

Random forest is the frozen CV development preference. Both meet the configured down-recall guardrail; neither reaches 0.45 macro F1. Runtime must read full-precision values, configured thresholds and identity from verified artifacts, never hard-code the rounded table.

The reference validation population has 5,857 rows across seven clients; train has 17,670 rows across eighteen clients. The reproduction run is a reproducibility check, not an additional independent sample to pool with reference results.

## 3. Scope and holdout policy

### In scope

- Verification and fresh prediction replay of both frozen finalists on canonical validation rows.
- Fixed-label aggregate/per-class metrics and raw/row-normalized confusion matrices.
- One-versus-rest down error accounting and paired model disagreement analysis.
- A bounded, predetermined set of one-dimensional content/history subgroups with explicit support rules.
- Descriptive client-composition sensitivity using existing predictions, without refitting.
- Metric eligibility, development reference, review flags and an auditable decision for SPEC-08.
- Private aggregate artifacts, deterministic plots, manifest-last publication and two-run verification.

### Out of scope

- Training, tuning, feature selection, threshold changes, calibration or probability/ranking evaluation.
- New models, ensembles or switching predictions by subgroup.
- New split creation, validation-driven grid changes, train-plus-validation refits or final-test evaluation.
- Reading outcome-window fields such as `trend_pct` to create attractive explanatory slices.
- Row-level error exports, client leaderboards, raw examples or identifiers in public reports.
- SHAP/permutation studies, causal attribution, fairness certification or deployment.
- Production batch-SLA certification; a timing diagnostic alone does not establish this.

Default partition is **validation only**; reject `test`, `all` or an arbitrary external dataset. Reuse `load_development` for upstream full-source integrity validation, then pass validation features/labels and approved context only to analysis. This integrity access is not test scoring. The analysis runner must accept no test dataframe or training entry point.

Validation informed earlier feature confirmation and SPEC-06 confirmation; historical test evaluation also exists. State this in every report and decision. Replaying validation is an audit and diagnostic step, not a fresh independent generalization estimate. Do not say that the historical holdout was never evaluated. Establishing a new final-test policy or obtaining genuinely new evaluation data is a separate future decision.

## 4. Proposed requirements and traceability

| Requirement | Behavior | Tests |
| --- | --- | --- |
| SPEC-07-REQ-001 | Validate analysis versions, partition, slices, support rules, thresholds and output locations before inference. | EVAL-T-001/002 |
| SPEC-07-REQ-002 | Verify source/split, both finalist bundles, selection/config provenance and baseline compatibility. | EVAL-T-003/004 |
| SPEC-07-REQ-003 | Replay canonical validation predictions with zero estimator/transformer fits and no test scoring. | EVAL-T-005/006 |
| SPEC-07-REQ-004 | Match SPEC-06 prediction fingerprints and fixed-label metrics before analyzing errors. | EVAL-T-007/008 |
| SPEC-07-REQ-005 | Compute complete class confusion/error counts with explicit denominators and undefined-rate behavior. | EVAL-T-009/010 |
| SPEC-07-REQ-006 | Compare model errors on identical rows using paired counts. | EVAL-T-011 |
| SPEC-07-REQ-007 | Apply predetermined subgroup definitions consistently, with support and suppression coverage. | EVAL-T-012/013/014 |
| SPEC-07-REQ-008 | Report client-composition sensitivity without refitting or inferential confidence claims. | EVAL-T-015 |
| SPEC-07-REQ-009 | Separate recommendation, metric eligibility, review concerns and production readiness; permit rejecting both. | EVAL-T-016/017/018 |
| SPEC-07-REQ-010 | Publish aggregate-only artifacts and safe plots without raw identities or row-level predictions. | EVAL-T-019/020 |
| SPEC-07-REQ-011 | Bind results to code/config/model/input provenance with safe, complete manifests. | EVAL-T-021/022 |
| SPEC-07-REQ-012 | Reproduce semantic outputs and preserve all existing prerequisites and model artifacts. | EVAL-T-023/024 |
| SPEC-07-REQ-013 | Preserve legacy report display and provide explicit SPEC-08 handoff/limitations. | EVAL-T-025/026 |

## 5. Configuration and analysis protocol

Add `configs/evaluation.yaml` with an `evaluation` mapping. Keep macro F1 target 0.45 and down recall 0.50 in `configs/training.yaml`, and minimum improvement 0.01 in `configs/baselines.yaml`. Evaluation must not introduce conflicting copies or lower them in response to results.

| Setting | Proposed default / policy |
| --- | --- |
| `evaluation_contract_version`, `artifact_schema_version` | `1.0` |
| `metric_contract_version` | `1.0`; required to match saved results |
| `partition` | `validation` only |
| `families` | `[logistic_regression, random_forest]`; both required |
| `ordering` | Stable configured row-key order, inherited from SPEC-05/06 |
| `minimum_subgroup_rows` | 100 |
| `minimum_class_support` | 20 for interpreting a class recall/F1 within a slice |
| `minimum_down_support` | 30 for a supported down-recall alert |
| `minimum_predicted_down_support` | 30 for a supported down-precision interpretation |
| `comparison_tolerance` | 0.001 for choosing among metric-eligible finalists |
| `subgroup_alert_macro_f1_gap` | 0.05 below the same model's whole-validation macro F1, only for adequately supported five-class slices |
| `client_sensitivity` | Leave-one-client-out recomputation from predictions; no fitting |
| `maximum_categories_per_dimension` | 12; rare/overflow categories grouped using frequency, never error score |
| `prediction_n_jobs` / numerical thread limit | 1, recorded; do not mutate fitted estimator settings |
| `output_root` | `reports/evaluation`; unique local run subdirectory |
| `plot_dpi` | 120, fixed figure size, label order and color scales |
| `row_exports`, `probability_metrics`, `resume` | false in initial protocol |

Initial version supports only the defined dimensions/bins and algorithms. Reject unknown keys/names, non-list/duplicate dimensions, booleans masquerading as integer supports, invalid finite ranges, conflicting labels, unsupported partitions and unsafe/occupied destinations. Changing the analysis protocol requires a recorded configuration/version change before running; do not repeatedly adjust slices to find an appealing result.

## 6. Loading, compatibility and replay

1. Validate configs, input paths and safe isolated output locations. A `--dry-run` checks artifacts/identities/configuration and resource scope without predictions, fits or publication.
2. Load and canonically order development rows via `load_development` and `prepare_benchmark_partitions`. Keep train only as needed for inherited identity verification; release it before analysis. No fitting or model evaluation uses it.
3. Verify the completed SPEC-06 training manifest and every report/model hash before trusted local model loading. Verify version, target, ordered labels, row counts, source/split/order fingerprints, model class set and both expected families against the current identity.
4. Check the training run's `resolved_config.json`, `selection.json` and `validation_metrics.json` against their recorded hashes and semantic references. Confirm the effective model parameters, family choice, feature config/manifest references and baseline-manifest reference match model metadata. Hash current data/preprocessing/training/baseline configs using the original conventions; reject unexplained conflicts rather than relabel stale scores.
5. Verify the frozen feature manifest/configs with the original training inputs and the frozen baseline with `load_benchmark`. Differences in model feature sets are legitimate; evaluation population and metric semantics must agree.
6. Select approved raw validation features using the existing feature contract. For each loaded pipeline call `predict` once on the complete canonical batch, under a bounded thread context. Do not call `fit`, `fit_transform`, `partial_fit`, candidate builders, tuning helpers or baseline fitting. Do not call `predict_proba` merely because it exists.
7. Reconstruct the exact SPEC-06 audit convention: `_hash_value(fingerprint(list(predictions)), 'content-trend-tuning-v1:' + source_sha256)`. Compare against the stored finalist prediction fingerprint. Recompute fixed-label metrics with `evaluate_predictions` and require identical stored semantic metrics in the supported recorded environment.
8. Reject prediction or metric drift before new error artifacts are published. Surface an environment/contract incompatibility; do not silently accept changed scores, overwrite the reference or automatically change tolerances.
9. Keep labels/predictions/context aligned in memory. Use positional arrays after canonicalization, validate lengths/index identity and reject null/unknown labels. All error tables, subgroup views and plots reuse these vectors.

An implementation-environment version difference is not automatically permission to deserialize or re-score incompatible artifacts. Require supported serialization/library compatibility and replay agreement; record both training and evaluation environments. Do not require identical source-tree hashes across different phases, since evaluation adds new code; preserve both code identities.

## 7. Aggregate and paired error contracts

### Overall and per-class metrics

Reuse metric contract 1.0 exactly. Record accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1/support, configured label order and confusion matrix. Validate matrix total equals validation rows and row supports match truth counts.

Produce a separate row-normalized confusion matrix where cell (i,j) is count(i,j)/truth_support(i). A zero-support row is marked undefined in diagnostic output, not divided by zero. Preserve the raw integer matrix for arithmetic and audits. Most confused class pairs are derived from off-diagonal counts and rates; rank deterministically by count, then configured truth/prediction order. Distinguish absolute error volume from within-class error rate.

### Decline detection accounting

Treat down versus all other labels as a diagnostic binary view; do not change the five-class predictions.

- TP: true down, predicted down; FN: true down, predicted another label.
- FP: true non-down, predicted down; TN: true non-down, predicted non-down.
- Verify TP+FN equals true-down support, TP+FP equals predicted-down count and all four sum to N.
- Recall = TP/(TP+FN); precision = TP/(TP+FP); false-negative rate = FN/(TP+FN); false-positive rate = FP/(FP+TN).
- Record predicted-down workload count/rate, missed-down count/rate and each non-down class contributing false alerts. Do not label these counts monetary savings or causal refresh benefit.
- For zero denominators return diagnostic `null` plus an explicit reason. Preserve existing metric-contract zero-valued per-class records separately, so an unsupported rate is not presented as evidence of poor performance.

### Paired finalist comparison

On the same ordered rows compute: both correct, only logistic correct, only forest correct, both wrong; prediction agreement/disagreement counts; and true-down cases caught by both, only one or neither. Counts must partition their respective populations exactly.

Use an aggregate truth x logistic_prediction x forest_prediction table (at most 5^3 cells) to explain disagreements and support reproducible marginal checks. Its marginals must reconstruct both confusion matrices. No row IDs or prediction vectors leave memory. Record absolute score/recall deltas without claiming statistical significance. Do not pool reference and reproduction predictions or compare unpaired populations.

Frozen dummy evidence supplies whole-validation comparisons only. Do not invent dummy slice-level scores from aggregate metrics or rerun dummy sampling as part of this phase.

## 8. Predetermined subgroup analysis

### Dimensions and derivation

Initial protocol uses these one-dimensional partitions only:

| Dimension | Definition |
| --- | --- |
| `content_type` | Existing approved categorical input, with explicit missing category. |
| `main_intent` | Existing approved categorical input, with explicit missing category. |
| `age_tier` | Existing configured category; historical cutoff uncertainty remains disclosed. |
| `freshness_tier` | Existing configured category; no post-outcome interpretation. |
| `previous_impressions_bucket` | From `impressions_prev_30d`: missing, zero, (0,100], (100,1000], >1000. Negative/nonfinite values fail upstream validation. |
| `numeric_missingness_bucket` | Missing count across the configured raw numeric feature columns before imputation: 0, 1-2, >=3. |

No cross-products, outcome-derived fields, adaptive quantile bins, protected-attribute fairness claims or score-driven subgroup discovery in version 1.0. These derived groups are diagnostic labels, not new model inputs. Never call a fitted feature-selection/engineering workflow to construct them.

Category handling uses only validation membership counts, never target/prediction scores. Keep categories meeting minimum support, select at most the configured category limit by descending frequency then stable value order, and pool remaining nonmissing categories into a reserved `OTHER` group. Represent missing/pooled categories with collision-safe typed keys so a literal category named OTHER or MISSING cannot collide with a sentinel. Treat this as descriptive grouping, not estimator training. Report pooling/suppression counts without publishing the suppressed category names.

### Supports, metrics and alerts

- Use identical grouping/membership for both finalists. Each dimension is a complete partition of validation before pooling/suppression.
- Publish N, observed truth-class support, missing/pooling indicators and publication status. A group below 100 rows is suppressed from detailed metric tables; its row count contributes to dimension-level coverage reporting.
- For published groups, fixed-label macro F1 retains all five labels. Balanced accuracy averages only supported truth classes, as in the shared metric contract. Report `all_classes_present` and do not treat missing classes as model failures.
- Mark per-class interpretations unsupported below 20 truth examples; down-recall alerts need at least 30 true-down examples; down-precision interpretation needs at least 30 predicted-down examples. Do not hide lack of support behind a displayed zero or a passing flag.
- Publish a macro-F1 gap alert only when every class has at least 20 truth examples. Compare with the same model's whole-validation macro F1; an absolute gap >=0.05 is a review flag, not a significance test or automatic disqualification.
- A supported group down recall <0.50 raises a decline-recall review flag. Keep its denominator and FN count visible. Multiple overlapping dimensions can flag the same rows; do not sum error totals across dimensions.
- Report counts/coverage for published, pooled and suppressed groups so omitted populations remain visible. Do not use small groups to break a model-selection tie.
- Support thresholds improve interpretability; they are not a formal privacy guarantee. Do not expose raw row/client identifiers, category values that are identifiers, rare-category names or downloadable row examples. Any future row-level review workflow requires a separate access/output contract.

The existing subgroup helper cannot silently serve both old and new semantics. Prefer a new `analyze_subgroups` API with required labels/config and explicit statuses. If migrating `subgroup_metrics`, update every caller/test and retain a clearly marked compatibility wrapper where needed; preserve overall metric contract 1.0.

## 9. Client-composition sensitivity and interpretation

Seven validation clients do not support a claim of broadly stable future-client performance from row counts alone. Add descriptive sensitivity using already-computed predictions:

1. Compute whole-validation metrics, then recompute them after omitting each validation client in turn. Do not refit or repredict models.
2. Verify each scenario removes exactly one complete client and that the remaining rows preserve original order and labels.
3. Record the number of scenarios, remaining-row ranges, minimum/maximum macro F1, down recall and paired model gaps. Keep class-coverage/support validity counts explicit; do not include unsupported down-recall scenarios in recall summaries or hide their omission.
4. Persist aggregate sensitivity summaries and a scoped assignment fingerprint, not per-client labels, raw IDs, ranked clients or identifiable scenario tables. Compute per-client intermediate data in memory only.
5. Record whether the pairwise macro-F1 ordering reverses in any supported scenario as a review flag. This does not replace the full-validation decision or authorize selecting the most favorable omitted population.

No bootstrap confidence intervals, hypothesis tests, p-values or significance labels in the initial protocol. These summaries describe sensitivity to the observed client composition, not uncertainty intervals or independent new evaluations. Formal inferential work would need a separate agreed estimand and sampling design; it is not required for this phase.

## 10. Decision contract and SPEC-08 handoff

### Separate decision fields

- `execution_status`: complete/incomplete; integrity/contract failures make analysis incomplete.
- `metric_eligible_models`: finalists satisfying all four existing comparison flags: strict superiority over canonical baselines, material improvement, down recall and project macro-F1 target.
- `development_reference`: preserve SPEC-06's frozen CV preference and source selection hash. This field never implies metric eligibility.
- `recommended_model`: a metric-eligible finalist or null; separate from the development reference.
- `review_flags`: supported subgroup alerts, model-order sensitivity and unresolved cutoff/holdout/runtime evidence, with reasons and coverage.
- `recommendation_status`: `no_candidate_meets_metric_requirements`, `eligible_pending_review` or `eligible_for_packaging_review`.
- `production_ready`: false for this development-only phase. A packaging review is not deployment authorization or proof of independent performance.

### Deterministic recommendation

1. Recompute baseline comparison flags from verified full-validation metrics/config, and confirm they match SPEC-06's saved flags.
2. If no finalist passes every metric flag, set `recommended_model=null` and `recommendation_status=no_candidate_meets_metric_requirements`. Retain each failing reason and the CV development reference. Expected behavior for the current reference is this rejection of both, conditional on successful replay.
3. Otherwise rank eligible finalists by full-validation macro F1. Within 0.001 of the maximum, prefer the frozen CV preference if eligible, then logistic regression, then stable family name. Record that this recommendation uses reused validation evidence; do not modify SPEC-06's selection artifact.
4. Supported subgroup/paired sensitivity concerns trigger `eligible_pending_review`; they do not automatically switch to another model or permit selection of a metric-ineligible finalist. Record all contenders' diagnostic evidence for reviewer judgment. Unresolved cutoff/holdout policy also keeps review pending.
5. `eligible_for_packaging_review` is available only when no declared review blocker remains; even then production readiness stays false and SPEC-08 owns its own acceptance contract. Do not infer that absent subgroup support proves safety.

Do not invent monetary false-positive/false-negative costs, declare one class more important than the approved metric contract, or change class weights/thresholds based on error tables. Convert observations into clearly labeled hypotheses for a future separately authorized experiment, not immediate modifications to frozen models.

The handoff includes immutable model/run manifest references and hashes, the recommended model or null, the development reference, metrics/flags, coverage/sensitivity concerns and known limitations. If no candidate qualifies, hand off the rejection evidence and research references rather than create a production model pointer. SPEC-08 may implement packaging mechanics separately, but this phase must not imply that packaging makes the model acceptable.

## 11. APIs and file-level work

| File | Planned responsibility |
| --- | --- |
| `configs/evaluation.yaml` | Analysis versions, fixed slices/bins, supports, review thresholds, plotting and output policy. |
| `utils/config.py` | `validate_evaluation_config(...)`; reject unsupported fields and conflicting inherited contracts. |
| `models/evaluate.py` | Preserve overall metric behavior; clarify subgroup compatibility/undefined-support semantics. |
| `models/error_analysis.py` (new) | Pure functions for confusion normalization, down accounting, paired counts, subgroup analysis and client sensitivity. No model fitting or raw-row publication. |
| `models/evaluation_decision.py` (new) | Deterministic eligibility/recommendation logic and explicit blocker reasons. |
| `models/evaluation_artifacts.py` (new) | Safe local paths, payload hashes, manifest verification and normalized semantic loading. Reuse shared utilities where appropriate. |
| `pipelines/evaluation_pipeline.py` (new) | `run_evaluation(...)`: validated frozen inputs, canonical replay, analysis/decision, staged publication. |
| `scripts/evaluate_model.py` | Add explicit analysis/config/run options while preserving legacy `--metrics` display; fix historical-holdout messaging. |
| `scripts/verify_evaluation.py` (new) | Focused/full/lint, two isolated replay/analysis runs, hash/privacy/immutability checks and actual evidence. |
| Tests / readme / SPEC-07 / this plan | Traceable tests, usage, limitations and completion linked to results. |

Suggested pure APIs:

- `analyze_errors(truth, predictions, labels)`.
- `compare_paired_predictions(truth, predictions_by_family, labels)`.
- `analyze_subgroups(context, truth, predictions_by_family, labels, config)`.
- `client_sensitivity(groups, truth, predictions_by_family, labels, config)`.
- `decide_evaluation(metrics, comparisons, cv_reference, diagnostics, config)`.

Orchestration should expose `run_evaluation(data_config, training_config, preprocessing_config, evaluation_config, training_report_dir, model_dir, feature_manifest, baseline_manifest, baseline_config, run_id, dry_run=False)`. Keep prediction replay separate from pure metric/diagnostic functions for straightforward no-fit and deterministic tests.

## 12. Artifact contract, plots and provenance

Use a new `reports/evaluation/<run_id>/` directory. Validate safe names/resolved local paths, reject occupied destinations and stage within the same approved root. Do not overwrite legacy metrics/model-card files or SPEC-03/04/05/06 evidence. No new model serialization is needed.

| Artifact | Required contents |
| --- | --- |
| `resolved_config.json` | Effective analysis config, inherited metric/threshold versions, input config hashes and run inputs. |
| `evaluation_metrics.json` | Freshly reproduced full-validation metrics, ordered identity, reference metric/prediction checks, baseline flags and scoped prediction hashes. |
| `class_errors.csv` | Per-family/class supports, errors, precision/recall/F1 and down diagnostic accounting with denominators/statuses. |
| `confusion_matrices.json` | Raw counts and normalized rates with ordered labels and undefined-row markers. |
| `paired_comparison.json` | Paired correctness/down-discovery counts, bounded joint table, score deltas and marginal checks. |
| `subgroup_metrics.csv` | Both finalists' published slice metrics, supports, coverage/statuses and review flags; no raw client or rare-category keys. |
| `subgroup_coverage.json` | Per-dimension mapping policy, published/pooled/suppressed population accounting and support-rule version. |
| `client_sensitivity.json` | Aggregate omission sensitivity and supported-scenario counts; scoped assignment hash, no client leaderboard. |
| `decision.json` | Eligibility, recommendation/null, preserved CV reference, review blockers, production_ready=false and SPEC-08 handoff hashes. |
| `evaluation_report.md` | Findings, error tables, plots, decision and limitations; distinguish observations from future experiment hypotheses. |
| `timings.json` | Load, predict, metric/analysis and publication diagnostics; no unsupported SLA claim. |
| `figures/` | Deterministic confusion and supported diagnostic plots with privacy-safe labels. |
| `evaluation_manifest.json` | Versions, all input/model/reference/config/code/environment hashes and payload/plot hashes; published last. |
| `verification.json` | Exact commands/outcomes, replay/reproducibility/privacy/no-fit/no-test/immutability evidence. Written after checks, referencing the completion manifest without circular hashes. |

Plot the two raw and row-normalized confusion matrices with identical label order/scales and clear counts/rate units. Add a supported-slice down-recall comparison and a class-error contribution chart only if the underlying data are valid and readable. Mark missing/unsupported rates explicitly; never convert them to visual zero. Save deterministic PNGs with fixed fonts/dimensions/DPI and CSV/JSON source values. Use existing matplotlib; no new dashboard or chart dependency is required.

Provenance must include source and split manifest/assignment hashes; ordered evaluation identity; exact SPEC-06 run/model/report/selection hashes; feature/baseline references; prediction audit convention; metric and analysis versions; effective support/bin/decision settings; code revision, dirty state, source-tree hash; and training/evaluation library versions. Record upstream references without rewriting them.

Before final publication recheck all protected inputs/models against preflight hashes. Hash every report and plot, validate safe paths including symlink resolution, then publish the manifest last. Missing/tampered/escaped payloads and incomplete runs must be rejected on readback. An interrupted operation may leave staging diagnostics but must not expose a valid completion manifest.

Semantic reproducibility covers identities, replay hashes/metrics, subgroup membership/coverage, paired and sensitivity summaries, decisions and source plot data. Exclude run IDs/physical paths, timestamps and timings. Verify each run's own PNG/payload hashes; distinguish semantic plot-data equality from renderer-dependent image bytes when environments differ. Never ignore changed decision flags, grouping or replay predictions under a broad numerical tolerance.

## 13. Delivery phases and exits

### Phase 0 - Adopt analysis and decision policy

Replace SPEC-07 placeholders with concrete requirements/tests/defaults. Adopt validation-only scope, the no-qualifying-candidate outcome, fixed subgroup supports and descriptive sensitivity. Record prior holdout exposure and unresolved cutoff evidence.

**Exit:** No normative placeholders remain; SPEC-08 handoff and review/promotion boundaries are explicit.

### Phase 1 - Configuration, frozen loaders and replay

Implement config validation, read-only preflight, compatible trusted loading, canonical prediction replay and exact saved-metric/fingerprint checks. Correct the legacy evaluation display's wording.

**Exit:** Invalid or stale inputs fail before scoring/publication; a valid replay invokes no fitting or test scoring.

### Phase 2 - Error and paired analysis

Implement pure confusion/down-accounting/paired-count functions and arithmetic invariants, including undefined denominators.

**Exit:** Analytic fixtures reproduce exact counts/rates and both joint-table marginals.

### Phase 3 - Supported slices and client sensitivity

Implement deterministic grouping, missing/pooled keys, support statuses/coverage, review flags and aggregate leave-one-client-out sensitivity.

**Exit:** Both finalists share memberships, suppressed populations remain accounted for, no unsupported metric drives a recommendation, and no raw client output appears.

### Phase 4 - Decision and publication

Implement the recommendation/null policy, report/plots/handoff, isolated staging and safe manifest reader.

**Exit:** Failed target flags lead to an explicit rejection rather than a silent fallback; valid decisions and artifacts remain readable and integrity-checked.

### Phase 5 - Verification and documentation

Run focused/full tests and Ruff; execute the analysis twice with the same frozen reference in different output directories. Record replay agreement, paired/slice/decision reproducibility, private outputs and immutable prerequisites. Update docs/status only with actual linked evidence.

**Exit:** SPEC-08 receives a frozen decision/handoff and reproducible aggregate error evidence. Candidate eligibility, final-test policy and deployment remain separate outcomes.

## 14. Test matrix

| Test ID | Scenario | Expected result / planned location |
| --- | --- | --- |
| EVAL-T-001 | Valid fixed analysis config | Versions, slices/bins/supports and inherited thresholds resolve; `tests/unit/test_evaluation_config.py`. |
| EVAL-T-002 | Unknown/duplicate slices, bad types/ranges, test partition, outcome field, unsafe path | Fail before predictions or publication. |
| EVAL-T-003 | Source/split/order/count/label/metric identity mismatch | Reject stale or legacy evidence rather than recompute under a new identity. |
| EVAL-T-004 | Tampered/incomplete model/report/selection/config/baseline evidence | Reject before trusted loading or scoring as appropriate; required families/params/features agree. |
| EVAL-T-005 | Instrument fit/fit_transform/partial_fit/builders to fail | Evaluation succeeds using fitted transforms/predict only; zero fit calls. |
| EVAL-T-006 | Instrument test access and prediction calls | Only canonical validation is scored; one complete predict call per family; no baseline regeneration. |
| EVAL-T-007 | Same saved model/input/environment replay | Exact prediction hash and metric agreement with SPEC-06. |
| EVAL-T-008 | Changed prediction vector or metrics despite plausible scores | Fail before new analysis completion; no automatic tolerance/reference update. |
| EVAL-T-009 | Known five-class errors and matrix | Correct labels/supports/totals, off-diagonal ranks and normalized rows; `tests/unit/test_error_analysis.py`. |
| EVAL-T-010 | No true/predicted down or no non-down truth | Correct TP/FP/FN/TN; null unsupported diagnostic rates, no divide-by-zero or false pass. |
| EVAL-T-011 | Known paired predictions | Correct both/one/neither counts and joint-table marginals; unpaired inputs rejected. |
| EVAL-T-012 | Missing categories, reserved-name collisions, bin-edge values and rare overflow | Deterministic complete grouping and shared membership across models. |
| EVAL-T-013 | N/support exactly below/at/above thresholds; missing truth classes | Explicit statuses, fixed-label macro, documented balanced accuracy and no unsupported alerts. |
| EVAL-T-014 | Suppressed/pooled slices and overlapping dimensions | Coverage sums correctly per dimension; no cross-dimension sum or rare identity output. |
| EVAL-T-015 | Client omission sensitivity fixture | Whole-client removal, no fits/predicts, correct ranges/reversal/support counts, no client leaderboard. |
| EVAL-T-016 | Neither finalist reaches target, although other flags pass | Complete scientific result; recommended_model=null; CV reference preserved. |
| EVAL-T-017 | One/both eligible, exact ties and reordered input dictionaries | Deterministic declared recommendation; threshold equality follows SPEC-05. |
| EVAL-T-018 | Supported slice alerts or missing evidence | Review status explicit; no automatic swap to an ineligible model, no production_ready=true. |
| EVAL-T-019 | Seeded private identifiers/categories in fixtures | Public artifacts/plots/logs contain no raw IDs or row prediction vectors. |
| EVAL-T-020 | Plotting zero/unsupported values and label ordering | Counts/rates/scales agree with source tables; unsupported values not shown as zeros. |
| EVAL-T-021 | Completed report/plots/manifest | Complete versions/provenance/hashes; `tests/contract/test_evaluation_contract.py`. |
| EVAL-T-022 | Missing/tampered/path-escaped payload or interrupted write | No valid completion accepted; existing outputs preserved. |
| EVAL-T-023 | Two synthetic and two real replay runs | Identical semantic errors/slices/decisions/prediction hashes and independently valid manifests. |
| EVAL-T-024 | Protected snapshots before/after success/failure | Source/splits/features/baselines/training reports/model bundles unchanged. |
| EVAL-T-025 | Legacy display and analysis dry run | Old reports still display with accurate provenance wording; dry run predicts/fits/writes nothing. |
| EVAL-T-026 | Report/SPEC-08 handoff review | No-eligible outcome supported; limitations/blocked readiness/hypotheses clearly separated. |

Use small synthetic multi-client fixtures and existing SPEC-06 fixture patterns. Do not retrain real-data models to test evaluation. Use fit spies that permit fitted transformer `transform` calls but reject fitting; instrument prediction count separately. Real runs prove replay/reproducibility, not leakage correctness by themselves.

## 15. Planned commands

The analysis-mode flags, new config and tests below are deliverables; these commands are not valid until implemented.

```powershell
# Read-only identity/config/artifact preflight: no predictions, fits or output publication.
.\.venv\Scripts\python.exe scripts/evaluate_model.py --evaluation-config configs/evaluation.yaml --training-report-dir reports/training/spec06-reference --model-dir artifacts/models/training/spec06-reference --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --dry-run

# Two independent output directories, the same frozen finalist inputs.
.\.venv\Scripts\python.exe scripts/evaluate_model.py --evaluation-config configs/evaluation.yaml --training-report-dir reports/training/spec06-reference --model-dir artifacts/models/training/spec06-reference --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --run-id spec07-reference
.\.venv\Scripts\python.exe scripts/evaluate_model.py --evaluation-config configs/evaluation.yaml --training-report-dir reports/training/spec06-reference --model-dir artifacts/models/training/spec06-reference --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --run-id spec07-reproduction

.\.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_config.py tests/unit/test_error_analysis.py tests/unit/test_evaluation.py tests/integration/test_evaluation_pipeline.py tests/contract/test_evaluation_contract.py tests/contract/test_training_contract.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

The verifier should orchestrate these two runs or validate them after manual execution, never accidentally retrain or perform the SPEC-06 grid. Two analysis runs should perform zero fits and four full-validation finalist prediction calls in total; diagnostics reuse those predictions. Record actual call counts and exact commands. Keep scratch outside pytest's active base directory and avoid overwriting prior completed runs.

## 16. Decisions and implementation risks

| Issue | Proposed resolution |
| --- | --- |
| SPEC-07 is only a placeholder | Adopt explicit replay/diagnostic/decision requirements during implementation. |
| Both current finalists miss macro F1 0.45 | Expected null recommendation if replay confirms; retain CV reference for research. |
| Historical holdout wording is misleading | Correct CLI/report text; no new test evaluation in this protocol. |
| Saved hashes can be intact but refer to another population/config | Verify fresh identity and original contract compatibility in addition to file hashes. |
| Existing subgroup macro F1 omits configured labels | New versioned subgroup API with fixed labels and explicit support semantics. |
| Rare/missing groups encourage overinterpretation | Predetermined support rules, pooled/suppressed coverage and null unsupported rates. |
| Group alerts overlap and involve many comparisons | Descriptive review flags only; no additive totals or significance claims. |
| Only seven validation clients | Aggregate composition sensitivity, no client leaderboard or inferential interval claims. |
| Outcome fields could make error explanations look stronger | Strict diagnostic allowlist; no post-outcome slices or causal claims. |
| Row-level examples can expose private identifiers | Aggregate error taxonomy/joint counts only; separate future review-access contract. |
| Favorable slices tempt retuning/model switching | Freeze models/policy, keep hypotheses separate and never override failed global eligibility. |
| Model replay changes under a different environment | Require supported loading and exact semantic replay; investigate drift rather than relabel reference. |
| Packaging mistaken for readiness | Explicit null/blocked handoff and production_ready=false in this phase. |

## 17. Acceptance checklist and definition of done

- [ ] SPEC-07 adopts concrete requirements/tests/defaults without normative placeholders.
- [ ] Both frozen finalist inputs and baseline/config/selection provenance match current validation identity.
- [ ] Replay reproduces saved metrics and scoped prediction fingerprints exactly.
- [ ] Evaluation invokes no estimator/transformer fitting, baseline generation or test scoring.
- [ ] Class/down/paired arithmetic and denominator edge cases are verified.
- [ ] Predetermined slices are shared, support-aware and fully accounted for through pooling/suppression.
- [ ] Client sensitivity is descriptive, uses no refitting and publishes no client leaderboard.
- [ ] Metric eligibility, CV reference, recommendation/null, review flags and readiness are distinct.
- [ ] Both failing the target is a supported complete result, with no silent fallback or retuning.
- [ ] Reports/plots are aggregate-only, correctly labeled and consistent with source tables.
- [ ] Isolated manifests bind safe payloads to full source/model/config/code/environment provenance.
- [ ] Existing source/split/feature/baseline/training/model artifacts remain unchanged.
- [ ] Focused/full tests, Ruff, two real replay runs and privacy/integrity checks pass with recorded evidence.
- [ ] Legacy report display remains compatible and accurately states historical exposure.
- [ ] SPEC-08 receives a frozen handoff and all limitations, even when no candidate qualifies.

Completion means reproducible diagnostic evidence and an honest, usable recommendation or rejection. It does not mean independent holdout success, resolved cutoff uncertainty, production readiness or deployment. This request creates the implementation plan only.
