# Implementation Plan - Feature Engineering and Selection

**Source:** `specs/SPEC-04-feature-engineering-and-selection.md`
**Status:** Development implementation complete and verified; production timing/test-policy limitations documented
**Prepared:** 2026-09-07
**Owner:** Babar Ali Khan
**Outcome:** Justified, reproducible model inputs for `trend_direction` classification.

## 1. Specification status and planning assumptions

At planning time, SPEC-04 defined its outcome but left scope, requirements, tests, and open decisions as `TBD`. The user subsequently authorized implementation. The development defaults below have been implemented and the source specification now records the resulting contract. Historical metadata timing remains an explicit production limitation rather than an assumed fact.

### Implementation record (2026-09-07)

- Delivered the v2 feature registry, estimator-compatible engineering, raw-input pipeline integration, stable missing-column handling, and serialized inference compatibility.
- Added `data/development.py` to verify the persisted split and return only development rows. Training no longer refits on train plus validation, evaluates test, or benchmarks all source rows.
- Delivered the five-variant grouped study, train-only diagnostics, validation confirmation, frozen estimator-specific configurations, and provenance-checked handoff through `train_model.py --feature-manifest`.
- Centralized study artifacts in `reports/features/`, including `feature_manifest.json`, instead of publishing a second manifest under `artifacts/metadata/`. This keeps isolated output directories self-contained and allows referenced artifact hashes to be verified locally.
- Used a fixed, validated operation registry rather than arbitrary expression graphs. Unsupported operations/dependencies are rejected; no optional learned selector or extra dependency was added.
- The existing zero-fill helper now uses v2 missing-plus-indicator semantics. The preprocessing configuration version migrated to 2.0 and must agree with engineering; saved legacy bundles retain their original path.
- Synthetic and full-repository tests cover leakage boundaries, edge cases, grouped fitting, serialization, frozen handoff, and source/split tampering. Verification results are recorded in `reports/features/verification.json`: 62 tests pass, Ruff passes, the 30-fold-result study reproduces exactly apart from timings, and the real-data frozen handoff passes.
- A notebook is deferred because the CLI and tested modules provide the complete workflow. Final-test authorization/auditing remains SPEC-03 work; historical cutoff availability still needs external evidence.

Use the current client-grouped split and strict data contract. Prediction occurs at the end of the previous 30-day observation period, before the following outcome window. A feature is eligible only if every dependency was available at that cutoff. Column names alone do not establish availability, especially for mutable metadata, age, freshness, and externally collected search measurements.

Maintain a usable unchanged-feature reference. Feature engineering need not improve measured performance to complete this phase: rejecting an experiment with documented evidence is a valid result. The project macro-F1 target of 0.45 and `down` recall guardrail of 0.50 are existing training settings, not guaranteed feature-engineering outcomes.

## 2. Repository assessment

Paths under `src/machine_learning_project/` below refer to the application package.

| Existing component | Observed behavior | Planned response |
| --- | --- | --- |
| `configs/data.yaml` | Strict 44-column source contract; 11 numeric and 8 categorical inputs | Preserve the raw schema; keep generated features in a separate contract. |
| `features/selection.py` | Ordered allowlist, numeric conversion, exclusion of IDs, target, and dropped columns | Extend validation to sensitive-only fields, generated dependencies, duplicate names, and output order. |
| `features/engineering.py` | `add_cutoff_safe_features` calculates two previous-period ratios on a copy | Turn this into a validated, serializable pipeline stage; currently undefined ratios become zero and missing dependencies silently skip features. |
| `features/preprocessing.py` | Train-fitted median imputation, constant categorical fill, one-hot encoding, optional numeric scaling | Reuse; define behavior for all-missing inputs and stable feature names. |
| `data/preparation.py` | Separates features, target, identifiers, and dropped fields; rejects unclassified source columns | Keep this raw-data boundary; avoid passing generated columns through raw-schema validation. |
| `models/train.py` | Logistic-regression and random-forest pipelines contain preprocessing and a model | Add the same engineering stage to both; preserve their different scaling settings. |
| `pipelines/training_pipeline.py` | Selects using validation, refits on train plus validation, evaluates test, and benchmarks all rows | Do not use unchanged for feature experiments; separate development from final evaluation first. |
| `inference/predictor.py` | Selects configured raw inputs before calling the saved pipeline | Keep callers supplying raw inputs; compute derived features inside the saved pipeline. |
| SPEC-03 and split artifacts | Deterministic source-bound grouped splitting exists; sealed-test workflow remains pending | Reuse assignments and provenance; explicitly track the unresolved test-access dependency. |
| Existing tests | Cover basic exclusions, row preservation, missing/unseen values, and train-only imputation | Add engineering, lineage, selection, serialization, and development-isolation coverage. |

The README already reports test results. Treat the existing holdout as previously evaluated; it cannot be described as a newly untouched final test. Existing full-data EDA is descriptive context, not a basis for fitting thresholds or choosing features. Record this exposure in the experiment report and defer an unbiased final estimate to an appropriately untouched future holdout under the SPEC-03 policy.

## 3. Scope

### In scope

- Versioned raw-to-derived feature definitions, lineage, availability assumptions, and exclusions.
- Safe deterministic ratios, a small optional log-transform family, and explicit missingness semantics.
- A reusable engineering/preprocessing/selection pipeline shared by training and inference.
- Train-only diagnostics and a bounded, reproducible feature-family ablation study.
- Ordered feature manifests, experiment evidence, and compatibility checks.
- Tests and documentation needed to hand inputs to SPEC-05 and SPEC-06.

### Out of scope

- Changing labels, imputing or rewriting the raw CSV, or rebuilding the split algorithm.
- Broad model/hyperparameter searches, resampling studies, target encoding, embeddings, or external enrichment.
- Final test evaluation, deployment, and a complete model-packaging redesign.
- Claims of future-period validity without timestamped observations.
- Automatic removal of unusual observations or correlated features based on full-data EDA.

## 4. Proposed requirements and verification

These IDs are proposed replacements for the placeholder `SPEC-04-REQ-001`, not requirements already approved in the specification.

| Requirement ID | Proposed requirement | Primary tests |
| --- | --- | --- |
| SPEC-04-REQ-001 | Maintain a versioned ordered registry of raw and derived inputs, types, formulas, dependencies, availability, and justification. | FE-T-001, 002, 013 |
| SPEC-04-REQ-002 | Reject target, IDs, sensitive fields, outcome-window fields, and any derived feature depending on them. | FE-T-002, 003 |
| SPEC-04-REQ-003 | Engineering preserves row count, index, order, and source values; missing required dependencies fail explicitly. | FE-T-004, 005 |
| SPEC-04-REQ-004 | Define missing, zero-denominator, negative, and non-finite handling for every numeric transformation. | FE-T-006, 007 |
| SPEC-04-REQ-005 | Fit all learned transformations and selection decisions only on the training partition or current training fold. | FE-T-008, 009 |
| SPEC-04-REQ-006 | Maintain consistent ordered names and dimensions for fitted training, validation, and inference transforms. | FE-T-010, 011 |
| SPEC-04-REQ-007 | Compare a fixed small set of feature families using grouped development data and a predeclared decision rule. | FE-T-009, 012 |
| SPEC-04-REQ-008 | Prevent feature development from transforming, scoring, or using outcomes from the final test partition. | FE-T-014 |
| SPEC-04-REQ-009 | Persist feature lineage, exclusions, fitted selection state, and source/split/config/code provenance. | FE-T-013, 015 |
| SPEC-04-REQ-010 | Reuse the saved feature pipeline at inference without requiring targets or precomputed derived inputs. | FE-T-011, 016 |
| SPEC-04-REQ-011 | Reproduce semantic manifests and decisions for the same source, split, configuration, environment, and seed. | FE-T-012, 015 |
| SPEC-04-REQ-012 | Document accepted/rejected features, timing uncertainty, previous test exposure, and empirical limitations. | FE-T-017 |

## 5. Feature contract

### 5.1 Raw inputs and exclusions

The reference uses the current 19 configured inputs:

- Numeric: `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`, `content_age_days`, `age_tier_order`, `days_since_last_update`.
- Categorical: `competition_level`, `content_type`, `main_intent`, `model_used`, `age_tier`, `freshness_tier`, `word_count_tier`, `char_count_tier`.

Retain all current exclusions in `configs/data.yaml`, including `trend_pct`, current/last-30-day metrics, 90-day aggregates, associated rates/tiers, and `provider_used`. The latter is already excluded; document its rationale without assuming it is itself an outcome measurement. Exclude `content_id` and `client_id` from model matrices; retain group information separately only for grouped splitting or folds.

Validate generated dependencies recursively against the raw allowlist. Reject unknown names, cycles, output collisions, duplicate entries, and role conflicts. Never allow a derived feature to bypass an excluded raw field. A predictor request may contain identifiers for response association, but the engineering transformer receives only approved raw model inputs.

### 5.2 Initial derived feature candidates

| Feature/family | Formula or behavior | Rationale and edge cases |
| --- | --- | --- |
| `previous_ctr` | `clicks_prev_30d / impressions_prev_30d` when denominator is positive | Previous-period click efficiency; zero numerator with positive denominator is a genuine zero. |
| `previous_sessions_per_click` | `sessions_prev_30d / clicks_prev_30d` when denominator is positive | Previous-period activity relationship; values above one are not automatically invalid. |
| Ratio unavailable indicators | One flag per ratio where either source is missing or denominator is zero | Preserve the distinction between undefined ratios and genuine zero activity. |
| Optional log family | `log1p` of search volume, CPC, word/character counts, and the three previous-period counts | Test skew compression while retaining original columns; only non-negative inputs are valid. |

Proposed ratio policy: retain undefined ratios as missing until train-fitted imputation, with the associated indicator set. Reject negative values in declared non-negative sources and reject non-finite observed numeric inputs with actionable errors. Do not silently clip, fill undefined ratios with zero, or drop rows. Handle overflow explicitly so generated outputs cannot contain infinity. Changing the helper's current zero-fill behavior requires a feature-contract version bump.

Require dependency columns even when their values are all missing. For all-missing training numeric columns, preserve dimensionality using a documented constant fallback of zero and record that fallback in metadata. Retain categorical missing-value and unknown-category behavior. The implementation must verify name/width stability for these cases rather than relying on library defaults.

Defer age/update ratios and new tier boundaries until their cutoff meaning is established. Existing tier/count redundancy and `model_used` are candidates for family ablation, not automatic deletion. Do not derive current trend, reconstruct `trend_pct`, or use aggregates that overlap the outcome window.

### 5.3 Configuration design

Add `configs/features.yaml` containing:

- `feature_contract_version`, `registry_schema_version`, and `engineering_version`.
- Enabled feature families, ordered derived definitions, dependency names, and output roles.
- Ratio and numeric-invalid policies, optional log-source allowlist, and empty-column policy.
- Selection mode (default `none`), fixed experiment variants, seeds, grouped-fold count, and decision thresholds.
- Paths for the feature manifest, diagnostics, and experiment reports.

Keep `configs/data.yaml` authoritative for source roles and `configs/preprocessing.yaml` authoritative for imputation/encoding/scaling. Existing callers obtain the feature-contract version from preprocessing config: migrate them to the feature config and reject disagreement during the transition. Do not add synthetic columns to raw `required_columns`, expected column count, or inference required inputs.

## 6. Pipeline architecture and fit lifecycle

Use this sequence:

`validated raw rows -> existing raw feature selection -> engineering -> preprocessing -> optional learned selector -> estimator`

Implement an estimator-compatible `CutoffSafeFeatureEngineer` in `features/engineering.py`, with explicit constructor configuration, `fit`, `transform`, and `get_feature_names_out`. `fit` validates and records the input/output schema without learning from labels. `transform` validates dependencies, copies input, and emits a stable ordered dataframe. Importable classes are required for reliable serialization. Retain the existing helper as a documented wrapper or migrate its callers explicitly.

Keep feature-contract allowlisting separate from statistical selection. Prefer a new `features/diagnostics.py` for train-only missingness, constant-column counts, cardinality, numeric correlation, and redundancy reports. In the first release, selection means choosing a predeclared feature-family configuration; it does not require introducing a supervised selector. If a constant-feature selector is enabled, fit it inside each fold's pipeline, record its support mask, preserve surviving names, support sparse output, and fail clearly if no features survive.

Clone the complete pipeline for every fold and experiment. Imputation medians, scaling, categorical vocabulary, any learned thresholds, and masks must be fitted on that fold's training rows. Diagnostics may inform a later explicitly versioned experiment, but must not be precomputed over validation/test and reused as fitted state.

During development, fit on train and predict validation only after the experiment rule is frozen. No train-plus-validation refit occurs in the feature-study command. Final refitting belongs to the later frozen-model workflow and must record its fitting population separately.

Persist the pipeline with its feature config and manifest reference. `Predictor` continues to select raw inputs using the saved data config, then invokes the saved engineering pipeline. New loaders must validate contract compatibility and define legacy behavior explicitly: existing artifacts without an engineering stage continue through their original path, while incomplete new-format artifacts fail clearly.

## 7. Bounded feature-selection experiment

1. Verify source SHA-256 and reconstruct the existing SPEC-03 assignments using its established hashing/aliasing logic. Fail on mismatch; do not search for a different split during feature comparison.
2. Expose only train and validation to the development runner. Full-source schema/integrity checking may remain upstream, but fitted diagnostics and candidate code must not receive test rows or labels.
3. Freeze variants: A, current raw inputs; B, A plus ratios/indicators; C, B plus logs; D, B with existing age/freshness/length tier representations removed while retaining continuous values; E, B without `model_used`. Resolve cutoff eligibility before running any variant.
4. Use three deterministic client-grouped folds within the training partition, provided each fold has usable class coverage. Preflight feasibility; fail and document a revised protocol if infeasible rather than falling back to random row folds.
5. Hold existing estimator hyperparameters fixed. Evaluate both existing estimators on the same folds, with five variants yielding at most 30 fold fits. Record per-fold macro F1, `down` recall, transformed width, fit/predict time, and group coverage. No hyperparameter tuning is included.
6. Select within each estimator family by mean grouped-fold macro F1 among variants whose mean `down` recall meets 0.50. Proposed tie tolerance is 0.005 absolute macro F1; within that tolerance prefer fewer transformed columns, then the reference variant, then stable variant ID. Record worst-fold recall and variability, not only means.
7. Fit each family's selected variant and its reference on the full training partition and evaluate once on validation. Proposed promotion requires validation `down` recall at least 0.50 and macro F1 no more than 0.005 below that family's reference; prefer the reference when complexity rises without at least 0.005 improvement. Deduplicate identical reference/selected runs.
8. If no candidate qualifies, retain the reference provisionally and report unmet guardrails. Do not relax thresholds after seeing results. Hand estimator-specific conclusions to SPEC-05/06 instead of declaring a final model winner.

These thresholds, variant budget, and three-fold design are proposed defaults requiring agreement before implementation experiments. Small numbers of independent clients limit statistical confidence; report fold dispersion and sample/group counts without claiming a significant gain from tiny differences. Validation confirmation must not become an iterative feature search.

## 8. Delivery phases

### Phase 0 - Finalize the contract and dependencies

**Tasks:** Replace SPEC-04 placeholders with agreed requirements/tests; audit cutoff availability for every raw input; confirm formulas, error policy, experiment budget, and promotion rule. Document the previous test evaluation. Resolve the SPEC-03 development/test interface before empirical experiments.

**Exit:** Every planned feature has a role and availability rationale; unresolved timing assumptions are explicit; the development path excludes test data. Pure transformer work and synthetic tests can proceed while the experiment prerequisite is resolved.

### Phase 1 - Registry and configuration validation

**Files:** `configs/features.yaml`, `utils/config.py`, `features/selection.py`; add `features/registry.py` if needed to isolate lineage logic.

**Tasks:** Validate version fields, families, dependencies, types, unique names, forbidden ancestry, numeric policies, experiment thresholds, and source/output role separation. Define canonical config hashing and ordered registry export.

**Exit:** Valid settings resolve to one ordered feature schema; invalid settings fail before reading experimental outcomes.

### Phase 2 - Deterministic engineering and preprocessing compatibility

**Files:** `features/engineering.py`, `features/preprocessing.py`, `tests/unit/test_feature_engineering.py`, `tests/unit/test_preprocessing.py`.

**Tasks:** Implement the transformer, safe division and indicators, optional logs, input non-mutation, index preservation, finite-output checks, and stable names. Make preprocessing consume resolved output roles and handle all-missing columns consistently.

**Exit:** Hand-calculated fixtures and missing/invalid edge cases pass; every output name maps back to its sources.

### Phase 3 - Training and inference integration

**Files:** `models/train.py`, `pipelines/training_pipeline.py`, `inference/predictor.py`, `inference/validation.py`, relevant scripts.

**Tasks:** Build both estimator pipelines from the same registry; thread feature config through callers; save it with the artifact; preserve inference raw-input requirements. Coordinate development/final-evaluation separation with SPEC-03, including removal of full-data benchmarking from the development path.

**Exit:** Raw requests traverse the same transformations before and after serialization; a development run does not prepare, transform, benchmark, or score test data.

### Phase 4 - Diagnostics and controlled comparison

**Files:** new `features/diagnostics.py`, `pipelines/feature_pipeline.py`, and `scripts/engineer_features.py`.

**Tasks:** Add train-only diagnostics, grouped-fold feasibility checks, fixed variants, deterministic decision logic, validation confirmation, and aggregate reports. Reuse existing evaluation helpers and estimator builders rather than duplicating metrics/model definitions.

**Exit:** A single command produces an auditable selection decision, including a valid outcome of no accepted change.

### Phase 5 - Provenance and handoff artifacts

**Files:** `artifacts/metadata/feature_manifest.json`, `reports/features/`, artifact saving/loading code.

**Tasks:** Write stable semantic metadata and ordered name mappings; bind them to source/split/config/code/environment; retain experiment-specific outputs without overwriting the current production model. Write individual files through temporary paths and atomic replacement; publish a final manifest only after all referenced artifacts exist and their hashes verify.

**Exit:** Selected inputs and exclusions reproduce; downstream training can load the frozen configuration without consulting full-data reports or recomputing selection.

### Phase 6 - Verification and documentation

**Files:** tests listed below, `readme.md`, SPEC-04, this plan.

**Tasks:** Run focused tests followed by the repository suite and lint; execute the feature command twice and compare semantic artifacts; review privacy, timing assumptions, and selection evidence. Add an optional notebook only as a thin walkthrough of tested functions.

**Exit:** Acceptance evidence is linked; requirements are marked complete only where verified. Leave unresolved final-test policy in SPEC-03 explicitly pending.

## 9. Artifact contract

| Output | Required content |
| --- | --- |
| `artifacts/metadata/feature_manifest.json` | Schema/engineering/feature versions; source SHA-256; split/assignment hashes; canonical config hash; code revision and dirty-worktree indicator; dependency versions; input/output order; selected variant per estimator; fitting population; exclusions and reasons. |
| `reports/features/feature_catalog.csv` | Feature name, role, source dependencies, formula, unit, cutoff rationale, missing/error policy, family, and inclusion status. |
| `reports/features/training_diagnostics.json` | Training-only missingness, ranges, cardinality, constants, redundancy, fallback decisions, and sample/group counts. |
| `reports/features/ablation_results.csv` | Variant, estimator, fold, seed, macro F1, `down` recall, input/output width, timing, and eligibility. |
| `reports/features/selection_report.md` | Frozen protocol, grouped-fold results, validation confirmation, accepted/rejected families, rationale, previous test exposure, and limitations. |
| Saved model bundle, when later training runs | Complete fitted pipeline, saved data/preprocessing/feature configs, resolved names and optional mask, feature-manifest identity, and compatibility version. |

Store aggregate evidence and names, not row-level transformed datasets or raw client/content identifiers. Reuse privacy-safe split references. Deterministic comparison covers semantic metadata, assignments, names, and decisions; timestamps, runtime measurements, and serialization bytes are not required to be identical. Record machine/environment details when interpreting runtime against the existing 30-second batch target; development measurements use development inputs only.

## 10. Test matrix

| Test ID | Scenario | Expected result / location |
| --- | --- | --- |
| FE-T-001 | Valid and invalid registry/config versions, roles, duplicates, cycles, collisions, policy values | Ordered schema or actionable error; new `tests/unit/test_feature_registry.py`. |
| FE-T-002 | A direct or nested generated dependency uses target, identifier, sensitive-only or dropped input | Fail before transformation or fitting; registry tests. |
| FE-T-003 | Full source rows contain forbidden fields | Prepared model input contains only allowed raw/derived names; extend data-pipeline integration tests. |
| FE-T-004 | Non-default indices and reordered rows | Row count/order/index and original dataframe remain intact; engineering unit tests. |
| FE-T-005 | Required dependency absent versus present but entirely missing | Absence errors; all-missing follows declared fallback without dropping dimensions. |
| FE-T-006 | Normal ratios, zero numerator, zero denominator, and missing sources | Exact expected values and unavailable indicators; no conflation of undefined with observed zero. |
| FE-T-007 | Log zero/positive values, negatives, infinity, and overflow-prone values | Correct formulas or explicit error; no infinite transformed output. |
| FE-T-008 | Validation-only category and extreme numeric values | Fitted medians, scales, vocabularies, diagnostics, and masks do not change; preprocessing tests. |
| FE-T-009 | Grouped folds with instrumented fit calls and infeasible class/group coverage | No group overlap; each fit sees fold-training rows only; infeasible protocol fails; new feature-pipeline integration tests. |
| FE-T-010 | Missing/unseen categories, all-missing numeric columns, numeric-only/categorical-only inputs | Stable fitted names and width; transformed matrix is finite and supported sparse output remains usable. |
| FE-T-011 | Pipeline saved and loaded, then predicts from raw inputs | Matching transforms, predictions and probabilities within fixed tolerance; new `tests/contract/test_feature_contract.py`. |
| FE-T-012 | Guardrail failures, near ties, simpler reference, and repeated runs | Frozen deterministic rule selects the expected variant; no automatic threshold relaxation. |
| FE-T-013 | Generated manifest and catalog | Complete lineage, ordered names, exclusions, hashes, fitting population and privacy-safe metadata. |
| FE-T-014 | Test preparation/transformation/prediction entry points instrumented to fail | Feature development succeeds without touching them; full-data benchmarking is absent. |
| FE-T-015 | Two identical runs and a run with changed source/split/config | Stable semantic outputs for identical runs; mismatch rejected or explicitly produces a new contract/run identity. |
| FE-T-016 | Inference omits target/derived columns, includes an unknown category, or loads a legacy bundle | New bundle computes derivatives internally; missing raw dependencies fail; legacy policy is honored. |
| FE-T-017 | Completed evidence and documentation review | Cutoff assumptions, test exposure, rejected variants, constraints and requirement links are explicit. |

Also cover the optional selector's all-features-removed error if that mode is implemented. Integration fixtures should be synthetic, small, and multi-client/multi-class; existing contract tests must remain valid. Do not use the real final test partition as a test fixture for this phase.

## 11. Planned execution and verification commands

The feature CLI below is a proposed deliverable; it does not exist yet. It should accept the four existing/new config files and an isolated output directory, perform the fixed development-only study, and write reports without invoking final test evaluation.

```powershell
.\.venv\Scripts\python.exe scripts\engineer_features.py --data-config configs/data.yaml --preprocessing-config configs/preprocessing.yaml --features-config configs/features.yaml --training-config configs/training.yaml --output-dir reports/features
.\.venv\Scripts\python.exe -m pytest tests/unit/test_feature_registry.py tests/unit/test_feature_engineering.py tests/unit/test_preprocessing.py tests/integration/test_feature_pipeline.py tests/contract/test_feature_contract.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

Run the feature command again to a separate output directory and compare normalized semantic JSON, ordered catalogs, and selected configurations. Recheck source SHA-256 before/after both runs. Capture actual command results during implementation; no tests or experiments are claimed as run by this planning document.

## 12. Decisions to settle before experiments

| Decision | Proposed default | Consequence if unresolved |
| --- | --- | --- |
| Raw metadata availability at prediction cutoff | Require documented availability; explicitly flag uncertain historical fields | Exclude uncertain fields from claims of cutoff-safe production readiness. |
| Undefined ratios | Missing plus an unavailable indicator | Do not silently inherit the current helper's zero-fill semantics. |
| All-missing numeric columns | Preserve output with zero fallback and record the fallback | Block a new contract release until dimensionality behavior is deterministic. |
| Selection complexity | Fixed family ablation; no supervised selector initially | Defer additional algorithms to a separately versioned study. |
| Experiment and promotion rules | Five variants, two fixed estimators, three grouped folds, thresholds in Section 7 | Freeze protocol before inspecting scores. |
| Previously evaluated test set | Record prior access and reserve untouched future data for independent final evaluation | Do not claim a fresh unbiased holdout result from current artifacts. |
| Legacy artifact compatibility | Preserve old pipeline path; require complete metadata for new bundles | Avoid silently applying new formulas to old models. |

These are implementation design decisions, not blockers to creating this plan. Approval of normative requirements should be recorded in SPEC-04 before declaring the implementation complete.

## 13. Acceptance checklist and definition of done

- [x] SPEC-04 requirements and tests replace normative placeholders; development defaults were adopted under the implementation instruction.
- [x] Every included feature has an ordered, versioned definition and cutoff rationale; historical metadata availability is explicitly unverified.
- [x] Forbidden inputs cannot enter directly or through derived-feature dependencies.
- [x] Ratios, logs, missingness, invalid values, and all-missing columns follow tested policies.
- [x] Transformations preserve rows and source data; fitted state uses training/fold-training data only.
- [x] Development diagnostics and experiments exclude test rows and outcomes.
- [x] The bounded study records reference comparisons, fold variability, guardrails, and deterministic decisions.
- [x] Validation confirmation is documented without iterative holdout-driven feature search.
- [x] Training/inference transformations and serialization round trips agree.
- [x] Ordered raw/derived/encoded/selected feature names and lineage are auditable.
- [x] Semantic artifacts reproduce and match source/split/config identities.
- [x] Reports contain no raw sensitive identifiers or persisted row-level feature matrices.
- [x] Existing test exposure and snapshot limitations are stated accurately.
- [x] Relevant tests, repository regression suite, and lint pass with recorded evidence.
- [x] SPEC-05/06 receive frozen feature configuration and evidence, even if the reference remains the preferred input set.

Implementation is complete when these items are verified, all deviations have explicit rationale, and the saved contract can be reused without repeating exploratory decisions. Improved test-set performance is not a completion criterion for SPEC-04.

Production readiness still requires historical cutoff evidence and the final-test audit policy. These limitations do not imply a new untouched holdout was created.
