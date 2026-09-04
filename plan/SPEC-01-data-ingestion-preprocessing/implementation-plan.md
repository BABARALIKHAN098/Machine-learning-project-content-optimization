# Implementation Plan — CSV Ingestion, Review, and Preprocessing

**Source specification:** `specs/SPEC-01-data-ingestion-preprocessing.md`  
**Plan status:** Implemented — automated acceptance gates pass; policy decisions remain  
**Owner:** Babar Ali Khan  
**Dataset:** `data/raw/dataset.csv`  
**Target:** `trend_direction`  
**Feature contract:** `configs/data.yaml` and `configs/preprocessing.yaml`

## 1. Objective

Deliver a deterministic, configuration-driven data preparation boundary that:

1. loads the approved CSV without changing it;
2. calculates and retains a source fingerprint;
3. validates the dataset and the column-role contract;
4. produces reproducible structural quality reports;
5. separates target, identifiers, excluded fields, and approved features explicitly;
6. builds an unfitted preprocessing pipeline;
7. guarantees that learned preprocessing state is fitted only on training data; and
8. transforms training and inference data consistently without silent row or feature loss.

The phase ends at validated, leakage-safe preprocessing. Dataset splitting policy belongs to SPEC-03, while model fitting and evaluation belong to later specifications.

## 2. Current-State Assessment

The repository already implements much of SPEC-01:

| Capability | Current location | Status |
| --- | --- | --- |
| Configured CSV loading | `data/ingestion.py` | Implemented |
| Empty/missing file checks | `data/ingestion.py` | Partially tested |
| SHA-256 fingerprint | `data/ingestion.py` | Implemented |
| Schema and role checks | `data/validation.py` | Partially implemented |
| Structural profile | `data/profiling.py` | Implemented |
| Human-readable audit | `scripts/validate_data.py` | Implemented |
| Central feature allowlist | `features/selection.py` | Implemented |
| Numeric coercion checks | `features/selection.py` | Implemented |
| Numeric/categorical preprocessing | `features/preprocessing.py` | Implemented |
| Unknown category handling | `features/preprocessing.py` | Implemented and tested |
| Train-only fitting | `pipelines/training_pipeline.py` | Implemented, insufficiently isolated/tested |
| Complete acceptance coverage | `tests/` | Incomplete |

Important gaps to close:

- malformed, unreadable, header-only, single-column, blank-header, and normalized duplicate-header inputs lack complete tests;
- configured role columns are not all checked for existence independently of `required_columns`;
- unexpected extra columns can remain unassigned unless all role lists are present, and extra-column policy is not explicit;
- target null handling and whether rows may be rejected are not formally defined;
- preprocessing configuration values are documented but the builder currently hard-codes strategies except for scaling;
- preprocessing is not exposed as a first-class unfitted artifact/result from the data phase;
- train-only fit behavior, feature-name output, deterministic ordering, input immutability, and row preservation need stronger tests;
- reports do not yet record configuration/contract versions or explicit row/column disposition.

## 3. Scope Boundaries

### In scope

- CSV path, delimiter, encoding, and missing-token handling.
- File existence, readability, non-empty content, header integrity, and parsing failures.
- SHA-256 calculation from the original source bytes.
- Schema, target, identifier, uniqueness, role, type, and configured domain validation.
- Structural profiling: shape, types, missingness, duplicates, constants, cardinality, and memory usage.
- Explicit target extraction and approved feature selection.
- Numeric coercion with errors for non-null invalid values.
- Construction of an unfitted `ColumnTransformer` for configured numeric and categorical features.
- Median numeric imputation, constant categorical imputation, unknown-category-safe one-hot encoding, and optional numeric scaling.
- Fit-on-train-only enforcement and stable transformed feature names.
- Machine-readable and human-readable validation artifacts.

### Out of scope

- Exploratory relationships and visualization beyond the structural audit (SPEC-02).
- Group-aware train/validation/test assignment (SPEC-03).
- New feature derivation and feature selection experiments (SPEC-04).
- Model training, tuning, evaluation, and inference packaging.
- Automatic data correction, source rewriting, implicit row dropping, or deduplication.

## 4. Design Decisions

### 4.1 Raw-data immutability

- Open the source read-only.
- Calculate SHA-256 from file bytes, not from a parsed dataframe.
- Recalculate the fingerprint at the end of the validation workflow and fail if it changed.
- Write generated output only beneath configured report/artifact locations.
- Never overwrite, normalize, or export back to the raw source path.

### 4.2 Fail-fast versus reportable findings

Fail ingestion or validation for issues that make the contract unreliable:

- missing, empty, unreadable, or malformed source;
- header-only input or fewer than two parsed columns;
- blank or duplicate normalized column names;
- missing required, target, ID, role, or approved feature columns;
- unknown target values, unusable target, duplicate required-unique identifiers;
- conflicting or unassigned column roles;
- non-null, non-numeric tokens in configured numeric features;
- forbidden columns configured as features.

Report without modifying data:

- ordinary missing values covered by an approved preprocessing strategy;
- duplicate full rows unless configured as a hard failure;
- constants and high-cardinality columns;
- high missingness and unusual but valid distributions.

### 4.3 Column-role contract

Every source column must have exactly one primary role:

- `target_column`;
- `id_columns` (including grouping identifiers);
- `numeric_columns`;
- `categorical_columns`; or
- `drop_columns` for excluded, sensitive-only, unsupported, or leakage fields.

Sensitive classification is an orthogonal annotation and may overlap an ID or dropped role. It must not be treated as a conflicting primary role. Unknown extra columns must fail validation by default so newly introduced fields cannot enter processing silently.

### 4.4 Target and feature separation

Create a single preparation function that returns a typed result containing:

- `features`: a defensive copy in configured numeric-then-categorical order;
- `target`: a defensive copy with its original index;
- optional identifiers/groups for downstream splitting and traceability;
- source fingerprint and feature-contract version;
- a disposition manifest for every input column.

The target, IDs, and dropped/leakage fields must never reach the transformer. Target extraction must occur before fitting preprocessing.

### 4.5 Preprocessing lifecycle

- `build_preprocessor` returns an unfitted transformer.
- Numeric and categorical strategies come from validated preprocessing configuration.
- Only the training partition may call `fit` or `fit_transform`.
- Validation, test, and inference partitions may call `transform` only.
- Median statistics and categorical vocabularies must therefore reflect training data exclusively.
- Unknown inference categories are ignored by one-hot encoding without changing learned output width.
- The transformer must expose stable, unique feature names after fitting.
- Transformations must preserve input row count and order.

## 5. Delivery Phases

### Phase 0 — Approve the data contract

**Files:** `specs/SPEC-01-data-ingestion-preprocessing.md`, `configs/data.yaml`, `configs/preprocessing.yaml`.

Tasks:

- Confirm the canonical source path, encoding, delimiter, and missing tokens.
- Confirm the expected 30,000 rows, 44 columns, five target labels, and unique `content_id` contract.
- Approve every column's role and leakage status.
- Decide target-null behavior; proposed policy is hard failure rather than silent row removal.
- Decide whether duplicate full rows are fatal or report-only.
- Confirm `provider_used` exclusion due to current missingness above 60%.
- Confirm whether strict rejection of unexpected columns is required; proposed default is yes.
- Record the approval date and contract version.

Exit criteria:

- Every column has one approved primary role.
- No unresolved ingestion or preprocessing policy remains undocumented.

### Phase 1 — Harden configuration validation

**Files:** `src/machine_learning_project/utils/config.py`, `src/machine_learning_project/data/validation.py`, `configs/*.yaml`.

Tasks:

- Validate that required configuration sections and keys exist.
- Validate types and supported values for delimiter, encoding, missing tokens, imputation strategies, encoder behavior, and scaling.
- Reject duplicate entries within role lists and conflicts across primary roles.
- Verify that target, ID, drop, numeric, categorical, unique, non-negative, and sensitive references point to known source columns.
- Add explicit `allow_extra_columns: false` and contract/schema versions.
- Ensure sensitive roles can overlap primary roles without false conflict errors.
- Produce ordered, actionable errors containing the configuration key and offending columns.

Exit criteria:

- Invalid configuration fails before preprocessing is built or fitted.

### Phase 2 — Harden ingestion

**Files:** `src/machine_learning_project/data/ingestion.py`, `tests/unit/test_ingestion.py`.

Tasks:

- Retain byte-level SHA-256 and immutable `DataLoadResult`.
- Wrap decode, parser, permissions, and malformed-file failures in the project's data exception type while preserving the cause.
- Clearly distinguish missing, zero-byte, header-only, malformed, unreadable, and structurally invalid CSVs.
- Normalize surrounding whitespace in headers once and reject collisions caused by normalization.
- Make pandas parsing behavior explicit where needed, including missing-token handling.
- Avoid logging row content or identifiers in errors.
- Verify the path points to a regular file.

Exit criteria:

- Every ING-001/002 failure mode returns a stable, useful error.
- The fingerprint always refers to original bytes.

### Phase 3 — Complete schema and data validation

**Files:** `src/machine_learning_project/data/validation.py`, `tests/unit/test_validation.py`.

Tasks:

- Validate expected shape, required columns, target vocabulary, target nulls, and target class count.
- Validate identifier presence, null policy, and configured uniqueness.
- Validate complete and non-conflicting primary role assignment.
- Validate configured numeric convertibility before selection.
- Retain non-negative validation and add approved finite/range rules when defined.
- Report all independent schema errors in one deterministic list where safe.
- Keep hard errors separate from audit warnings.
- Add a structured validation result if warnings need to be emitted alongside errors.

Exit criteria:

- Every input column is accounted for and invalid data cannot proceed to preprocessing.

### Phase 4 — Expand reproducible review artifacts

**Files:** `src/machine_learning_project/data/profiling.py`, `scripts/validate_data.py`, `pipelines/data_pipeline.py`, `reports/`.

Tasks:

- Preserve existing shape, dtype, missingness, duplicate, constant, cardinality, memory, and hash outputs.
- Add schema/profile version, feature-contract version, role, sensitive flag, and model eligibility per column.
- Add a column-disposition section proving that target, IDs, dropped fields, and selected features are accounted for.
- Record warnings for high missingness and high cardinality using configured thresholds.
- Sort fields deterministically and isolate volatile timestamps if added.
- Recheck the source hash after report generation.
- Write JSON and Markdown atomically so failed runs do not leave misleading complete artifacts.

Outputs:

- `reports/data_profile.json` — versioned machine-readable profile;
- `reports/data_audit.md` — human-readable review and risk summary.

Exit criteria:

- Repeated runs against identical source/config produce identical substantive profiles.
- Report totals reconcile exactly with the input schema and row count.

### Phase 5 — Formalize dataset preparation

**Files:** `src/machine_learning_project/features/selection.py`, optionally `src/machine_learning_project/data/preparation.py`, `tests/unit/test_preprocessing.py`.

Tasks:

- Add an explicit target-extraction/preparation API rather than repeating dataframe slicing in training code.
- Return defensive copies and preserve indexes.
- Coerce configured numeric columns with explicit invalid-value reporting.
- Select features exclusively through the central allowlist and stable configured order.
- Return identifiers/groups separately when requested; never include them in features.
- Emit a column-disposition manifest with selected/excluded reason.
- Assert row count is unchanged during selection.

Exit criteria:

- Target, IDs, dropped/leakage fields, and approved features cannot be confused at call sites.
- Selection performs no silent row removal or source mutation.

### Phase 6 — Make preprocessing fully configuration-driven

**Files:** `src/machine_learning_project/features/preprocessing.py`, `configs/preprocessing.yaml`.

Tasks:

- Accept and validate the full preprocessing configuration rather than only `scale_numeric`.
- Map approved `median` numeric imputation and `__MISSING__` categorical imputation into sklearn pipelines.
- Configure one-hot encoding with `handle_unknown="ignore"` and stable sparse output.
- Support optional scaling without changing selection semantics.
- Validate empty numeric-only or categorical-only feature groups gracefully.
- Prevent preprocessing of zero approved features.
- Expose transformed feature names and detect duplicate names.
- Keep the returned object unfitted and free of dataset-dependent state.

Exit criteria:

- Configuration and actual transformer behavior cannot diverge silently.

### Phase 7 — Enforce train-only fitting in orchestration

**Files:** `pipelines/training_pipeline.py`, `scripts/preprocess_data.py`, relevant SPEC-03 interfaces.

Tasks:

- Split data before any learned preprocessing operation.
- Extract training features/target, build the transformer, and fit only with training features.
- Transform validation and test data using the same fitted transformer.
- Add runtime guards or a narrow pipeline API that makes accidental full-dataset fitting difficult.
- Keep the current training pipeline's end-to-end sklearn pipeline pattern.
- Update `scripts/preprocess_data.py` so its name and behavior match: either build/verify preprocessing or rename it to describe split inspection.
- Never persist separately transformed raw partitions unless a later approved contract requires them.

Exit criteria:

- Tests prove validation/test values cannot influence imputation statistics or category vocabulary.

### Phase 8 — Complete automated verification

**Files:** `tests/unit/`, `tests/integration/test_data_pipeline.py`, optional `tests/contract/`.

Tasks:

- Implement the test matrix below.
- Run all existing unit, integration, and contract tests.
- Run Ruff and fix only task-related violations.
- Add an end-to-end test using temporary source and output paths.
- Assert generated reports contain no raw sensitive values beyond approved aggregate metadata.

Exit criteria:

- All SPEC-01 tests and existing regression checks pass from a clean checkout.

### Phase 9 — Documentation and approval

**Files:** `readme.md`, `specs/SPEC-01-data-ingestion-preprocessing.md`, generated audit artifacts.

Tasks:

- Document validation and preprocessing commands, inputs, outputs, and failure behavior.
- Explain column roles, leakage exclusions, missing-value strategies, and train-only fitting.
- Record current dataset fingerprint and known quality exceptions.
- Check all acceptance criteria and change SPEC-01 status only after evidence is linked.

Exit criteria:

- Another developer can reproduce validation and construct the preprocessing pipeline without undocumented steps.

## 6. File-Level Work Breakdown

| File or area | Planned change |
| --- | --- |
| `specs/SPEC-01-data-ingestion-preprocessing.md` | Record approved policies, expanded tests, evidence, and final status. |
| `configs/data.yaml` | Version schema; finalize roles, strict extras, identifier/target null rules, and quality thresholds. |
| `configs/preprocessing.yaml` | Define validated preprocessing strategies and contract version. |
| `utils/config.py` | Add section/key/type/value validation or dedicated config validators. |
| `data/ingestion.py` | Normalize ingestion errors and cover all malformed/unreadable cases. |
| `data/validation.py` | Complete schema, roles, references, target, ID, and domain validation. |
| `data/profiling.py` | Add role/disposition, warnings, and schema-version metadata. |
| `data/preparation.py` or `features/selection.py` | Return explicit features, target, IDs, and disposition without mutation. |
| `features/preprocessing.py` | Build an unfitted, fully configuration-driven transformer. |
| `pipelines/data_pipeline.py` | Orchestrate loading, validation, profile creation, and hash recheck. |
| `pipelines/training_pipeline.py` | Consume the preparation contract and preserve train-only fitting. |
| `scripts/validate_data.py` | Add atomic, reproducible report generation and stable failures. |
| `scripts/preprocess_data.py` | Align command behavior with its name and verify preprocessing lifecycle. |
| `tests/unit/` | Cover ingestion, validation, selection, configuration, and transformer edge cases. |
| `tests/integration/` | Cover end-to-end artifacts, immutability, and train-only preprocessing. |
| `readme.md` | Document commands, contracts, artifacts, and troubleshooting. |

## 7. Test Matrix and Traceability

| Test ID | Requirement | Given | When | Then |
| --- | --- | --- | --- | --- |
| T-ING-001 | ING-001/003 | Valid configured CSV | Loading runs | Shape, normalized columns, source path, and 64-character SHA-256 are returned. |
| T-ING-002 | ING-001 | Custom delimiter, encoding, and missing tokens | Loading runs | Values are parsed according to configuration. |
| T-ING-003 | ING-002 | Missing file | Loading runs | A stable actionable error identifies the path. |
| T-ING-004 | ING-002 | Zero-byte file | Loading runs | It is rejected as empty. |
| T-ING-005 | ING-002 | Header-only CSV | Loading runs | It is rejected for having no data rows. |
| T-ING-006 | ING-002 | Malformed or undecodable CSV | Loading runs | A project data error preserves the cause without exposing rows. |
| T-ING-007 | ING-002 | Blank, duplicate, or normalization-colliding headers | Loading runs | It is rejected with the exact header problem. |
| T-ING-008 | ING-003 | Known file bytes | Hashing runs | SHA-256 equals the known digest and changes when bytes change. |
| T-REV-001 | REV-001 | Small known dataframe | Profiling runs | Shape, types, missingness, duplicates, constants, and cardinality are exact. |
| T-REV-002 | REV-001 | Same input/config twice | Profiling runs | Non-volatile output and ordering match. |
| T-REV-003 | REV-002 | Missing role/reference columns | Validation runs | All missing configured references are reported deterministically. |
| T-REV-004 | REV-002 | Conflicting or unassigned primary roles | Validation runs | Analysis fails with the conflicting/unassigned names. |
| T-REV-005 | REV-002 | Unknown target, null target, duplicate ID, or invalid numeric value | Validation runs | Each contract violation is reported. |
| T-REV-006 | REV-001/002 | High-missingness, constant, or high-cardinality fields | Review runs | Warnings are emitted without modifying data. |
| T-PRE-001 | PRE-001/002 | Frame containing target, IDs, approved and dropped fields | Preparation runs | Features contain only approved columns; target and IDs are returned separately. |
| T-PRE-002 | PRE-002 | Forbidden field configured as numeric/categorical | Contract builds | It fails before transformation. |
| T-PRE-003 | PRE-003 | Numeric/categorical missing values | Fitted training transformation runs | No unexpected missing values remain. |
| T-PRE-004 | PRE-004 | Unseen inference category | Transform runs | It succeeds with unchanged output width. |
| T-PRE-005 | PRE-005 | Extreme validation values and unseen categories | Transformer fits on training only | Learned medians/categories equal training-only values. |
| T-PRE-006 | PRE-003/004 | Same fitted transformer and same input | Transform runs twice | Values, shape, row order, and feature names match. |
| T-PRE-007 | PRE-001/002 | Source dataframe | Selection/transformation runs | Source values, columns, index, and row count are unchanged. |
| T-PRE-008 | PRE-003 | Numeric token that cannot be coerced | Preparation runs | It fails instead of converting the value silently to missing. |
| T-PRE-009 | PRE-003 | Numeric-only or categorical-only approved schema | Preprocessor builds and fits | It operates without an empty-branch failure. |
| T-INT-001 | All | Temporary valid CSV and configs | Data pipeline runs | Versioned JSON/Markdown reports are created with matching fingerprint. |
| T-INT-002 | ING-003/acceptance | Existing source | Full phase runs | Source bytes and fingerprint remain unchanged. |
| T-INT-003 | Acceptance | Complete input schema | Disposition is generated | Every column is counted exactly once as target, ID, feature, or dropped. |
| T-INT-004 | Acceptance | Deliberately failed report write/run | Pipeline exits | Partial output is not presented as complete. |

## 8. Requirements-to-Implementation Map

| Requirement | Primary implementation | Verification |
| --- | --- | --- |
| ING-001 | `load_csv`, validated data config | T-ING-001/002 |
| ING-002 | ingestion guards and normalized exceptions | T-ING-003 through T-ING-007 |
| ING-003 | `sha256_file`, pipeline recheck | T-ING-008, T-INT-002 |
| REV-001 | `build_profile`, Markdown renderer | T-REV-001/002/006 |
| REV-002 | config/schema/role validation | T-REV-003 through T-REV-005 |
| PRE-001 | explicit preparation result | T-PRE-001, T-PRE-007 |
| PRE-002 | central allowlist and forbidden-role checks | T-PRE-001/002, T-INT-003 |
| PRE-003 | configured sklearn pipelines | T-PRE-003/008/009 |
| PRE-004 | unknown-safe encoder | T-PRE-004/006 |
| PRE-005 | split-before-fit orchestration and fit-state tests | T-PRE-005, T-INT-001 |

## 9. Acceptance Evidence

| Acceptance criterion | Required evidence |
| --- | --- |
| Raw CSV remains unchanged | Before/after byte hash assertion and integration test. |
| Fingerprint/profile reproducible | Two-run comparison plus stored `source_sha256`. |
| Every column has approved role | Versioned disposition manifest with exact reconciliation. |
| No silent row/column removal | Row/index preservation tests and explicit disposition reasons. |
| Target/preprocessing leakage prevented | Forbidden-role tests and train-only learned-statistic test. |
| Mandatory tests pass | Clean pytest and Ruff results recorded in handoff/CI. |

## 10. Recommended Implementation Order

1. Approve the column roles and unresolved policies.
2. Add contract versions and strict configuration validation.
3. Complete ingestion error handling and tests.
4. Complete schema, target, ID, and role validation.
5. Expand versioned review artifacts and hash verification.
6. Formalize feature/target/identifier preparation.
7. Make transformer construction fully configuration-driven.
8. Add train-only fitting guards and lifecycle tests.
9. Add end-to-end reproducibility and immutability tests.
10. Update documentation, generate evidence, and approve SPEC-01.

## 11. Definition of Done

SPEC-01 is complete only when:

- the data and preprocessing contracts are approved and versioned;
- all configured valid CSV behavior and failure modes are tested;
- source bytes are proven unchanged across the workflow;
- the profile and audit are reproducible and carry the source fingerprint;
- every input column has one explicit disposition;
- target, identifiers, and leakage fields cannot enter preprocessing;
- numeric and categorical missingness is handled exactly as configured;
- unseen inference categories transform without failure or output-width changes;
- validation/test data cannot influence learned preprocessing state;
- row count, row order, input values, and approved feature ordering are preserved;
- all SPEC-01 unit/integration tests and existing regression checks pass;
- the README provides a reproducible command and explains artifacts and limitations.

## 12. Open Decisions

- Whether null target values fail the entire dataset or are quarantined through an explicitly approved process; silent dropping is prohibited.
- Whether duplicate full rows are fatal or report-only.
- Whether unexpected extra columns always fail or can be allowed in a named compatibility mode.
- Whether expected row/column counts are hard production constraints or warnings for new dataset versions.
- Whether `provider_used` remains excluded because its observed missingness is approximately 71.46%.
- Whether numeric scaling defaults to enabled for all downstream models or is selected per estimator.
- Whether structural audit artifacts are committed or generated in CI.
- Whether `scripts/preprocess_data.py` should build/inspect preprocessing or be renamed to reflect its current split-summary behavior.
