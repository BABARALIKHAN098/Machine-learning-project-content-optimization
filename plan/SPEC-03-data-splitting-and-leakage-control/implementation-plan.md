# Implementation Plan — Data Splitting and Leakage Control

**Source specification:** `specs/SPEC-03-data-splitting-and-leakage-control.md`  
**Plan status:** In progress — foundational split and artifact controls implemented  
**Owner:** Babar Ali Khan  
**Dataset:** `data/raw/dataset.csv`  
**Target:** `trend_direction`  
**Grouping unit:** `client_id`  
**Default split:** 60% train / 20% validation / 20% test by target row proportions, subject to whole-group assignment

## 1. Objective

Create reproducible train, validation, and test partitions that prevent the same client from appearing in multiple partitions, preserve every eligible source row exactly once, maintain usable target-class coverage, and keep the final test partition isolated from preprocessing, feature selection, model selection, and tuning.

The output of this phase is a deterministic partition assignment plus a versioned, privacy-safe manifest. It does not include fitted preprocessing, model training, threshold tuning, or final test evaluation.

## 2. Current-State Assessment

The repository already contains a working grouped split:

| Capability | Current location | Status |
| --- | --- | --- |
| Group-aware split | `data/splitting.py` | Implemented with `GroupShuffleSplit` |
| Train/validation/test partitions | `data/splitting.py` | Implemented |
| Deterministic seed | `configs/training.yaml` | Implemented |
| Pairwise group-disjoint assertion | `data/splitting.py` | Implemented |
| Full row-count assertion | `data/splitting.py` | Implemented, but not an exact identity proof |
| Every target class in every partition | `_best_group_split` | Implemented |
| Distribution-aware candidate selection | `_best_group_split` | Implemented |
| Split manifest | `artifacts/metadata/split_manifest.json` | Implemented |
| Basic determinism/disjointness test | `tests/unit/test_splitting.py` | Implemented |
| Split-before-preprocessing | `training_pipeline.py` | Implemented |

Current dataset result:

| Partition | Rows | Clients |
| --- | ---: | ---: |
| Train | 18,246 | 18 |
| Validation | 5,281 | 7 |
| Test | 6,473 | 7 |

Important gaps:

- SPEC-03 has no approved normative requirements or test contract.
- The manifest is not versioned or bound to the source SHA-256, schema version, code/config fingerprint, or split algorithm version.
- Row totals alone cannot prove that no row was duplicated while another was omitted.
- Raw group values are stored in the manifest; even pseudonymous client strings should be masked consistently.
- Split-size and class-distribution deviations are optimized but not measured against approved tolerances.
- The number of search attempts is hard-coded and candidate tie-breaking is not documented.
- Insufficient groups, null targets, duplicate row keys, single-group inputs, and infeasible class/group structures lack complete tests.
- There is no explicit test-set access policy or auditable state showing when the test partition is first evaluated.
- Feature leakage is controlled elsewhere, but split orchestration does not reassert the feature contract before handing data downstream.
- The dataset has no timestamp column, so true forward-looking temporal validation cannot be performed.

## 3. Scope

### In scope

- Validation of target, group, row-key, split ratio, seed, and feasibility prerequisites.
- Deterministic whole-client assignment to train, validation, and test.
- Complete, non-overlapping row and group coverage.
- Target-class coverage and quantified distribution deviation.
- A versioned, source-bound, privacy-safe split manifest.
- Central leakage assertions at the split-to-feature boundary.
- A sealed-test access contract and evaluation audit metadata.
- Unit, property-style, integration, determinism, and regression tests.
- Clear limitations for snapshot-only data and future temporal splitting.

### Out of scope

- Data cleaning, imputation, encoding, or scaling (SPEC-01).
- Exploratory analysis and anomaly discovery (SPEC-02).
- Feature creation or experimental feature selection (SPEC-04).
- Model training/tuning and final performance interpretation.
- Synthetic oversampling or class balancing before partition creation.
- Prospective temporal validation until timestamped snapshots exist.

## 4. Proposed Requirements

| ID | Requirement |
| --- | --- |
| SPLIT-001 | Read the target, group column, row key, ratios, seed, attempt count, and tolerances from validated configuration. |
| SPLIT-002 | Reject missing target/group columns, null target/group values, duplicate/null row keys, invalid ratios, and infeasible group structures with actionable errors. |
| SPLIT-003 | Assign each group wholly to exactly one of train, validation, or test. |
| SPLIT-004 | Assign every eligible input row to exactly one partition without duplication, omission, or mutation. |
| SPLIT-005 | Produce identical assignments for the same source, configuration, algorithm version, and seed. |
| SPLIT-006 | Require every configured target class in every partition unless an explicitly approved exception policy applies. |
| SPLIT-007 | Optimize candidate assignments against row-size and class-distribution objectives and report deviations from configured tolerances. |
| SPLIT-008 | Fail clearly when no feasible split is found within the configured deterministic search budget. |
| SPLIT-009 | Bind the manifest to source SHA-256, data schema, split contract, algorithm version, parameters, and creation provenance. |
| SPLIT-010 | Store privacy-safe group aliases or hashes rather than raw sensitive group values. |
| SPLIT-011 | Store per-partition row/group/class counts, proportions, deviations, and stable assignment fingerprints. |
| SPLIT-012 | Reassert that target, identifiers, dropped fields, and leakage fields are absent from downstream feature matrices. |
| SPLIT-013 | Fit learned preprocessing only on training data and transform validation/test without fitting. |
| SPLIT-014 | Prevent validation/test records from influencing feature selection, imputation, encoding vocabularies, scaling, resampling, or model selection. |
| SPLIT-015 | Keep the test partition sealed until one model and decision policy have been selected from development data. |
| SPLIT-016 | Record the first authorized test evaluation and prohibit iterative tuning against its results. |
| SPLIT-017 | Document that client-grouped snapshot validation estimates unseen-client generalization, not future-period performance. |

## 5. Split Contract

### 5.1 Inputs

The split function accepts:

- the validated dataframe;
- `target_column` (`trend_direction`);
- `group_column` (`client_id`);
- stable `row_key` (`content_id`);
- configured allowed labels;
- `test_size`, `validation_size`, and implied train size;
- `random_seed`;
- deterministic `search_attempts`;
- row-size and class-distribution tolerances;
- split-contract and algorithm versions;
- source SHA-256 and data-schema version.

The API must not depend on global random state.

### 5.2 Preconditions

Validate before searching:

- all required split columns exist;
- target, group, and row key contain no nulls;
- row keys are unique;
- target values match the approved label set;
- ratios are numeric, strictly between zero and one, and sum to less than one;
- at least three distinct groups exist;
- each target class occurs in enough distinct groups to appear in three partitions;
- the requested grouping unit is explicitly approved and marked sensitive where required.

Return all independent feasibility problems together where practical.

### 5.3 Assignment algorithm

Retain deterministic repeated group-shuffle search for the initial implementation:

1. Generate whole-group development/test candidates using `seed + attempt_offset`.
2. Reject candidates missing any configured target class.
3. Split development into train/validation with a non-overlapping seed range.
4. Reject candidates that violate group/row coverage.
5. Score feasible candidates using a documented weighted objective:
   - absolute row-ratio deviation;
   - total absolute class-proportion deviation from the full dataset;
   - optional group-count imbalance penalty.
6. Select the lowest score, breaking exact ties through stable sorted assignment fingerprints.

The two-stage algorithm cannot guarantee exact row proportions because groups vary in size. The report must show actual ratios and deviation rather than claiming exact 60/20/20 allocation.

If the approved tolerances cannot be met, fail with the best observed deviations and recommended actions rather than silently accepting a poor split.

### 5.4 Output objects

Return immutable split metadata plus defensive dataframe copies:

- `train`, `validation`, `test` frames preserving original columns and row indexes;
- a long-form assignment table containing safe row-key fingerprints and partition names;
- a manifest containing provenance, parameters, distributions, warnings, and fingerprints;
- a validation result showing all invariants passed.

Do not persist transformed partitions in this phase. Downstream consumers should apply the manifest/assignment to the approved source version.

## 6. Leakage Controls

### 6.1 Group leakage

- Build group sets for all partitions and assert pairwise disjointness.
- Assert their union exactly equals the set of source groups.
- Use stable salted/project-scoped aliases in artifacts; retain raw group values only in memory.
- Never expose `client_id` as a model feature.

### 6.2 Row leakage and coverage

- Require a unique row key before splitting.
- Assert partition row-key sets are pairwise disjoint.
- Assert their union equals the source row-key set.
- Assert partition concatenation retains exact source row count and no row-key duplication.
- Create stable per-partition assignment hashes from sorted row keys and group aliases.

This replaces the current weaker check that compares only total row counts.

### 6.3 Target and feature leakage

At each downstream boundary:

- separate target before preprocessing;
- select only centrally approved numeric/categorical features;
- assert target, IDs, sensitive-only fields, and `drop_columns` are absent;
- retain the feature-contract version in the split/training metadata;
- fail on unknown newly introduced columns.

### 6.4 Preprocessing leakage

- Split raw validated rows before building learned transformations.
- Call `fit` or `fit_transform` only with training features.
- Use only `transform` for validation and test.
- Prove in tests that extreme validation/test values do not affect medians/scalers and unseen categories do not enter the fitted vocabulary.
- Do not perform global oversampling, normalization, target encoding, or feature selection before splitting.

### 6.5 Test-set sealing

Introduce a procedural and metadata contract:

- development work may access train and validation only;
- candidate selection and decision thresholds must be frozen before test access;
- the final evaluation command records model/artifact fingerprint, split-manifest fingerprint, timestamp, reason, and evaluation count;
- test results must not feed another tuning cycle on the same holdout;
- if the test set is used iteratively, mark it contaminated and create a new untouched holdout when data permits.

Code cannot make intentional leakage impossible, so documentation, separate commands, artifact provenance, and review gates must complement automated assertions.

### 6.6 Temporal limitation

The current CSV is a single snapshot without an approved snapshot timestamp. Client-grouped splitting tests generalization to unseen clients, but it does not demonstrate future-period prediction.

For production validation, require:

- timestamped feature snapshots;
- an explicit prediction cutoff;
- labels calculated from a later outcome window;
- optional temporal gap/embargo between features and outcomes;
- group-aware temporal splitting if unseen-client and future-period generalization both matter.

## 7. Manifest Contract

Proposed `artifacts/metadata/split_manifest.json` structure:

| Section | Required fields |
| --- | --- |
| Schema | `manifest_schema_version`, `split_contract_version`, `algorithm_version` |
| Source | `source_sha256`, `data_schema_version`, row count, group count, target labels |
| Parameters | target, group, row key, ratios, seed, attempts, objective weights, tolerances |
| Partition summary | rows, actual ratio, ratio deviation, group count, class counts/proportions/deviations |
| Privacy-safe assignment | group aliases/hashes and partition assignment fingerprint |
| Invariants | row completeness, row disjointness, group completeness, group disjointness, class coverage |
| Limitations | lack of temporal validation and any tolerance exceptions |

Also create `artifacts/metadata/split_assignments.csv` with:

- stable one-way row-key fingerprint;
- masked group alias;
- partition;
- split-contract version.

Raw `content_id` and `client_id` values should not be committed in metadata artifacts.

## 8. Delivery Phases

### Phase 0 — Approve the split contract

**Files:** `specs/SPEC-03-data-splitting-and-leakage-control.md`, this plan.

Tasks:

- Confirm `client_id` is the correct independence/generalization unit.
- Confirm `content_id` is the stable unique row key.
- Approve target class coverage requirements.
- Approve whether ratios are targets by rows, groups, or a weighted compromise; proposed choice is rows with whole-group constraints.
- Approve size/class tolerances and the policy when no candidate meets them.
- Approve test-set access and contamination policy.
- Confirm privacy rules for assignment metadata.

Exit criteria:

- No normative `TBD` remains in SPEC-03.
- Split objectives and exceptions are measurable and approved.

### Phase 1 — Configuration and validation

**Files:** `configs/training.yaml`, `src/machine_learning_project/utils/config.py`, `data/splitting.py`.

Tasks:

- Add `split_contract_version`, `split_algorithm_version`, `row_key`, `search_attempts`, tolerance, objective-weight, alias-salt/environment-key, and manifest paths.
- Validate ratio types/ranges, total ratio, positive attempts, non-negative weights, tolerances, and required metadata.
- Validate dataset feasibility by distinct group counts per class.
- Return actionable errors listing deficient classes/groups.

Exit criteria:

- Invalid or infeasible configuration fails before candidate generation.

### Phase 2 — Deterministic assignment and quality scoring

**Files:** `src/machine_learning_project/data/splitting.py`.

Tasks:

- Extend the API with row key, labels, source metadata, attempts, and tolerances.
- Generate full three-way candidate assignments deterministically.
- Calculate row-ratio, group-count, and class-distribution deviations.
- Add stable tie-breaking independent of incidental dataframe ordering.
- Fail with best-observed diagnostics when no assignment meets hard requirements.
- Preserve input order within partitions or document deterministic canonical ordering.

Exit criteria:

- Same source/config/seed yields identical row assignments and hashes, even if input row order changes when the row key is stable.

### Phase 3 — Strong invariants and privacy-safe manifest

**Files:** `data/splitting.py`, optionally `data/split_manifest.py`, `artifacts/metadata/`.

Tasks:

- Add exact row-key and group set invariants.
- Create stable project-scoped aliases/hashes without writing raw identifiers.
- Version and bind the manifest to source/config/algorithm.
- Add actual ratios, deviations, distributions, invariant results, and warnings.
- Create the privacy-safe assignment CSV.
- Write manifest and assignments atomically.

Exit criteria:

- Every source row and group is provably assigned once.
- Published metadata contains no raw sensitive identifiers.

### Phase 4 — Downstream leakage boundaries

**Files:** `pipelines/training_pipeline.py`, `features/selection.py`, `data/preparation.py`, `scripts/preprocess_data.py`.

Tasks:

- Consume the versioned split result and central preparation contract.
- Assert feature exclusions independently for each partition.
- Fit preprocessing only on train during candidate development.
- Ensure validation drives candidate selection while test remains inaccessible.
- Refactor final test evaluation into a separate explicit command or guarded phase so training cannot inspect it during model selection.
- Bind preprocessing/model metadata to split-manifest and feature-contract fingerprints.

Exit criteria:

- A normal development command can train/select without loading test features or labels.

### Phase 5 — Test evaluation audit

**Files:** `scripts/evaluate_model.py`, `pipelines/`, `artifacts/metadata/test_evaluation.json`.

Tasks:

- Require a frozen selected-model artifact and split manifest before final evaluation.
- Verify source, feature-contract, and assignment fingerprints match.
- Record first access, evaluator command/version, artifact hash, reason, and evaluation count.
- Warn or fail on repeated test evaluation according to approved policy.
- Document how to invalidate/replace a contaminated test partition.

Exit criteria:

- Final test use is explicit, attributable, and cannot silently become a tuning loop.

### Phase 6 — Automated verification

**Files:** `tests/unit/test_splitting.py`, `tests/integration/test_data_pipeline.py`, `tests/integration/test_training_pipeline.py`, `tests/contract/`.

Tasks:

- Implement the test matrix below.
- Add randomized/property-style fixtures spanning group sizes and class distributions without adding nondeterministic tests.
- Verify manifest schema and privacy.
- Verify split-before-fit and test-sealing interfaces.
- Run the complete pytest and Ruff suites.

Exit criteria:

- All SPEC-03 and existing regression tests pass.

### Phase 7 — Generate, review, and approve

**Files:** `artifacts/metadata/split_manifest.json`, `split_assignments.csv`, `reports/`, SPEC-03, README.

Tasks:

- Regenerate the split from the approved source/config.
- Review actual row ratios, class deviations, and group allocation.
- Confirm no raw client/content identifiers appear in committed metadata.
- Document snapshot/temporal limitations.
- Link evidence and approve SPEC-03.

Exit criteria:

- Stakeholders accept split quality and limitations.
- Manifest and assignment artifacts reproduce from one documented command.

## 9. File-Level Work Breakdown

| File or area | Planned change |
| --- | --- |
| `specs/SPEC-03-data-splitting-and-leakage-control.md` | Replace placeholders with approved requirements, tests, policies, and evidence. |
| `configs/training.yaml` | Add split versions, row key, attempts, tolerances, weights, privacy, and artifact paths. |
| `utils/config.py` | Validate split configuration and cross-field constraints. |
| `data/splitting.py` | Implement deterministic scored assignment, feasibility checks, exact invariants, and manifest data. |
| `data/split_manifest.py` | Optionally isolate aliasing, fingerprinting, schema construction, and atomic writes. |
| `data/preparation.py` | Reassert row identity and feature/target separation per partition. |
| `features/selection.py` | Retain the centralized deny/allow contract and expose auditable exclusions. |
| `pipelines/training_pipeline.py` | Separate development selection from final test evaluation and bind metadata. |
| `scripts/preprocess_data.py` | Display split quality and verify train-only preprocessing. |
| `scripts/evaluate_model.py` | Enforce explicit final-test access and write evaluation audit metadata. |
| `artifacts/metadata/split_manifest.json` | Store versioned, source-bound, privacy-safe split summary. |
| `artifacts/metadata/split_assignments.csv` | Store privacy-safe row/group partition assignments. |
| `tests/unit/test_splitting.py` | Cover validation, feasibility, determinism, exact coverage, quality, and privacy. |
| `tests/integration/` | Cover split/preprocessing/training boundaries and artifact reproducibility. |
| `readme.md`, notebook | Document commands, interpretation, sealing, and temporal limitations. |

## 10. Test Matrix

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| SPLIT-T-001 | Valid multi-group, multi-class data | Splitting runs | Train, validation, and test are returned with every class. |
| SPLIT-T-002 | Same source/config/seed twice | Splitting runs | Row assignments, manifest, and fingerprints are identical. |
| SPLIT-T-003 | Same rows in different input order | Splitting runs | Assignment by stable row key is unchanged. |
| SPLIT-T-004 | Distinct groups | Invariants run | Partition group sets are pairwise disjoint and their union matches the source. |
| SPLIT-T-005 | Unique row keys | Invariants run | Row-key sets are pairwise disjoint and their union exactly matches the source. |
| SPLIT-T-006 | Invalid/zero/negative/oversummed ratios | Configuration validates | A clear error identifies the invalid ratios. |
| SPLIT-T-007 | Missing target, group, or row-key column | Splitting runs | A clear contract error identifies the missing column. |
| SPLIT-T-008 | Null target/group/row key | Splitting runs | The dataset is rejected without dropping rows. |
| SPLIT-T-009 | Duplicate row key | Splitting runs | The dataset is rejected with the duplicate-key count. |
| SPLIT-T-010 | Fewer than three groups | Feasibility validates | Splitting fails before candidate search. |
| SPLIT-T-011 | A class exists in fewer than three groups | Feasibility validates | The deficient class and group count are reported. |
| SPLIT-T-012 | Feasible but imbalanced groups | Candidate search runs | The selected split minimizes the documented objective deterministically. |
| SPLIT-T-013 | No candidate meets class/tolerance rules | Search exhausts | Failure includes attempts and best-observed deviations. |
| SPLIT-T-014 | Sensitive group and row identifiers | Manifest writes | Raw identifier values do not appear in committed artifacts. |
| SPLIT-T-015 | Valid split | Manifest writes | Source/config/algorithm versions, parameters, distributions, deviations, hashes, and invariants are present. |
| SPLIT-T-016 | Extreme validation/test numeric values | Preprocessing fits | Training imputation/scaling statistics are unchanged. |
| SPLIT-T-017 | Validation/test-only category | Preprocessing fits | It is absent from fitted vocabulary and transforms safely. |
| SPLIT-T-018 | Target, ID, or dropped field enters feature configuration | Preparation runs | It fails before preprocessing or training. |
| SPLIT-T-019 | Development training command | Candidate selection runs | Test features/labels are never loaded by the selection stage. |
| SPLIT-T-020 | Frozen model and matching manifest | Final evaluation runs | One auditable test evaluation is recorded. |
| SPLIT-T-021 | Mismatched source/model/manifest fingerprints | Evaluation runs | Evaluation fails before reading test outcomes. |
| SPLIT-T-022 | Repeated final-test request | Evaluation runs | Approved warning/failure policy is enforced. |
| SPLIT-T-023 | Valid source and configuration | CLI runs twice | Manifest and assignments are byte-for-byte reproducible. |
| SPLIT-T-024 | Complete workflow | Source is rechecked | Raw CSV bytes and fingerprint remain unchanged. |

## 11. Requirements Traceability

| Requirement | Primary implementation | Verification |
| --- | --- | --- |
| SPLIT-001/002 | Training config validation and split preconditions | SPLIT-T-006 through T-011 |
| SPLIT-003/004 | Group assignment and exact set invariants | SPLIT-T-004/005/009 |
| SPLIT-005 | Stable ordering, seeded search, assignment hashing | SPLIT-T-002/003/023 |
| SPLIT-006/007/008 | Feasibility filtering and scored candidate search | SPLIT-T-001/011/012/013 |
| SPLIT-009/010/011 | Manifest schema, aliasing, and fingerprints | SPLIT-T-014/015/023 |
| SPLIT-012 | Central feature preparation contract | SPLIT-T-018 |
| SPLIT-013/014 | Split-before-fit orchestration | SPLIT-T-016/017/019 |
| SPLIT-015/016 | Sealed-test command and evaluation audit | SPLIT-T-019 through T-022 |
| SPLIT-017 | Manifest, model card, SPEC, and README limitation | Documentation review |

## 12. Acceptance Criteria

- [ ] SPEC-03 contains approved requirements and no normative placeholders.
- [ ] `client_id` grouping and `content_id` row identity are approved.
- [ ] Every eligible row and group appears in exactly one partition.
- [ ] Train, validation, and test groups and row keys are pairwise disjoint.
- [ ] Every configured class appears in every partition or has an approved exception.
- [ ] Actual row ratios and class deviations satisfy approved tolerances.
- [ ] Repeated runs produce identical assignments and artifact hashes.
- [ ] The manifest is versioned and bound to source, configuration, feature contract, and algorithm.
- [ ] Published split artifacts contain no raw sensitive identifiers.
- [ ] Target, identifiers, and leakage fields cannot enter feature matrices.
- [ ] Learned preprocessing and selection state uses training data only.
- [ ] Candidate selection does not access test features or labels.
- [ ] Final test access is explicit and auditable.
- [ ] Snapshot and temporal-validation limitations are documented.
- [ ] Unit, integration, contract, and lint checks pass.

## 13. Recommended Implementation Order

1. Approve grouping, row-key, ratio, tolerance, privacy, and test-access policies.
2. Add and validate the versioned split configuration.
3. Add feasibility checks for groups, classes, keys, and ratios.
4. Refactor deterministic candidate scoring and tie-breaking.
5. Implement exact row/group invariants and assignment fingerprints.
6. Implement privacy-safe manifest and assignment artifacts.
7. Separate development selection from final test evaluation.
8. Bind preprocessing/model/evaluation metadata to the split manifest.
9. Add the complete unit, integration, and contract test matrix.
10. Regenerate artifacts, review split quality, and approve SPEC-03.

## 14. Definition of Done

SPEC-03 is complete only when:

- the split contract and exception policy are approved;
- a single command reproduces privacy-safe, source-bound split artifacts;
- exact row and group set invariants prove complete disjoint assignment;
- all partitions have approved class coverage and distribution quality;
- feature/preprocessing leakage assertions pass for every partition;
- development selection cannot access test data through its normal interface;
- final test evaluation is separately authorized and audited;
- the current split is explicitly described as client-grouped snapshot validation, not temporal validation;
- all SPEC-03 and regression tests pass.

## 15. Open Decisions

- Confirm `client_id` is the operational independence unit and `content_id` is stable across snapshots.
- Decide whether 60/20/20 targets apply primarily to rows, groups, or both.
- Approve maximum row-ratio and per-class proportion deviations.
- Approve objective weights for size balance, class balance, and group-count balance.
- Approve deterministic search-attempt budget and behavior when tolerances are infeasible.
- Choose the project-scoped group/row aliasing salt and secure storage mechanism; it must not be committed.
- Decide whether split assignments are committed, generated in CI, or stored in controlled artifact storage.
- Approve final-test repeated-access behavior: hard failure, explicit override, or contamination warning.
- Define who authorizes final test evaluation and how authorization is recorded.
- Confirm whether client-grouped validation alone matches deployment or whether future data must use group-aware temporal validation.
