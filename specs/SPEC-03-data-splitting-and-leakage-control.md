# SPEC-03 — Data Splitting and Leakage Control

**Status:** In progress — split contract implemented; sealed-test workflow pending
**Owner:** Babar Ali Khan  
**Outcome:** Create reproducible train, validation, and test sets.
**Implementation plan:** `plan/SPEC-03-data-splitting-and-leakage-control/implementation-plan.md`

## Purpose

Create deterministic, client-grouped partitions with exact row/group coverage, usable class
coverage, measurable split quality, privacy-safe metadata, and leakage-safe preprocessing
boundaries. The current snapshot split measures generalization to unseen clients; it does
not establish future-period performance.

## In Scope

- Validated split ratios, identity columns, labels, search budget, and tolerances.
- Whole-`client_id` assignment to train, validation, or test.
- Exact `content_id` and client coverage/disjointness assertions.
- Deterministic candidate scoring and stable assignment fingerprints.
- Source-bound, versioned, privacy-safe manifests and assignments.
- Split-before-fit preprocessing and central feature exclusions.
- A separately auditable sealed-test workflow.

## Out of Scope

- Data cleaning and exploratory analysis.
- Feature engineering, model tuning, or metric interpretation.
- Oversampling before splitting.
- Temporal validation until timestamped snapshots exist.

## Requirements

| ID | Requirement | Status |
| --- | --- | --- |
| SPLIT-001 | Validate split configuration and dataset feasibility before search. | Implemented |
| SPLIT-002 | Assign every client wholly to one partition. | Implemented |
| SPLIT-003 | Assign every unique row exactly once without mutation. | Implemented |
| SPLIT-004 | Preserve every configured class in every partition. | Implemented |
| SPLIT-005 | Produce order-independent deterministic assignments for the same source/config/seed. | Implemented |
| SPLIT-006 | Enforce configured row-ratio and class-distribution tolerances. | Implemented |
| SPLIT-007 | Bind metadata to source, schema, contract, algorithm, and parameters. | Implemented |
| SPLIT-008 | Publish only hashed row keys and masked client aliases. | Implemented |
| SPLIT-009 | Fit learned preprocessing on train only and transform validation/test. | Implemented |
| SPLIT-010 | Exclude target, IDs, dropped fields, and leakage fields from features. | Implemented |
| SPLIT-011 | Keep candidate selection independent of final test outcomes. | Pending |
| SPLIT-012 | Record and control first/repeated final-test access. | Pending |
| SPLIT-013 | Document snapshot and temporal-validation limitations. | Implemented |

## Tests

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| SPLIT-T-001 | Valid grouped multi-class data | Splitting runs twice | Assignments and manifests match. |
| SPLIT-T-002 | Same rows in a different order | Splitting runs | Assignment by row key remains unchanged. |
| SPLIT-T-003 | Missing/null/duplicate identity data | Validation runs | Actionable errors are raised without dropping rows. |
| SPLIT-T-004 | A class in fewer than three groups | Feasibility runs | The class and group count are reported. |
| SPLIT-T-005 | Valid partitions | Invariants run | Row/group unions and pairwise disjointness are exact. |
| SPLIT-T-006 | Sensitive identifiers | Artifacts write | Raw row/client identifiers are absent. |
| SPLIT-T-007 | Invalid split ratios/settings | Configuration validates | An actionable contract error is raised. |
| SPLIT-T-008 | Validation/test-only statistics/categories | Preprocessing runs | Learned state remains train-only. |
| SPLIT-T-009 | Development selection | Training runs | Test data is not accessed. |
| SPLIT-T-010 | Frozen artifact and manifest | Final evaluation runs | Test access is recorded and repeat policy enforced. |

## Acceptance Criteria

- [x] Every input row and client appears in exactly one partition.
- [x] Partition row and client sets are pairwise disjoint.
- [x] Every configured class occurs in every partition.
- [x] Actual split and class deviations meet configured tolerances.
- [x] Repeated and input-reordered runs produce identical assignments.
- [x] Manifest and assignment artifacts are versioned and source-bound.
- [x] Published split metadata contains no raw client or content identifiers.
- [x] Preprocessing is fitted on train only.
- [x] Target, identifiers, and leakage fields are excluded from features.
- [ ] Development candidate selection cannot access final test data.
- [ ] Final test access is explicit, separately invoked, and auditable.
- [x] Snapshot/temporal limitations are documented.
- [ ] All planned sealed-test contract tests pass.

## Current Partition

| Partition | Rows | Actual ratio | Clients |
| --- | ---: | ---: | ---: |
| Train | 17,670 | 58.90% | 18 |
| Validation | 5,857 | 19.52% | 7 |
| Test | 6,473 | 21.58% | 7 |

All five target classes occur in each partition. Exact assignments are bound to source
SHA-256 `c43bdac4eccfa17fcd8a33974fe36f2c998c03a3ae3af8d80cf712abab5d6396`.

## Open Decisions

- Approve whether split targets primarily optimize rows, groups, or a weighted compromise.
- Approve the current 5% row-ratio and per-class deviation tolerances.
- Select a secure project-scoped alias salt if reversible linkage risk requires one.
- Approve final-test authorization, repeat-access, and contamination policies.
- Confirm whether deployment requires group-aware temporal validation when snapshots exist.

## Definition of Done

This phase is complete when the implemented partition invariants and privacy controls pass,
candidate selection is separated from test evaluation, test access is auditable, all tests
pass, and the remaining policy choices are approved or accepted as limitations.
