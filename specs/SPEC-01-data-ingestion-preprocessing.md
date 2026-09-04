# SPEC-01 — CSV Ingestion, Review, and Preprocessing

**Status:** Implemented — pending stakeholder approval of open policy decisions
**Owner:** Babar Ali Khan

**Implementation plan:** `plan/SPEC-01-data-ingestion-preprocessing/implementation-plan.md`

## Purpose

Load the approved CSV, validate its schema, report data-quality risks,
assign every column a role, and construct an unfitted preprocessing pipeline.

## Requirements

| ID | Requirement |
| --- | --- |
| ING-001 | Load the CSV using configured path, encoding, delimiter, and missing tokens. |
| ING-002 | Fail clearly when the source is missing, empty, unreadable, or malformed. |
| ING-003 | Calculate a SHA-256 source fingerprint. |
| REV-001 | Report shape, types, missingness, duplicates, constants, and cardinality. |
| REV-002 | Validate required, target, ID, dropped, and sensitive columns. |
| PRE-001 | Separate the target before feature preprocessing. |
| PRE-002 | Exclude IDs, target, dropped fields, and confirmed leakage columns. |
| PRE-003 | Handle numeric and categorical missing values through configured strategies. |
| PRE-004 | Allow unseen inference categories without crashing. |
| PRE-005 | Fit every learned transformation on training data only. |

## Tests

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| T-ING-001 | A valid CSV | Loading runs | Shape and fingerprint are returned. |
| T-ING-002 | A missing or empty file | Loading runs | A useful validation error is raised. |
| T-REV-001 | Invalid schema | Validation runs | Exact issues are reported. |
| T-PRE-001 | Numeric and categorical features | Training transformation runs | No unexpected null values remain. |
| T-PRE-002 | An unseen category | Inference transformation runs | Transformation succeeds. |

## Acceptance Criteria

- [x] The original raw CSV remains unchanged.
- [x] Dataset fingerprint and profile are reproducible.
- [x] Every configured column has an explicit role.
- [x] No row or column is silently removed.
- [x] Target leakage and preprocessing leakage are prevented.
- [x] All mandatory automated tests pass.

## Implementation Notes

- Profiles include schema/feature-contract versions, column roles, model eligibility,
  warnings, and a source SHA-256 that is rechecked after review.
- Unknown source columns fail role validation by default.
- Target, identifiers, and dropped fields are separated from features through one central
  preparation contract.
- Numeric and categorical transformations are configuration-driven. Learned preprocessing
  state is fitted on the training partition only; validation, test, and inference use
  `transform`.
- Ingestion and preparation never rewrite the raw CSV or silently remove rows.

## Open Policy Decisions

- Confirm missing targets remain a hard failure rather than entering a quarantine flow.
- Confirm duplicate full rows remain report-only while duplicate `content_id` is fatal.
- Confirm expected row and column counts remain hard constraints for this dataset version.
- Confirm `provider_used` remains excluded because its current missingness exceeds 60%.
