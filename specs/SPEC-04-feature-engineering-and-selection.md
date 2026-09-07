# SPEC-04 — Feature Engineering and Selection

**Status:** Implemented for development; historical cutoff evidence and final-test policy remain limitations
**Owner:** Babar Ali Khan
**Outcome:** Produce justified, reproducible model inputs.

**Implementation plan:** `plan/SPEC-04-feature-engineering-and-selection/implementation-plan.md`
**Feature contract:** `configs/features.yaml`, version 2.0
**Evidence:** `reports/features/feature_manifest.json`, `reports/features/selection_report.md`

## Purpose

Create auditable inputs for five-class `trend_direction` prediction using the existing
client-grouped development partitions. Prediction timing is the end of the previous
30-day observation period. Feature studies cannot use test rows for fitting, diagnostics,
candidate prediction, or selection. Full-source integrity validation remains upstream.

## In Scope

- Ordered raw/derived registries, dependency validation, and exclusions.
- Previous-period ratios, unavailable indicators, and optional log transforms.
- Stable train-fitted preprocessing and identical saved transformations at inference.
- Fixed grouped-fold feature-family ablations and one validation confirmation.
- Source/split/config/code provenance, aggregate diagnostics, and frozen input handoff.
- Training workflow separation from final test evaluation.

## Out of Scope

- Label changes, split searches during feature experiments, and raw-data mutation.
- Target encoding, arbitrary formulas, learned tier thresholds, and broad tuning.
- Final-test evaluation/auditing, deployment, and prospective temporal validation.

## Requirements

| ID | Requirement |
| --- | --- |
| SPEC-04-REQ-001 | Resolve versioned ordered raw/derived roles and formula dependencies. |
| SPEC-04-REQ-002 | Exclude target, IDs, sensitive fields, dropped fields and forbidden dependencies. |
| SPEC-04-REQ-003 | Preserve source values, rows, index, and order; reject missing dependencies. |
| SPEC-04-REQ-004 | Define and test zero, missing, negative, non-finite and overflow behavior. |
| SPEC-04-REQ-005 | Fit transformations on training rows or the current training fold only. |
| SPEC-04-REQ-006 | Preserve fitted output names and dimensions across missing/unseen inputs. |
| SPEC-04-REQ-007 | Compare the fixed five families across grouped folds using the frozen rule. |
| SPEC-04-REQ-008 | Expose only train/validation rows to development model code. |
| SPEC-04-REQ-009 | Publish ordered lineage, exclusions, selected configs and provenance. |
| SPEC-04-REQ-010 | Compute derived inputs inside saved pipelines; retain legacy artifact behavior. |
| SPEC-04-REQ-011 | Reproduce semantic decisions/configurations for fixed inputs and environment. |
| SPEC-04-REQ-012 | Document results, rejected variants, timing uncertainty and previous test exposure. |

## Implemented Contract

- Keep the 19 currently approved raw inputs as reference A. Never add generated names to
  the raw CSV schema or require callers to send them.
- B adds `previous_ctr`, `previous_sessions_per_click`, and one unavailable flag per ratio.
- C adds B plus `log1p` of search volume, CPC, word/character counts and previous-period counts.
- D uses B without age/freshness/length tiers and `age_tier_order`.
- E uses B without `model_used`.
- Undefined ratios remain missing until imputation; a genuine zero numerator with a positive
  denominator stays zero. Negative/non-finite numeric observations and ratio overflow fail.
- Entirely missing training numeric columns retain their dimension with zero imputation.
- The registry supports a fixed operation language, not arbitrary dependency graphs;
  unsupported generated dependencies and cycles are rejected as unavailable raw inputs.
- Each estimator uses three seeded client-grouped folds within train. Rank by mean macro F1
  subject to mean `down` recall >= 0.50. Within 0.005, prefer fewer columns, then A, then ID.
- Confirm the selected variant and A once on validation. Require recall >= 0.50 and macro F1
  no more than 0.005 below A; additional columns require at least 0.005 improvement.
- Feature config owns v2 input behavior. The preprocessing version must agree when engineering
  is enabled. Legacy artifacts keep their original saved pipeline and schema.
- `train_model.py` performs development-only fitting and validation; it no longer refits on
  train plus validation or evaluates/benchmarks the final test partition.

## Tests

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| FE-T-001/002/003 | Invalid roles, versions, names or dependencies | Registry resolves | Invalid contracts fail before fitting. |
| FE-T-004/005/006/007 | Reordered rows, missing dependencies and ratio/log edge cases | Engineering transforms | Values and indicators are correct; source unchanged; invalid input fails. |
| FE-T-008/009/010 | Missing/unseen values and grouped training folds | Pipelines fit/transform | Fitted state uses training rows; names/width remain stable. |
| FE-T-011/016 | Serialized, legacy, incomplete or tampered bundle | Predictor loads/predicts | Raw-input parity and compatibility checks hold. |
| FE-T-012/013/015 | Fixed study repeated, frozen configs or changed provenance | Study/handoff runs | Decisions reproduce; incompatible evidence is rejected. |
| FE-T-014 | Persisted train/validation/test partitions | Development training runs | No test rows reach supervised preparation or fitting. |
| FE-T-017 | Study evidence | Documentation is reviewed | Exposure, timing assumptions and limits are explicit. |

Executable coverage is in `tests/unit/test_feature_registry.py`,
`tests/unit/test_feature_engineering.py`, `tests/unit/test_preprocessing.py`,
`tests/integration/test_feature_pipeline.py`, and `tests/contract/test_feature_contract.py`.

## Acceptance Criteria

- [x] Development requirements/defaults are adopted through the user's instruction to implement the plan.
- [x] Registry, engineering, train-only fitting, development isolation and inference parity are implemented.
- [x] A bounded feature study and frozen handoff are available from documented commands.
- [x] Decisions and limitations are documented.
- [x] Full regression and real-data reproducibility evidence is recorded in `reports/features/verification.json`: 62 tests pass, Ruff passes, all 30 repeated fold results match, and real-data frozen handoff passes.
- [ ] Historical metadata availability is confirmed for production cutoff claims.

## Open Decisions

- Historical availability of search metadata, tiers, age/freshness and `model_used` remains
  unverified. Development evidence is provisional rather than proof of production cutoff safety.
- The existing holdout has already been evaluated. This phase does not restore its independence.
  First/repeat final-test authorization and audit remain the pending SPEC-03 work.
- A future untouched timestamped holdout is needed for an independent future-period estimate.
- No optional learned statistical selector was added: selection is the fixed family study.

## Definition of Done

Development implementation is complete when the automated suite, reproducibility checks,
and frozen handoff pass. Production readiness additionally requires historical availability
evidence and the final-test policy owned by SPEC-03. A model-performance gain is not required.
