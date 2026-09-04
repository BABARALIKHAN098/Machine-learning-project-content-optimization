# SPEC-02 — Exploratory Data Analysis

**Status:** Implemented — domain thresholds pending stakeholder approval
**Owner:** Babar Ali Khan  
**Outcome:** Document distributions, relationships, anomalies, and data risks.
**Implementation plan:** `plan/SPEC-02-exploratory-data-analysis/implementation-plan.md`

## Purpose

Produce a deterministic, leakage-aware analysis of the validated dataset before model
development. The analysis is descriptive: it does not clean data, select features, fit
models, or establish causation.

## In Scope

- Target, numeric, categorical, missingness, and client-cohort distributions.
- Numeric correlations and model-free feature-to-target effect sizes.
- Configurable domain and cross-field anomaly rules.
- Column-level cutoff availability and leakage classification.
- Versioned JSON/CSV summaries, aggregate PNG figures, and a Markdown report.
- Privacy masking, support flags, source immutability, and reproducibility checks.

## Out of Scope

- Automatic source correction, imputation, encoding, scaling, or row removal.
- Train/validation/test creation or final holdout inspection.
- Feature selection, model fitting, tuning, causal inference, or monitoring.
- Publication of raw sensitive identifiers or row-level extracts.

## Requirements

| ID | Requirement |
| --- | --- |
| EDA-001 | Run only after SPEC-01 ingestion and schema validation succeed. |
| EDA-002 | Record source fingerprint and artifact/data schema versions. |
| EDA-003 | Preserve the source dataframe, row count, and raw CSV bytes. |
| EDA-004 | Report target counts, proportions, imbalance ratio, entropy, and supported cohort distributions. |
| EDA-005 | Report numeric distributions, quantiles, skewness, zeros, non-finite values, and robust outliers. |
| EDA-006 | Report categorical frequency, missingness, coverage, and rare-level flags without exposing identifiers. |
| EDA-007 | Report per-column and common-pattern missingness plus missingness-to-target association. |
| EDA-008 | Report Pearson/Spearman numeric associations and numeric/categorical target effect sizes. |
| EDA-009 | Run stable, configurable anomaly rules without modifying records. |
| EDA-010 | Account for every source column in a leakage and model-eligibility register. |
| EDA-011 | Mask client groups and identify insufficient cohort support. |
| EDA-012 | Generate deterministic machine-readable artifacts, narrative report, and aggregate figures. |
| EDA-013 | Handle constant, all-null, non-finite, and unsupported fields without invalid calculations. |
| EDA-014 | Clearly document descriptive, temporal, leakage, privacy, and snapshot-data limitations. |

## Tests

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| EDA-T-001 | Known numeric and categorical values | Distribution analysis runs | Counts, quantiles, outliers, missing levels, and rare flags are exact. |
| EDA-T-002 | Imbalanced labeled rows across clients | Target analysis runs | Counts and proportions are exact and client values are masked. |
| EDA-T-003 | Known missingness and correlated columns | Relationship analysis runs | Missingness and association measures match expected values. |
| EDA-T-004 | Rows violating enabled rules | Anomaly analysis runs | Stable rule IDs and exact aggregate violation counts are returned. |
| EDA-T-005 | Target-derived and outcome-window fields | Leakage analysis runs | Fields are marked ineligible with explicit rationale. |
| EDA-T-006 | Invalid thresholds or quantiles | Configuration validation runs | Actionable validation errors are raised. |
| EDA-T-007 | Valid temporary CSV and configurations | Full pipeline runs | All artifacts are created and the source remains unchanged. |
| EDA-T-008 | Sensitive client values | Reports render | Raw client identifiers do not appear in artifacts. |

## Acceptance Criteria

- [x] Requirements are implemented and traceable to automated tests.
- [x] The raw CSV and analysis dataframe remain unchanged.
- [x] All required versioned EDA artifacts are reproducible from one command.
- [x] Every source column appears in the leakage register.
- [x] Target, numeric, categorical, missingness, relationship, and anomaly analyses are present.
- [x] Sensitive identifiers are excluded or masked in published summaries.
- [x] EDA performs no implicit cleaning, row removal, preprocessing, or model fitting.
- [x] Automated EDA tests and project lint checks pass.
- [ ] Domain ranges, tier formulas, and target construction thresholds are approved.

## Observed Findings

- The target is imbalanced: `down` is 54.21% and `flat` is 3.84%, an imbalance ratio of 14.12.
- `provider_used` has 71.46% missingness; word/character length fields have 25.66% missingness.
- Nineteen numeric pairs exceed the configured Pearson or Spearman threshold of 0.90.
- Twenty-five columns are prohibited or unavailable as model inputs under the current contract.
- Provisional rate-domain checks generate substantial warnings and must not be treated as confirmed errors until their units are approved.

## Open Decisions

- Approve the formula and thresholds used to create `trend_direction` and tiers.
- Confirm the prediction cutoff and exact periods represented by all 30-day and 90-day fields.
- Confirm whether rate fields use fractions, percentages, or another scale.
- Approve parent-child consistency rules for analytics metrics.
- Approve client/cohort support and suppression policy.
- Confirm `provider_used` remains excluded because of high missingness.

## Definition of Done

This phase is complete when every automated acceptance criterion passes and the remaining
domain definitions are approved or explicitly accepted as documented limitations.
