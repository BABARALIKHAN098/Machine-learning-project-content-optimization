# Implementation Plan — Exploratory Data Analysis

**Source specification:** `specs/SPEC-02-exploratory-data-analysis.md`  
**Plan status:** Implemented — automated gates pass; domain approvals remain  
**Owner:** Babar Ali Khan  
**Dataset:** `data/raw/dataset.csv`  
**Target:** `trend_direction`  
**Task:** Leakage-aware exploratory analysis for five-class classification

## 1. Objective

Create a deterministic exploratory-data-analysis workflow that documents the dataset's distributions, missingness, feature relationships, target balance, client-level variation, anomalies, redundancy, and leakage risks before further modeling decisions are made.

The workflow must produce machine-readable results and a concise human-readable report. It must not modify the raw data, fit predictive models, select features against holdout performance, or present association as causation.

## 2. Specification Gap and Assumptions

SPEC-02 currently contains placeholder scope, requirements, tests, and open decisions. This plan is therefore a proposed implementation contract based on:

- the stated SPEC-02 outcome;
- SPEC-00's content-trend classification objective;
- SPEC-01's strict ingestion and column-role contract;
- SPEC-03's planned grouped splitting and leakage controls;
- the current 30,000-row, 44-column source dataset.

Assumptions requiring approval:

- EDA may use the validated full raw dataset for data-quality discovery.
- Target-aware analysis is descriptive only and cannot be used to tune repeatedly against a sealed test set.
- `content_id` is limited to uniqueness and internal anomaly traceability.
- `client_id` may support grouped summaries, but published reports use stable aliases rather than raw identifiers.
- Outcome-window and target-derived fields may be analyzed to verify target construction, but remain prohibited model inputs.
- All plots contain aggregated data and deterministic ordering.

## 3. Scope

### In scope

- Dataset shape, types, uniqueness, constants, cardinality, and memory usage.
- Global target distribution and target distribution across supported cohorts.
- Numeric distributions, quantiles, skewness, zero inflation, and robust outliers.
- Categorical frequencies, rare levels, and missing categories.
- Column missingness, co-missingness, and missingness-to-target relationships.
- Numeric correlations, categorical associations, feature-to-target effect sizes, and redundancy.
- Domain anomalies and cross-field consistency checks.
- Client-level variation relevant to grouped splitting and generalization.
- A column-level prediction-cutoff and leakage register.
- Reproducible JSON/CSV summaries, PNG figures, and a Markdown report.

### Out of scope

- Imputation, encoding, scaling, feature selection, or automatic row removal.
- Train/validation/test creation and final holdout inspection.
- Model fitting, tuning, causal inference, or production monitoring.
- Automatic correction of source records.
- Publication of raw sensitive identifiers or unrestricted row-level extracts.

## 4. Proposed Requirements

| ID | Requirement |
| --- | --- |
| EDA-001 | Load data through the validated SPEC-01 ingestion and schema contract. |
| EDA-002 | Record source SHA-256, schema/config version, package versions, row count, and column count in the run manifest. |
| EDA-003 | Verify that EDA does not mutate the dataframe or raw CSV. |
| EDA-004 | Assign every column an EDA role and model-eligibility status. |
| EDA-005 | Report target counts, proportions, imbalance, entropy, and supported cohort distributions. |
| EDA-006 | Report numeric missingness, zeros, non-finite values, descriptive statistics, quantiles, skewness, and robust outliers. |
| EDA-007 | Report categorical missingness, cardinality, frequency, cumulative coverage, and rare levels. |
| EDA-008 | Analyze common missingness patterns, co-missingness, and target/client differences in missingness. |
| EDA-009 | Calculate Pearson and Spearman numeric associations and flag configurable high-correlation pairs. |
| EDA-010 | Calculate supported, model-free feature-to-target association measures appropriate to each feature type. |
| EDA-011 | Detect configurable domain and cross-field anomalies without changing or deleting data. |
| EDA-012 | Produce a leakage register covering identifiers, target-derived fields, post-cutoff fields, and overlapping windows. |
| EDA-013 | Generate deterministic aggregate figures with stable names and readable labels. |
| EDA-014 | Generate versioned machine-readable artifacts and a curated human-readable EDA report. |
| EDA-015 | Mask sensitive group values and suppress low-support report cells. |
| EDA-016 | Handle constant, all-null, high-cardinality, non-finite, and unsupported fields gracefully. |

## 5. Analysis Design

### 5.1 Column inventory

Extend the SPEC-01 structural profile so each column records:

- primary role: target, identifier, group, candidate feature, or dropped field;
- dtype, missing count, unique count, and cardinality ratio;
- constant/all-null/high-cardinality flags;
- sensitivity and model eligibility;
- prediction-cutoff availability;
- relationship to a source or derived field.

Unknown or conflicting roles must fail before EDA. No new column may be inferred as model-safe automatically.

### 5.2 Target analysis

For `trend_direction`:

- verify the configured classes `down`, `stable`, `up`, `new`, and `flat`;
- report counts, proportions, imbalance ratio, and entropy;
- generate a labeled target-distribution chart;
- cross-tabulate by masked client, `content_type`, `main_intent`, `age_tier`, and `freshness_tier`;
- flag cohorts with absent classes or support below a configurable minimum;
- examine whether `new`, `flat`, and `stable` have distinct documented definitions.

Strong target separation in `trend_pct` or outcome-window fields must be described as target construction/leakage, not predictive evidence.

### 5.3 Numeric distributions

For each numeric candidate and excluded numeric field, calculate:

- non-null count and missing percentage;
- zero, negative, and non-finite counts;
- mean, standard deviation, minimum, maximum, and skewness;
- 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles;
- IQR and outlier count using a configurable default of `1.5 × IQR`.

Produce selected histogram/box plots. Use log-scaled views for highly skewed non-negative count variables when they materially improve interpretation. Outliers are flagged, never removed.

### 5.4 Categorical distributions

For each categorical field:

- report missing count, cardinality, level count, percentage, and cumulative coverage;
- represent nulls as an explicit analysis category without altering source data;
- flag levels below the configured rare-category threshold;
- cross-tabulate supported levels against the target;
- group or suppress report cells below the approved privacy/support threshold.

### 5.5 Missingness

- Rank columns by missing percentage and compare with configured limits.
- Report common row-level missingness patterns and pairwise co-missingness.
- Calculate bias-corrected Cramér's V between missingness indicators and the target.
- Compare missingness by masked client to expose source-specific collection gaps.
- Prioritize known risks: `provider_used` (~71%), `word_count`/`char_count` (~26%), and `model_used` (~19%).
- Describe missingness as structural, collection-related, or unexplained only when evidence supports the label; do not assert MCAR/MAR/MNAR without evidence.

### 5.6 Relationships and redundancy

- Calculate Pearson correlations for linear numeric relationships.
- Calculate Spearman correlations for monotonic numeric relationships.
- Flag absolute correlation at or above a configurable threshold, proposed default `0.90`.
- Use per-class summaries and eta-squared for numeric-to-target relationships.
- Use contingency tables and bias-corrected Cramér's V for categorical-to-target relationships.
- Detect exact or near-deterministic mappings between values and tiers.
- Highlight redundancy among age, freshness, length, impression, and position representations.
- Never use full-dataset associations alone to approve a feature; selection belongs in leakage-safe development folds.

### 5.7 Anomaly rules

Implement configurable rules with stable IDs and severities for:

- negative counts, cost, age, or elapsed-day values;
- non-finite numeric values;
- rates or percentages outside approved ranges;
- clicks exceeding impressions within a matching time window;
- child activity totals exceeding an approved parent total;
- `days_with_*` exceeding its aggregation window;
- inconsistent exact values and tier labels;
- duplicate IDs, duplicate rows, constants, all-null columns, and unknown categories;
- target labels inconsistent with approved `trend_pct` thresholds.

Every rule result contains rule ID, severity, checked rows, violation count, violation percentage, and remediation guidance. Rules without approved business definitions remain disabled and listed as open decisions.

### 5.8 Leakage register

Create one record per column containing:

- column name and primary role;
- availability at prediction cutoff: `yes`, `no`, or `unknown`;
- aggregation window when known;
- relationship to target construction;
- leakage category;
- model eligibility;
- rationale and approval status.

At minimum, review `trend_direction`, `trend_pct`, all `*_last_30d` fields, potentially overlapping `*_90d` fields, their derived rates/tiers, identifiers, and other potentially post-cutoff fields.

## 6. Configuration Contract

Add `configs/eda.yaml` with validated settings for:

- artifact schema version and random seed;
- quantiles and IQR multiplier;
- rare-category and high-correlation thresholds;
- minimum cohort/cell support and suppression behavior;
- maximum displayed categories and plot DPI/style;
- enabled anomaly rules, severities, domains, and thresholds;
- selected target cohorts and report-priority columns;
- output directories and formats.

Invalid types, unsupported values, duplicate rules, or contradictory thresholds must fail before analysis begins.

## 7. Artifact Contract

| Artifact | Purpose |
| --- | --- |
| `reports/eda/eda_report.md` | Curated findings, risks, decisions, and limitations. |
| `reports/eda/eda_summary.json` | Run manifest and versioned machine-readable overview. |
| `reports/eda/numeric_summary.csv` | Numeric distributions and outlier counts. |
| `reports/eda/categorical_summary.csv` | Frequencies, missingness, and rare-level flags. |
| `reports/eda/target_crosstabs.csv` | Long-form supported target/cohort counts and percentages. |
| `reports/eda/missingness_summary.csv` | Column and pattern-level missingness analysis. |
| `reports/eda/associations.csv` | Correlations and feature-to-target effect sizes. |
| `reports/eda/anomalies.csv` | Aggregate anomaly-rule results. |
| `reports/eda/leakage_register.csv` | Prediction-time availability and eligibility decisions. |
| `reports/figures/eda/*.png` | Deterministic aggregate plots. |

Artifacts must have stable column order, deterministic row ordering, and an explicit schema version. Volatile timestamps must be isolated from reproducibility comparisons. Output writes should be atomic so a failed run is not presented as complete.

## 8. Delivery Phases

### Phase 0 — Approve the EDA contract

**Files:** `specs/SPEC-02-exploratory-data-analysis.md`, this plan.

Tasks:

- Replace all SPEC-02 placeholders with approved scope, requirements, tests, and acceptance criteria.
- Confirm prediction cutoff, target formula, and time-window definitions.
- Confirm valid rate domains and tier boundaries.
- Approve association, rarity, support, suppression, and anomaly thresholds.
- Decide whether generated artifacts are committed or built in CI.

Exit criteria:

- No normative `TBD` remains in SPEC-02.
- Every enabled anomaly rule has an approved definition.

### Phase 1 — Configuration and result schemas

**Files:** `configs/eda.yaml`, `configs/data.yaml`, `src/machine_learning_project/data/eda.py`.

Tasks:

- Add and validate EDA configuration.
- Add approved cutoff availability, time-window, domain, and tier metadata.
- Define typed internal results for analysis sections, findings, and warnings.
- Version all artifact schemas.

Exit criteria:

- Every analysis threshold and behavior is configured and validated.

### Phase 2 — Pure analysis functions

**Files:** `src/machine_learning_project/data/eda.py`, `src/machine_learning_project/data/profiling.py`.

Tasks:

- Implement separate functions for inventory, target, numeric, categorical, missingness, association, redundancy, and anomaly analysis.
- Keep computation independent of file output and plotting.
- Normalize pandas/numpy scalar values for JSON serialization.
- Return structured warnings for unsupported or degenerate columns.
- Work on defensive views/copies and assert input immutability.

Exit criteria:

- Every component is independently testable on a small dataframe.

### Phase 3 — Anomaly and leakage controls

**Files:** `src/machine_learning_project/data/eda.py`, `src/machine_learning_project/data/validation.py`, `configs/*.yaml`.

Tasks:

- Generate the leakage register from the approved column and availability metadata.
- Verify every source column occurs exactly once in the register.
- Implement stable, configuration-driven anomaly rule IDs.
- Separate schema-blocking errors from descriptive warnings.
- Aggregate violations without publishing raw sensitive records.

Exit criteria:

- Every column has a traceable eligibility decision.
- Every anomaly is tied to a documented rule and threshold.

### Phase 4 — Visualization and reporting

**Files:** `src/machine_learning_project/data/visualization.py`, `src/machine_learning_project/data/reporting.py`, `pyproject.toml`.

Tasks:

- Add approved plotting dependencies, proposed `matplotlib` and `seaborn`.
- Render target, missingness, distribution, correlation, and group-comparison figures.
- Use fixed styles, explicit denominators, stable order, and accessible labels/colors.
- Render a concise Markdown report with an executive summary, evidence, risks, decisions, and artifact links.
- Include leakage, privacy, descriptive-not-causal, and snapshot-data limitations.

Exit criteria:

- Figures are readable, deterministic, and contain no raw sensitive identifiers.

### Phase 5 — Pipeline and CLI

**Files:** `pipelines/data_pipeline.py`, `scripts/run_eda.py`, `Makefile`, `readme.md`.

Tasks:

- Add a pipeline that runs ingestion, validation, analysis, rendering, and final fingerprint verification.
- Add CLI options for data config, EDA config, and output directory.
- Write artifacts through a staged/atomic output process.
- Return non-zero status for validation failures and enabled error-severity gates.
- Document the command and interpretation guidance.

Proposed command:

```powershell
.\.venv\Scripts\python.exe scripts\run_eda.py `
  --data-config configs/data.yaml `
  --eda-config configs/eda.yaml `
  --output-dir reports/eda
```

Exit criteria:

- One documented command reproduces the complete artifact set.

### Phase 6 — Automated tests

**Files:** `tests/unit/test_eda.py`, `tests/unit/test_eda_reporting.py`, `tests/integration/test_eda_pipeline.py`.

Tasks:

- Cover balanced/imbalanced targets, missingness, skew, outliers, rare levels, known associations, constants, all-null fields, and non-finite values.
- Verify domain and cross-field anomaly rules.
- Verify masking and low-support suppression.
- Verify deterministic artifacts and input/source immutability.
- Verify incomplete output is not published after failure.
- Run the existing full suite and Ruff.

Exit criteria:

- All EDA tests and existing regression checks pass.

### Phase 7 — Findings review and sign-off

**Files:** `reports/eda/`, `specs/SPEC-02-exploratory-data-analysis.md`, downstream configs/specs.

Tasks:

- Review findings with the data owner and modeling stakeholder.
- Convert confirmed risks into validation rules, exclusions, preprocessing decisions, or accepted exceptions.
- Confirm leakage exclusions before modeling continues.
- Assign an owner, due date, and disposition to each material unresolved issue.
- Link acceptance evidence and approve SPEC-02.

Exit criteria:

- Each material issue is fixed, excluded, accepted, or assigned for investigation.
- Column eligibility and limitations are approved.

## 9. File-Level Work Breakdown

| File or area | Planned change |
| --- | --- |
| `specs/SPEC-02-exploratory-data-analysis.md` | Replace placeholders and record approval/evidence. |
| `configs/eda.yaml` | Define EDA thresholds, rules, privacy, plots, and outputs. |
| `configs/data.yaml` | Add approved domains, tiers, windows, and cutoff availability. |
| `data/profiling.py` | Reuse and extend the structural inventory. |
| `data/eda.py` | Implement all statistical, anomaly, and leakage analysis. |
| `data/visualization.py` | Generate deterministic aggregate figures. |
| `data/reporting.py` | Render versioned JSON/CSV and Markdown artifacts atomically. |
| `pipelines/data_pipeline.py` | Orchestrate the end-to-end EDA workflow. |
| `scripts/run_eda.py` | Expose a reproducible CLI. |
| `tests/unit/` | Verify calculations, edge cases, privacy, and rendering. |
| `tests/integration/` | Verify complete artifacts, immutability, and reproducibility. |
| `reports/eda/` | Store narrative and machine-readable analysis outputs. |
| `reports/figures/eda/` | Store stable aggregate figures. |
| `readme.md`, `Makefile` | Document and expose the workflow. |

## 10. Test Matrix

| Test ID | Given | When | Then |
| --- | --- | --- | --- |
| EDA-T-001 | Valid configured dataset | EDA runs | Required artifacts exist and carry the source fingerprint. |
| EDA-T-002 | Same source and config | EDA runs twice | Non-volatile content and artifact ordering match. |
| EDA-T-003 | Known numeric values | Numeric analysis runs | Statistics, quantiles, zero rate, skew, and outliers match expected values. |
| EDA-T-004 | Missing and rare categorical levels | Categorical analysis runs | Counts, percentages, missing category, and rare flags are correct. |
| EDA-T-005 | Imbalanced five-class target | Target analysis runs | Counts, proportions, entropy, and imbalance warning are correct. |
| EDA-T-006 | Linear and monotonic numeric pairs | Associations run | Pearson/Spearman results and threshold flags are correct within tolerance. |
| EDA-T-007 | Known target contingency table | Associations run | Cramér's V and support counts match expected values. |
| EDA-T-008 | Coupled missing columns | Missingness analysis runs | Column, co-missingness, and common-pattern counts are correct. |
| EDA-T-009 | Domain and cross-field violations | Anomaly analysis runs | Stable rule IDs and exact aggregate counts are returned. |
| EDA-T-010 | Constant, all-null, and non-finite fields | EDA runs | Warnings are recorded without invalid calculations or crashes. |
| EDA-T-011 | Sensitive client identifiers | Reports render | Raw client IDs and unsafe row contents are absent. |
| EDA-T-012 | New unconfigured column | Inventory runs | It fails rather than marking the column model-safe. |
| EDA-T-013 | Raw CSV and dataframe | Complete pipeline runs | Source fingerprint and dataframe values remain unchanged. |
| EDA-T-014 | Target-derived/post-cutoff fields | Leakage register builds | Fields are ineligible with explicit rationale. |
| EDA-T-015 | Excess categorical levels | Plot/report renders | Tail levels are grouped transparently and totals reconcile. |
| EDA-T-016 | Insufficient cohort support | Cross-tabs render | Sensitive/unstable cells are suppressed or flagged. |
| EDA-T-017 | Failed analysis/output stage | Pipeline exits | Partial output is not presented as a completed report. |
| EDA-T-018 | Complete source schema | Register reconciles | Every source column appears exactly once. |

## 11. Acceptance Criteria

- [ ] SPEC-02 contains approved requirements and no normative placeholders.
- [ ] The raw CSV and input dataframe remain unchanged.
- [ ] One documented command reproduces all EDA artifacts.
- [ ] Every artifact records the source fingerprint and schema version.
- [ ] Every column has an approved role, cutoff status, and leakage decision.
- [ ] Target, numeric, categorical, missingness, relationship, group, anomaly, and redundancy analysis is complete.
- [ ] Statistics show denominators or support where interpretation requires them.
- [ ] Published artifacts contain no raw sensitive identifiers or unsafe row-level examples.
- [ ] Findings distinguish errors, warnings, hypotheses, and accepted limitations.
- [ ] EDA performs no implicit cleaning, row removal, preprocessing, feature selection, or model fitting.
- [ ] Artifact schemas and non-volatile content are deterministic.
- [ ] Unit, integration, regression, and lint checks pass.
- [ ] Material risks have owners and documented dispositions.

## 12. Recommended Implementation Order

1. Approve the expanded SPEC-02 contract and prediction-time definitions.
2. Finalize column availability, windows, domains, tiers, and privacy thresholds.
3. Add validated EDA configuration and result schemas.
4. Implement pure distribution and target-analysis functions.
5. Implement missingness, association, redundancy, anomaly, and leakage analysis.
6. Add deterministic visualization and report rendering.
7. Add atomic pipeline/CLI orchestration.
8. Implement unit and integration tests.
9. Generate artifacts for the current source fingerprint.
10. Review findings, assign risk dispositions, and approve SPEC-02.

## 13. Definition of Done

SPEC-02 is complete only when:

- its proposed requirements and policies are approved;
- the validated EDA run produces every required artifact;
- the report documents distributions, relationships, anomalies, missingness, client heterogeneity, leakage, and limitations;
- every column is accounted for and no prohibited field is recommended as a model feature;
- privacy and minimum-support rules are verified;
- reproducibility, calculation, immutability, rendering, and integration tests pass;
- each material issue has a documented downstream action or accepted exception.

## 14. Open Decisions

- Formula and thresholds used to derive `trend_direction` from `trend_pct`.
- Prediction cutoff and exact start/end dates for all 30-day and 90-day windows.
- Valid units/ranges for `ctr`, `engagement_rate`, `scroll_rate`, and `ai_traffic_pct`.
- Parent-child consistency rules for sessions, users, engaged sessions, AI sessions, and scroll events.
- Boundaries and inclusivity for age, freshness, length, impression, and position tiers.
- Minimum support and masking method for client/cohort reporting.
- Rare-category, high-correlation, and anomaly severity thresholds.
- Whether inferential p-values are needed; effect sizes and support are proposed by default.
- Whether `provider_used` remains excluded due to high missingness.
- Whether plots and generated tables are committed or regenerated in CI.
- Approved plotting dependencies and artifact retention policy.
