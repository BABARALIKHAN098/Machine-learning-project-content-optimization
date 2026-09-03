# SPEC-00 — Problem Definition

**Status:** Proposed  
**Owner:** Babar Ali Khan

## Problem

- Business or user problem: Content teams need a consistent way to identify published content whose search performance is likely to decline, so refresh work can be prioritized.
- Intended user: SEO specialists, content strategists, and content operations teams.
- Decision supported by the prediction: Whether a content item should be reviewed and prioritized for a refresh before the next reporting period.
- Prediction target: `trend_direction` with classes `down`, `stable`, `up`, `new`, and `flat`.
- Task type: Multiclass classification.
- What one row represents: One anonymized published content item for one client, with content metadata, search attributes, age/freshness information, and historical performance measurements.
- Prediction timing: At the end of the previous 30-day observation period, before performance in the following 30-day outcome period is known.

## Success Criteria

- Primary model metric: Macro-averaged F1 score, because all five trend classes matter and the observed class distribution is imbalanced.
- Minimum acceptable value: Macro F1 at least 0.45 on a client-grouped holdout set; this threshold is provisional until the baseline is measured.
- Baseline: Most-frequent-class classifier and a stratified random classifier; the trained model must materially outperform both on macro F1.
- Operational constraint: Batch scoring of 30,000 content items must complete on a standard CPU with deterministic, reproducible preprocessing.

## Risks and Constraints

- Data leakage risk: `trend_pct`, current/last-30-day outcome metrics, 90-day aggregates containing the outcome window, and tiers or rates calculated from those metrics can reveal the target. They must be excluded from model inputs. Preprocessing must be fitted only on training data.
- Validation leakage risk: Rows from the same `client_id` may share client-specific patterns. Evaluation must group by client rather than randomly mixing a client's rows across train and test sets.
- Privacy or sensitive attributes: `content_id` and `client_id` are anonymized identifiers. They must not be model features or exposed in prediction responses beyond authorized operational use.
- Cost of false positive: Refreshing content predicted to decline when it would not decline wastes editorial time and may disrupt content that already performs well.
- Cost of false negative: Missing content that will decline can lead to continued traffic loss and delayed recovery; recall for the `down` class must therefore be reported separately.
- Label limitation: The current target describes an observed trend rather than a confirmed causal need for a refresh. Model results should prioritize human review, not automatically trigger publication changes.

## Dataset Evidence

- Source: `content_refresh_anonymized.csv`, copied unchanged to `data/raw/dataset.csv`.
- Shape: 30,000 rows and 44 columns.
- Target distribution: `down` 16,262; `stable` 5,962; `up` 4,388; `new` 2,236; `flat` 1,152.
- Entity coverage: 30,000 unique content IDs across 32 anonymized clients.
- Missingness is present in search metadata, intent, content length, provider/model, and `trend_pct`; configured preprocessing must handle it explicitly.

## Acceptance Criteria

- [x] Target and task type are proposed and documented.
- [x] Prediction timing is defined.
- [x] Success is measurable.
- [x] Intended users, decision, risks, and limitations are documented.
- [ ] Stakeholder approves the proposed target, prediction timing, and minimum metric.
- [ ] A leakage-safe, client-grouped baseline confirms feasibility.
