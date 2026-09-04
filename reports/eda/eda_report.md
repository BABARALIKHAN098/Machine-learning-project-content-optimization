# Exploratory Data Analysis

## Run manifest

- Rows: 30,000
- Columns: 44
- Source SHA-256: `c43bdac4eccfa17fcd8a33974fe36f2c998c03a3ae3af8d80cf712abab5d6396`
- Artifact schema: `1.0`

## Target distribution

| Class | Rows | Proportion |
| --- | ---: | ---: |
| down | 16,262 | 54.21% |
| flat | 1,152 | 3.84% |
| new | 2,236 | 7.45% |
| stable | 5,962 | 19.87% |
| up | 4,388 | 14.63% |

Imbalance ratio: 14.116. Entropy: 1.808 bits.

## Material data-quality findings

- `provider_used`: 21,438 missing (71.46%).
- `word_count_tier`: 7,699 missing (25.66%).
- `char_count`: 7,699 missing (25.66%).
- `word_count`: 7,699 missing (25.66%).
- `char_count_tier`: 7,699 missing (25.66%).
- `model_used`: 5,733 missing (19.11%).
- `trend_pct`: 3,388 missing (11.29%).
- `competition_level`: 2,610 missing (8.70%).
- `search_volume`: 2,468 missing (8.23%).
- `competition`: 2,468 missing (8.23%).
- `ANOM-RANGE-CTR` (warning): 1,689 violations (5.63%).
- `ANOM-RANGE-ENGAGEMENT_RATE` (warning): 7,852 violations (26.17%).
- `ANOM-RANGE-SCROLL_RATE` (warning): 18,481 violations (61.86%).
- `ANOM-RANGE-AI_TRAFFIC_PCT` (warning): 23 violations (0.08%).

## Relationships and redundancy

- `word_count` / `char_count`: Pearson 0.939, Spearman 0.971.
- `impressions_90d` / `days_with_impressions`: Pearson 0.240, Spearman 0.918.
- `impressions_90d` / `impressions_last_30d`: Pearson 0.918, Spearman 0.952.
- `impressions_90d` / `impressions_prev_30d`: Pearson 0.973, Spearman 0.974.
- `clicks_90d` / `clicks_last_30d`: Pearson 0.947, Spearman 0.867.
- `clicks_90d` / `clicks_prev_30d`: Pearson 0.977, Spearman 0.871.
- `pageviews_90d` / `sessions_90d`: Pearson 0.974, Spearman 0.992.
- `pageviews_90d` / `users_90d`: Pearson 0.971, Spearman 0.991.
- `pageviews_90d` / `days_with_sessions`: Pearson 0.688, Spearman 0.972.
- `pageviews_90d` / `sessions_last_30d`: Pearson 0.907, Spearman 0.869.
- `sessions_90d` / `users_90d`: Pearson 0.998, Spearman 0.999.
- `sessions_90d` / `days_with_sessions`: Pearson 0.692, Spearman 0.979.
- `users_90d` / `days_with_sessions`: Pearson 0.684, Spearman 0.979.
- `engaged_sessions_90d` / `engagement_rate`: Pearson 0.135, Spearman 0.966.
- `ai_sessions_90d` / `ai_traffic_pct`: Pearson 0.280, Spearman 0.999.

## Leakage review

25 columns are prohibited or unavailable as model inputs. See `leakage_register.csv` for the field-level rationale.

Target-derived and outcome-window associations describe label construction or contemporaneous relationships; they are not evidence of deployable predictive signal.

## Figures

- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/target_distribution.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/missingness.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_search_volume.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_word_count.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_impressions_prev_30d.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_clicks_prev_30d.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_sessions_prev_30d.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_content_age_days.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/distribution_days_since_last_update.png`
- `e:/Portfolio_maker/mL learning project content optimization/reports/figures/eda/spearman_correlations.png`

## Limitations and decisions

- This analysis is descriptive and does not establish causation.
- The source is a snapshot; prospective validity requires timestamped feature snapshots.
- Client identifiers are masked in cohort artifacts and excluded from category summaries.
- Rate domains and target/tier formulas require stakeholder confirmation.
- No row was removed, corrected, imputed, or otherwise mutated by EDA.
