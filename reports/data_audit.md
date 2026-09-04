# Dataset Audit

- Rows: 30,000
- Columns: 44
- Duplicate rows: 0
- Source SHA-256: `c43bdac4eccfa17fcd8a33974fe36f2c998c03a3ae3af8d80cf712abab5d6396`

| Column | Role | Type | Missing | Missing % | Unique | Constant | Warnings |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| content_id | identifier | str | 0 | 0.00 | 30000 | False | high_cardinality |
| client_id | identifier | str | 0 | 0.00 | 32 | False | - |
| search_volume | feature | float64 | 2468 | 8.23 | 41 | False | - |
| competition | feature | float64 | 2468 | 8.23 | 101 | False | high_cardinality |
| competition_level | feature | str | 2610 | 8.70 | 3 | False | - |
| cpc | feature | float64 | 2468 | 8.23 | 915 | False | high_cardinality |
| content_type | feature | str | 0 | 0.00 | 3 | False | - |
| main_intent | feature | str | 2374 | 7.91 | 4 | False | - |
| word_count | feature | float64 | 7699 | 25.66 | 5476 | False | high_cardinality |
| char_count | feature | float64 | 7699 | 25.66 | 14839 | False | high_cardinality |
| provider_used | dropped | str | 21438 | 71.46 | 2 | False | high_missingness |
| model_used | feature | str | 5733 | 19.11 | 5 | False | - |
| impressions_90d | dropped | int64 | 0 | 0.00 | 9438 | False | high_cardinality |
| clicks_90d | dropped | int64 | 0 | 0.00 | 477 | False | high_cardinality |
| pageviews_90d | dropped | int64 | 0 | 0.00 | 856 | False | high_cardinality |
| sessions_90d | dropped | int64 | 0 | 0.00 | 666 | False | high_cardinality |
| users_90d | dropped | int64 | 0 | 0.00 | 644 | False | high_cardinality |
| engaged_sessions_90d | dropped | int64 | 0 | 0.00 | 68 | False | - |
| ai_sessions_90d | dropped | int64 | 0 | 0.00 | 35 | False | - |
| scroll_events_90d | dropped | int64 | 0 | 0.00 | 155 | False | high_cardinality |
| days_with_impressions | dropped | int64 | 0 | 0.00 | 88 | False | - |
| days_with_sessions | dropped | int64 | 0 | 0.00 | 90 | False | - |
| impressions_last_30d | dropped | int64 | 0 | 0.00 | 5182 | False | high_cardinality |
| clicks_last_30d | dropped | int64 | 0 | 0.00 | 239 | False | high_cardinality |
| sessions_last_30d | dropped | int64 | 0 | 0.00 | 359 | False | high_cardinality |
| impressions_prev_30d | feature | int64 | 0 | 0.00 | 5931 | False | high_cardinality |
| clicks_prev_30d | feature | int64 | 0 | 0.00 | 258 | False | high_cardinality |
| sessions_prev_30d | feature | int64 | 0 | 0.00 | 311 | False | high_cardinality |
| content_age_days | feature | int64 | 0 | 0.00 | 225 | False | high_cardinality |
| age_tier | feature | str | 0 | 0.00 | 4 | False | - |
| age_tier_order | feature | int64 | 0 | 0.00 | 4 | False | - |
| days_since_last_update | feature | int64 | 0 | 0.00 | 57 | False | - |
| freshness_tier | feature | str | 0 | 0.00 | 4 | False | - |
| word_count_tier | feature | str | 7699 | 25.66 | 4 | False | - |
| char_count_tier | feature | str | 7699 | 25.66 | 4 | False | - |
| ctr | dropped | float64 | 0 | 0.00 | 401 | False | high_cardinality |
| avg_position | dropped | float64 | 0 | 0.00 | 869 | False | high_cardinality |
| engagement_rate | dropped | float64 | 0 | 0.00 | 915 | False | high_cardinality |
| scroll_rate | dropped | float64 | 125 | 0.42 | 1773 | False | high_cardinality |
| ai_traffic_pct | dropped | float64 | 0 | 0.00 | 518 | False | high_cardinality |
| impression_tier | dropped | str | 0 | 0.00 | 4 | False | - |
| position_tier | dropped | str | 0 | 0.00 | 5 | False | - |
| trend_direction | target | str | 0 | 0.00 | 5 | False | - |
| trend_pct | dropped | float64 | 3388 | 11.29 | 2712 | False | high_cardinality |
