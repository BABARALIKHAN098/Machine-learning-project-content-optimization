# Feature selection report

Development-only fixed family study.

The existing test holdout was previously evaluated; no test rows were used here.
Snapshot data and unverified metadata timing do not establish future performance.
Undefined ratios remain missing with indicators; empty numeric columns use zero.

- logistic_regression: selected C; grouped-fold choice C; validation promotion passed: True.

| Variant | Mean fold macro F1 | Mean down recall | Worst fold down recall | Mean width |
| --- | ---: | ---: | ---: | ---: |
| A | 0.216730 | 0.361139 | 0.137335 | 45.3 |
| B | 0.379102 | 0.540296 | 0.368260 | 49.3 |
| C | 0.401598 | 0.614404 | 0.551712 | 56.3 |
| D | 0.361141 | 0.497678 | 0.355183 | 30.3 |
| E | 0.388441 | 0.410203 | 0.080845 | 44.7 |

Validation confirmation:

| Variant | Macro F1 | Down recall |
| --- | ---: | ---: |
| A | 0.239934 | 0.403745 |
| C | 0.385857 | 0.802562 |

- random_forest: selected E; grouped-fold choice E; validation promotion passed: True.

| Variant | Mean fold macro F1 | Mean down recall | Worst fold down recall | Mean width |
| --- | ---: | ---: | ---: | ---: |
| A | 0.410372 | 0.756901 | 0.637504 | 45.3 |
| B | 0.410309 | 0.802466 | 0.687691 | 49.3 |
| C | 0.407832 | 0.811694 | 0.692777 | 56.3 |
| D | 0.393499 | 0.826922 | 0.709732 | 30.3 |
| E | 0.413265 | 0.811450 | 0.769413 | 44.7 |

Validation confirmation:

| Variant | Macro F1 | Down recall |
| --- | ---: | ---: |
| A | 0.397522 | 0.933968 |
| E | 0.400468 | 0.924113 |


All variants and fold dispersion are recorded in feature_manifest.json.
Variants not selected were rejected by the frozen guardrail, complexity, or validation confirmation rule. No threshold was relaxed.
