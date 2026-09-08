# Baseline benchmark

Development validation only. No feature or candidate model fitting.
Training: 17670 rows / 18 clients. Validation: 5857 rows / 7 clients.

| Baseline | Macro F1 | Weighted F1 | Accuracy | Balanced accuracy | Down recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| most_frequent | 0.136794 | 0.355472 | 0.519720 | 0.200000 | 1.000000 |
| stratified | 0.208024 | 0.362706 | 0.370326 | 0.208457 | 0.569317 |

Reference seed: 42; repeat seeds: [42, 43, 44, 45, 46].
Stratified macro F1 mean 0.197043, population SD 0.005586, range 0.192753 to 0.208024.
Seed dispersion describes classifier randomness, not a generalization confidence interval.

Training majority: down (ties: lexicographically first). Class priors use training labels only.
A majority-down model has down recall 1 despite failing to discriminate other classes.
Per-class metrics and ordered confusion matrices are in baseline_metrics.json. Unpredicted classes retain zero precision/recall/F1; main partitions contain every class.

Adopted material improvement: +0.010 absolute macro F1 over majority, canonical stratified and stratified mean. Recall and project-target checks are separate from this comparison.

Fit/prediction timings are diagnostic and exclude metrics computation; dummy timing does not establish the full model's 30,000-row runtime.
The validation set was previously used for feature confirmation and the test holdout was previously evaluated. These results are a development benchmark, not an independent future-period estimate. Historical metadata cutoff evidence remains unverified.
