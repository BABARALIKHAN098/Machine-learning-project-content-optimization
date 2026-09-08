# SPEC-06 development training

CV preference: random_forest; recall fallback: False.
15 configurations, three shared client-grouped training folds, 47 estimator fits.
Mean fold macro F1 determines choices; 0.001 ties use declared complexity and stable IDs.

| Family | CV macro F1 | Validation macro F1 | Validation down recall | Material improvement | Recall | Target |
| --- | ---: | ---: | ---: | --- | --- | --- |
| logistic_regression | 0.411854 | 0.389920 | 0.768068 | True | True | False |
| random_forest | 0.443426 | 0.430224 | 0.861367 | True | True | False |

Both finalists are fitted on train only. Validation cannot restart the search.
CV uses unweighted fold means and population SD; worst-fold recall is also recorded.
Frozen feature selection used broader training evidence: this is not nested evaluation.
Validation informed earlier feature confirmation; historical test evaluation exists.
These are development results, not independent generalization estimates or promotion decisions.
Historical metadata cutoff evidence remains unresolved. Human review is required.
Timings exclude any production SLA claim; the optional 30,000-row throughput test was not run.
SPEC-07: consume both finalists via load_training_run with the report and model directories.
SPEC-08: production packaging and promotion remain separate.
