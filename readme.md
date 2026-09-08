# Content Trend Classification

A leakage-aware, specification-driven machine-learning project that predicts the next-period `trend_direction` of published content to support human refresh prioritization.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Place the immutable source CSV at `data/raw/dataset.csv`. The configured source contains 30,000 rows and 44 columns.

## Reproduce the workflow

```powershell
.\.venv\Scripts\python.exe scripts\validate_data.py
.\.venv\Scripts\python.exe scripts\preprocess_data.py
.\.venv\Scripts\python.exe scripts\train_model.py
.\.venv\Scripts\python.exe scripts\evaluate_model.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

Generated outputs:

- `reports/data_profile.json` and `reports/data_audit.md`
- `artifacts/metadata/split_manifest.json`
- `reports/metrics/model_metrics.json`
- `artifacts/metadata/model_metadata.json`
- `artifacts/models/content_trend_pipeline.joblib`

## Exploratory data analysis

Run the deterministic, leakage-aware EDA workflow with:

```powershell
.\.venv\Scripts\python.exe scripts\run_eda.py `
  --data-config configs/data.yaml `
  --eda-config configs/eda.yaml
```

The workflow validates the source first, records and rechecks its SHA-256 fingerprint,
then writes versioned tables to `reports/eda/` and aggregate PNG figures to
`reports/figures/eda/`. Client identifiers are replaced with stable aliases in cohort
outputs and excluded from categorical summaries. EDA is descriptive and does not modify,
impute, remove, or select records or features.

An executable walkthrough of the complete SPEC-01 and SPEC-02 workflow is available at
`notebooks/SPEC-01-02-data-preparation-and-eda.ipynb`.

## Reproducible grouped splitting

Create the versioned, privacy-safe split manifest and row assignment artifact with:

```powershell
.\.venv\Scripts\python.exe scripts\split_data.py `
  --data-config configs/data.yaml `
  --training-config configs/training.yaml
```

The splitter assigns complete `client_id` groups, proves exact `content_id` coverage and
disjointness, preserves all target classes, enforces configured quality tolerances, and
writes only hashed row keys and client aliases to `artifacts/metadata/`. The current split
tests unseen-client generalization on a snapshot; it is not a future-period evaluation.

The notebook walkthrough also contains the live SPEC-03 split contract, manifest generation,
invariant assertions, and train-only preprocessing example.

For a focused executable walkthrough, use
`notebooks/SPEC-03-data-splitting-and-leakage-control.ipynb`.

## Data contract and preprocessing

`configs/data.yaml` is the strict column-role contract. Every source column must be the
target, an identifier, an approved numeric/categorical feature, or an explicitly dropped
field. Unexpected columns, missing targets, duplicate required-unique identifiers, and
non-numeric values in numeric features fail validation instead of being removed silently.

The validation command records a byte-level source SHA-256 and writes a versioned profile
with each column's role, model eligibility, and quality warnings. It rechecks the source
fingerprint after profiling.

To verify the split-before-fit preprocessing lifecycle:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_data.py `
  --data-config configs/data.yaml `
  --preprocessing-config configs/preprocessing.yaml `
  --training-config configs/training.yaml
```

This command fits median imputation, categorical imputation, one-hot vocabularies, and
optional scaling on the training partition only. Validation and test partitions are only
transformed. The raw CSV and source dataframe are not modified, and transformed datasets
are not persisted by this phase.

## Feature engineering and selection

The v2 feature contract is in `configs/features.yaml`. Ratios use previous-period inputs;
undefined ratios remain missing with explicit indicators. Optional log transforms and
feature-family ablations are implemented inside serializable model pipelines. The raw CSV
schema stays unchanged. Historical availability of mutable metadata remains unverified.

Run the fixed study on the recorded split (run `scripts/split_data.py` first if its artifacts
do not exist):

```powershell
.\.venv\Scripts\python.exe scripts\engineer_features.py --output-dir reports/features
.\.venv\Scripts\python.exe scripts\train_model.py --feature-manifest reports/features/feature_manifest.json
```

The study writes a feature catalog, training-only diagnostics, grouped-fold metrics,
validation confirmations, frozen per-estimator configurations, and a source/split/config/code
manifest under `reports/features/`. Later training verifies and reuses that manifest.
Feature transforms are saved with the estimator, so inference needs only raw inputs.

Training now fits on train and reports validation metrics. It does not refit on development,
evaluate test, or benchmark all source rows. `evaluate_model.py` displays the applicable
stored validation report; it can still display historical test reports. Final test evaluation
and repeated-access auditing remain a separate SPEC-03 deliverable. The existing holdout
was previously evaluated and is not a fresh untouched test set.

The frozen study compares five variants and two fixed estimators over three client-grouped
folds, with one validation confirmation. It does not tune models or change thresholds in
response to results. See `reports/features/selection_report.md` and
`plan/SPEC-04-feature-engineering-and-selection/implementation-plan.md` for the protocol.

## Inference

```python
import pandas as pd

from machine_learning_project.inference.predictor import Predictor

model = Predictor.load("artifacts/models/content_trend_pipeline.joblib")
rows = pd.read_csv("data/raw/dataset.csv")
predictions = model.predict(rows)          # preserves input order
review_queue = model.rank_review_queue(rows)  # highest probability of decline first
```

Predictions support human review only. They must not trigger automatic content changes. See `reports/MODEL_CARD.md` for intended use and limitations.

## Current result

The selected random forest achieved test macro F1 of 0.4215 and `down` recall of 0.6603 on a client-grouped holdout. Because macro F1 is below the provisional 0.45 threshold, the current model remains experimental.


## Development baseline benchmark (SPEC-05)

Run `.\.venv\Scripts\python.exe scripts/run_baselines.py` to publish the verified,
training-label-only majority and stratified benchmark under `reports/baselines`.
Existing source and split artifacts are required; the command never regenerates splits.
Configuration is owned by `configs/baselines.yaml`: canonical seed 42, fixed repeats 42-46,
and a 0.01 absolute macro-F1 improvement margin. Fit and prediction timings are separate.

Run `.\.venv\Scripts\python.exe scripts/verify_baselines.py` for tests, lint and two-run
real-data verification. See `reports/baselines/verification.json` for actual outcomes.
For SPEC-06, `scripts/train_model.py --baseline-manifest reports/baselines/baseline_manifest.json`
reuses verified frozen evidence. Source, split, ordered rows and metric contracts must match;
legacy reports without this provenance cannot be compared. Candidate reports retain
`baselines_validation` and add independent improvement, recall and project-target flags.
Selection for development does not imply that these flags passed or authorize promotion.

Validation was previously used for feature confirmation and historical test results exist.
This benchmark is developmental. Random-seed dispersion is not a generalization confidence
interval, and dummy-model timing does not establish full-pipeline performance.


## Grouped model tuning (SPEC-06)

Run `python scripts/train_model.py --tuning-config configs/tuning.yaml --feature-manifest
reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json
--run-id spec06-reference` in the project environment. Add `--dry-run` for read-only preflight.
The fixed protocol performs 15 configurations across three client-grouped training folds,
then two train-only finalist refits: 47 fits. Frozen feature/baseline manifests are mandatory.

Outputs use `reports/training/<run_id>` and `artifacts/models/training/<run_id>`; existing run
IDs are rejected. Both finalist bundles remain development-only. Choices use training CV,
then validation reports independent baseline-improvement/recall/target flags without retuning.
Legacy training without `--tuning-config` remains available.

`python scripts/verify_training.py` runs tests/lint and two isolated real searches (94 fits),
checking semantic reproducibility, bundle reloads and protected prerequisite/model/report bytes.
Do not also run the same two searches manually before invoking this verifier. Use a new
`--run-prefix` for later verification runs. SPEC-07 can consume both finalists with
`load_training_run(report_dir, model_dir)` after hash checks. No test evaluation or promotion runs.

These scores reuse development evidence and frozen feature selection; they are not a new
independent or nested generalization estimate. Historical cutoff uncertainty remains unresolved.
