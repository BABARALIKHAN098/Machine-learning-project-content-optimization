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
