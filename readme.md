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

## Historical holdout result

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

## Frozen evaluation and error analysis (SPEC-07)

`python scripts/evaluate_model.py --evaluation-config configs/evaluation.yaml --dry-run`
verifies the frozen SPEC-06 reference, source/splits, feature choices, baseline and configuration
without predictions or writes. Remove `--dry-run` and supply a fresh `--run-id` to replay both
finalists once on validation and publish private aggregate error evidence under reports/evaluation.
The default reference paths can be overridden with `--training-report-dir`, `--model-dir`,
`--feature-manifest` and `--baseline-manifest`. Legacy `--metrics` display remains available.

`python scripts/verify_evaluation.py` runs full tests and Ruff, then a dry run and two instrumented
replays with zero fitting and four complete validation prediction calls. It writes verification.json
beside each run's manifest. Use fresh `--first-run` and `--second-run` names on subsequent invocations;
do not pre-create the same runs with the analysis CLI.

Reports include class/down errors, paired disagreement counts, support-aware slices, descriptive
client omission sensitivity, confusion plots and a hashed SPEC-08 decision. Category labels are
anonymous frequency-ordered aliases; suppressed rows remain in coverage totals. No row examples,
client leaderboard, outcome-derived slices, retraining or test scoring are performed.
Eligibility, the frozen CV reference and review blockers remain separate. A null recommendation
is a valid rejection outcome; production_ready is always false. Validation has been reused and
historical test evaluation exists, so replay is not independent generalization evidence.


Verified SPEC-07 evidence: [evaluation report](reports/evaluation/spec07-reference/evaluation_report.md),
[decision / SPEC-08 handoff](reports/evaluation/spec07-reference/decision.json), and
[verification](reports/evaluation/spec07-reference/verification.json). Full suite: 198 passed; Ruff passed.
Both real replays exactly reproduce SPEC-06 with unchanged prerequisites. Validation macro F1 is
0.389920 for logistic regression and 0.430224 for random forest, below the unchanged 0.45 target.
Neither is recommended; random forest remains the research development reference.

## Research model packaging and local inference (SPEC-08)

SPEC-08 packages the two frozen finalists without refitting. The current SPEC-07
recommendation remains null and neither model is production-ready. See the
[adopted contract](specs/SPEC-08-model-packaging-and-inference.md) and
[SPEC-09 handoff](reports/packaging/SPEC-09-handoff.md).

From an activated environment in Bash (or use `.venv/Scripts/python.exe` explicitly):

```bash
python scripts/package_model.py --purpose research --run-id my-research-run --dry-run
python scripts/package_model.py --purpose research --run-id my-research-run
python scripts/predict_batch.py --package-dir artifacts/packages/my-research-run-random_forest --purpose research --input-path data/examples/synthetic_inference.csv --output-path reports/private-inference/example.json
```

Choose a fresh run ID and output filename; completed artifacts are never overwritten.
Both families are packaged by default. Add `--include-probabilities` to batch inference
only when uncalibrated model scores are needed. `--chunk-rows` controls bounded prediction
chunks; the whole request must contain at most 30,000 rows. JSON envelopes preserve
request order, IDs, original model version, package identity and manifest hash.

```python
from machine_learning_project.inference.contracts import read_request_csv
from machine_learning_project.inference.packaged_predictor import PackagedPredictor

predictor = PackagedPredictor.load(
    "artifacts/packages/my-research-run-random_forest", purpose="research"
)
frame = read_request_csv(
    "data/examples/synthetic_inference.csv", predictor.schema, predictor.config
)
result = predictor.predict(frame)  # No probability calls in the default mode.
```

Only trusted local packages are supported. A package contains the unchanged model,
schemas, decision, model card, exact environment requirements and a project wheel.
Compatibility requires the exact Python/OS/architecture, dependency versions and
project source digest in `environment.json`. For an already provisioned compatible
runtime, install the bundled wheel with `python -m pip install --no-index --no-deps
<package>/runtime/machine_learning_project-0.1.0-py3-none-any.whl`. Provision exact
dependencies from a local wheelhouse using `python -m pip install --no-index
--find-links <wheelhouse> -r <package>/runtime/requirements.txt`. The project wheel
does not contain those dependencies. No dependency downloads happen during packaging.

The verifier installs the project wheel offline, disables site/editable hooks, reuses
existing dependency files and denies reads from upstream data/reports/model paths.
This verifies standalone project imports on this environment; it does not certify a
fresh machine dependency installation. Package hashes check integrity; a separately
trusted `expected_manifest_sha256` can pin the package manifest.

```bash
python scripts/verify_packaging.py --run-prefix my-verification
```

The verifier runs tests and lint, builds twice with one frozen wheel, checks validation-only
parity, audits fitting/inference calls and measures a synthetic 30,000-row workload.
Private response files are ignored by Git. The supplied example uses invented IDs
and feature values. Existing `Predictor` and `run_batch_inference` remain legacy APIs.

## Content Trend frontend

The HTML/CSS and vanilla JavaScript workspace is served at `/` by the local research API.
It supports schema-driven single-content input, strict JSON batches, optional model scores,
validation, searchable/paginated results, and explicit full-response JSON downloads.
The technical console remains at `/docs`. No frontend build step or Node runtime is needed
to use the application.

Start the API from a provisioned compatible environment with a complete local model package.
The following PowerShell example generates a token in the current terminal and uses the
verified reference random-forest package and manifest pin. Enter the same token in the
workspace's masked connection field; never commit it or put it in a URL.

```powershell
$env:CONTENT_TREND_API_TOKEN = & .venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
.venv/Scripts/python.exe scripts/serve_api.py --purpose research --package-dir artifacts/packages/spec08-final-reference-random_forest --expected-manifest-sha256 17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6 --port 8000
```

Open **http://127.0.0.1:8000/**. To transfer the token on Windows, run
`Set-Clipboard -Value $env:CONTENT_TREND_API_TOKEN` in the same terminal before launching
the server, paste it into the workspace, and clear the clipboard afterward. The token is
kept in page memory only and is cleared by **Clear session** or a page reload.

Use the API's own origin: a separate Live Server port, `file://`, or an externally hosted
frontend is incompatible with the current host/origin policy. Package selection and token
configuration remain server-side operator tasks; the UI cannot switch models.

1. Connect, then choose **Single content** or **JSON batch**. **Load synthetic example**
   loads invented inputs from the running model's schema.
2. Supply a content ID and feature values. Every feature key is required. Select **Missing**
   to send JSON `null`; numeric zero and empty category strings remain real values.
   Unknown categories are accepted. IDs must be unique and have no surrounding whitespace.
3. For batches, import/paste the API request envelope with `purpose: "research"`,
   `records`, and an optional boolean `include_probabilities`. CSV is not supported.
   Duplicate JSON keys, extra/missing fields, invalid types, and active row/byte limits
   are checked before submission.
4. Validate and run. **Include model scores** is off by default. Scores are uncalibrated
   and non-causal; they do not estimate the benefit of editing content.
5. Review labels, filter or search IDs, and open record details. **Download results JSON**
   saves the complete original response, even when a filter is active.

No tokens, inputs, or predictions are written to browser storage. Downloads happen only
when requested. Clearing a session stops the browser waiting; a submitted server prediction
may continue. A busy service requires an explicit retry. A prediction/output failure can
make readiness fail even while the health endpoint succeeds, requiring an operator restart.

Frontend verification, using Node 22+ for development tests only:

```text
node --test tests/frontend/contracts.test.mjs
python -m pytest tests/contract/test_frontend_routes.py tests/contract/test_api_contract.py
python -m ruff check app tests scripts
```

The complete browser runner is `node tests/frontend/browser.mjs`. It requires an installed
Playwright module, installed Chrome, the compatible Python environment, both complete
reference packages, and free local port 8127. Set `PLAYWRIGHT_MODULE` to an existing
Playwright module directory if it is not resolvable by Node; set `FRONTEND_TEST_PYTHON`
to override `.venv/Scripts/python.exe`. It starts and stops its own local servers, uses
ephemeral tokens and invented data, checks both model families including 30,000-row batches,
and saves screenshots and aggregate evidence in ignored `.frontend-work/`.
Use `node tests/frontend/browser.mjs --ui-only` for the smaller keyboard, error-focus,
null/zero serialization, contrast, reduced-motion, and zoom verification run.

See the [implementation plan](plan/frontend-html-css/implementation-plan.md) and
[implementation handoff](plan/frontend-html-css/implementation-handoff.md).
