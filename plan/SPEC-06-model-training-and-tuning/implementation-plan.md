# Implementation Plan - Model Training and Tuning

**Source:** `specs/SPEC-06-model-training-and-tuning.md`  
**Status:** Implemented and verified for development on 2026-09-08  
**Prepared:** 2026-09-08  
**Owner:** Babar Ali Khan  
**Outcome:** Reproducibly tune the existing candidate families, retain train-fitted development finalists, and supply versioned evidence to SPEC-07.

## 1. Specification status and objective

SPEC-06 is a draft whose scope, requirements, tests and decisions are `TBD`. This plan proposes a concrete protocol using the approved problem definition and implemented SPEC-03/04/05 contracts. Search settings and selection rules below are proposed defaults, not already approved requirements or completed experiments. Adopt them in SPEC-06 when implementation is authorized. This planning task changes no specification, code, configuration, dataset, model or benchmark.

Tune logistic regression and random forest on client-grouped folds within the existing training partition. Freeze one configuration per family and a cross-family development preference from training-fold evidence, then evaluate the two finalists on the existing validation population. Report baseline/recall/target flags independently. SPEC-07 owns broader error analysis and the eventual selection decision; SPEC-08 owns production packaging.

Successful execution means valid, reproducible evidence, regardless of whether a candidate reaches macro F1 0.45. Do not lower thresholds, expand the search after inspecting validation, or conflate weak scores with execution errors.

## 2. Current implementation and dependency evidence

Package-relative paths below are under `src/machine_learning_project/`; root `pipelines/` is the existing orchestration location.

| Area | Current behavior | Required change |
| --- | --- | --- |
| `models/train.py` | Builds logistic regression and random forest pipelines with optional cutoff-safe engineering; reads fixed `training.candidates` settings. | Validate settings and apply allowlisted trial overrides to fresh pipelines. Preserve estimator-specific preprocessing. |
| `models/tune.py` | `select_candidate` ranks validation macro F1, weighted F1 and name; prefers recall-eligible models, otherwise falls back. No search exists. | Add deterministic training-fold ranking and structured eligibility/fallback reasons; retain the legacy API. |
| `pipelines/training_pipeline.py` | Loads development rows and optional frozen manifests, fits fixed candidates on train, scores validation, writes one bundle to configured paths. | Add a tuning path that freezes choices before validation, retains both finalists and publishes isolated artifacts. |
| `scripts/train_model.py` | Has feature/baseline options but no search mode, tuning config or run directory. | Add explicit tuning options and read-only preflight; preserve fixed-training invocation. |
| `features/artifacts.py` | Frozen input fingerprint includes data/preprocessing, original candidate settings and random seed. | Verify the original contract before applying separate trial overrides; do not weaken fingerprint checks. |
| `pipelines/feature_pipeline.py` | Already uses three seeded `StratifiedGroupKFold` folds within train and fold-local fitting. | Reuse the pattern without rerunning feature-family selection; verify tuning fold identities explicitly. |
| `models/evaluate.py`, `models/benchmark.py` | Fixed-label metric contract 1.0, canonical ordering, frozen baseline integrity and independent comparison flags. | Reuse these contracts and exactly the same validation identity. |
| Publication | Training writes metrics/metadata and joblib directly, potentially replacing previous artifacts. | Isolated staging, payload hashes and manifest-last completion. |
| Tests | Existing guardrail, baseline integration, feature and inference tests. | Add search/config/fold-isolation/selection/failure/publication/reproducibility coverage. |

Frozen choices in `reports/features/selected_features.json`:

- Logistic regression: ratio and log families; no excluded inputs.
- Random forest: ratio family; `model_used` excluded.
- `configs/features.yaml` does not represent both choices. Tuning must require the estimator-specific feature manifest rather than apply the shared default to both models.

The published baseline report records train 17,670 rows / 18 clients and validation 5,857 rows / 7 clients. Macro F1: majority 0.136794; seed-42 stratified 0.208024; five-seed stratified mean 0.197043. These are readings of existing evidence, not new measurements. Resolve runtime counts, identities and full-precision scores from verified artifacts, never constants.

Prerequisites: validated source, SPEC-03 split manifest/assignments, SPEC-04 feature manifest/payloads, SPEC-05 baseline manifest/config/payloads and the project environment. Missing or stale inputs fail with an instruction to run the relevant existing workflow explicitly; never regenerate dependencies inside tuning.

## 3. Scope and scientific boundaries

### In scope

- A small predetermined search over the two existing model families.
- Three client-grouped training folds shared by all trials.
- Fresh fold-local engineering, imputation, encoding, scaling and estimator fitting.
- Deterministic scoring/ranking, convergence diagnostics and bounded CPU work.
- Train-only refits and one validation evaluation per finalist.
- Frozen baseline comparisons and a SPEC-07 handoff containing both finalists.
- Isolated versioned artifacts, reload checks, reproducibility and privacy verification.
- Optional clearly scoped development batch-throughput diagnostics.

### Out of scope

New model families, external search services, GPU work, feature selection/ablations, altered labels/leakage rules, new outer splits, resampling/SMOTE, validation-driven search, calibration, threshold tuning, subgroup/error analysis, interpretation studies, train-plus-validation refitting, test scoring/timing, deployment and production model replacement.

Validation already informed SPEC-04 confirmation; historical test results also exist. Training CV is a selection tool and is optimistic after choosing the best trial. Frozen features were selected using broader training evidence, so this is not nested evaluation of the whole feature-and-parameter selection procedure. Fold dispersion is descriptive, not a confidence interval. Neither CV nor reused validation is a new independent generalization estimate. Historical metadata cutoff evidence remains unresolved; tuning cannot prove that uncertain fields existed at prediction time.

## 4. Proposed requirements and traceability

| Requirement | Behavior | Tests |
| --- | --- | --- |
| SPEC-06-REQ-001 | Validate families, parameter grids, versions, seeds, folds and budgets before fitting. | TRAIN-T-001/002 |
| SPEC-06-REQ-002 | Verify source/split/frozen feature/baseline evidence without regeneration. | TRAIN-T-003/004 |
| SPEC-06-REQ-003 | Separate trial overrides from the original feature-study input contract. | TRAIN-T-005 |
| SPEC-06-REQ-004 | Reuse reproducible client-disjoint training folds with class coverage. | TRAIN-T-006/007 |
| SPEC-06-REQ-005 | Fit complete pipelines only on fold-training rows and exclude identifiers/outcomes. | TRAIN-T-008/009 |
| SPEC-06-REQ-006 | Record every trial and fixed-label fold metrics. | TRAIN-T-010/011 |
| SPEC-06-REQ-007 | Freeze deterministic CV finalist choices with explicit eligibility/fallback reasons. | TRAIN-T-012/013 |
| SPEC-06-REQ-008 | Refit on train only; evaluate validation after freezing choices. | TRAIN-T-014/015 |
| SPEC-06-REQ-009 | Report compatible baseline comparisons and independent thresholds. | TRAIN-T-016 |
| SPEC-06-REQ-010 | Bound work and expose warnings, timings and incomplete runs. | TRAIN-T-017/018 |
| SPEC-06-REQ-011 | Publish private aggregate evidence and reloadable verified bundles. | TRAIN-T-019/020/021 |
| SPEC-06-REQ-012 | Reproduce semantic outputs and preserve existing protected artifacts. | TRAIN-T-022/023 |
| SPEC-06-REQ-013 | Preserve legacy fixed-training behavior and expose explicit tuning mode. | TRAIN-T-024 |
| SPEC-06-REQ-014 | Document scientific limits and hand off frozen evidence to SPEC-07/08. | TRAIN-T-025 |

## 5. Proposed configuration and bounded search

Add `configs/tuning.yaml` with a `tuning` mapping. Keep `configs/training.yaml` as the original model/split contract, and preserve existing preprocessing/feature/baseline configuration ownership. Read the project target and recall guardrail from training config and the improvement margin from baseline config.

| Setting | Proposed policy |
| --- | --- |
| `tuning_contract_version`, `artifact_schema_version` | `1.0` |
| `families` | `[logistic_regression, random_forest]`; both required |
| `search_method` | Exhaustive small fixed grid |
| `folds`, `shuffle`, `random_seed` | 3, true, 42 |
| `primary_metric` | Fixed-label macro F1; reject conflicts with training config |
| `selection_tolerance` | 0.001 absolute mean CV macro F1 |
| `search_n_jobs`, `estimator_n_jobs` | 1, 1; no nested parallelism |
| `max_configurations`, `max_estimator_fits` | 15 and 47 per complete run |
| `convergence_policy` | Exclude nonconverged trials from finalist eligibility; retain diagnostics |
| `evaluation_partition` | `validation` only |
| `require_feature_manifest`, `require_baseline_manifest` | true in tuning mode |
| `output_root`, `model_output_root` | `reports/training`, `artifacts/models/training`; isolated run subdirectories |
| `resume` | false in version 1.0; incomplete runs require a new run ID |

### Logistic regression: three configurations

Search `model__C`: `[0.1, 1.0, 10.0]`. Fix `class_weight=balanced`, solver `lbfgs`, maximum iterations 1000 and seed 42. Record the installed supported regularization defaults and effective parameters; verify local API compatibility during implementation. Preserve numeric scaling and categorical handling from the verified preprocessing contract.

Do not silently increase iterations after observing convergence. If inadequate, publish diagnostics and propose a separately versioned protocol change.

### Random forest: twelve configurations

Search `model__min_samples_leaf`: `[1, 2, 5]`; `model__max_features`: `[sqrt, 0.5]`; `model__max_depth`: `[null, 20]`. Fix `n_estimators=200`, `class_weight=balanced_subsample`, seed 42 and estimator `n_jobs=1`. Preserve unscaled numeric preprocessing and the frozen feature exclusion.

Both grids include the original model settings as controls. Calculate before fitting: `(3 + 12) * 3 = 45` fold fits plus two train-only finalist refits = **47 fits**. Two reproducibility runs cost up to 94 fits. Frozen baseline reuse adds no fits. These are estimator-call bounds, not wall-clock promises.

Reject unknown parameter names, duplicate grid values/configurations, empty/non-list grids, booleans masquerading as integers, nonfinite/out-of-range numbers, unsupported model combinations and insufficient budgets. Version 1.0 uses the fixed adopted schedule; changes require an explicitly recorded protocol/config revision before evaluation, never automatic expansion. Derive stable trial IDs from normalized family/parameter mappings, not timing or filesystem order.

## 6. Loading, folds and leakage control

1. Validate configuration, output locations and the full schedule before constructing estimators.
2. Use `load_development` and `prepare_benchmark_partitions` for verified canonical train/validation rows. Upstream full-source integrity checks are allowed; no test dataframe reaches tuning.
3. Call `load_frozen_configs` with the **unmodified original** data, preprocessing and training configs.
4. Call `load_benchmark` with the canonical validation identity before expensive fitting.
5. Build per-family templates from frozen features and original model settings. Apply allowlisted trial overrides to fresh clones only. Record original feature-study contract and effective trial hashes separately. Never mutate the original config or bypass its fingerprint check.
6. Create three shuffled, seed-42 `StratifiedGroupKFold` folds from canonical train rows. Check enough clients, disjoint row/client membership, every train row scored exactly once, and all five classes in every fit/scoring subset. Grouped stratification does not guarantee coverage: fail if absent rather than trying other seeds.
7. Record aggregate counts/supports and project-scoped ordered/membership hashes, not raw client/content IDs. Every trial uses exactly this fold list.
8. Fit the complete fresh pipeline inside each fold. No whole-training imputer/encoder/scaler fitting before CV and no shared fitted state between folds/trials. A frozen feature configuration is allowed; a shared fitted transformer is not.
9. Predict each complete scoring fold once and reuse its vector for all metrics. Keep row predictions in memory; publish only metrics and scoped audit fingerprints.

The tuning core receives training features, labels and groups only. Keep validation confirmation in a separate orchestration step so premature validation access is structurally excluded and testable.

## 7. Scoring, selection and validation confirmation

### Fold metrics

Reuse `evaluate_predictions` for accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1/support, confusion matrix and down false negatives. Record down recall explicitly, effective parameters, transformed width, convergence status and sanitized diagnostics. Separate fit/predict timing from scoring and publication.

Aggregate mean, population SD (`ddof=0`), minimum and maximum for fold macro F1 and down recall. Rank on the unweighted mean across three folds; disclose differing row/client counts. If pooled out-of-fold scores are provided, label them separately and never substitute them silently. Do not compare CV means with validation-only dummy metrics.

### Deterministic choices

1. Valid trials have three completed folds, finite metrics and no convergence failure. Never rank a partial-fold mean.
2. Recall eligibility requires mean fold down recall >=0.50 from config. Report worst-fold recall separately; mean eligibility does not imply all folds pass.
3. Within each family prefer valid recall-eligible trials; if none qualifies, select from valid trials with explicit `recall_fallback_used=true`.
4. Find the maximum mean CV macro F1; scores within 0.001 of that maximum are tied. Prefer lower declared complexity, then stable trial ID. Logistic regression prefers smaller C. Forest prefers bounded depth, smaller depth, larger leaf size, then `sqrt` before `0.5`. Timing never breaks ties.
5. Rank the family finalists using the same recall-pool and tolerance policy; prefer logistic regression on a cross-family tie, then stable ID. Freeze `preferred_for_development` before validation scoring. Retain both finalists for SPEC-07.
6. Preserve `select_candidate` for legacy fixed-training callers; tuning uses a distinct structured CV selection result.

If either family has no valid trial, publish incomplete diagnostic evidence without a successful final manifest; do not silently call a one-family search complete. Valid low-recall/low-F1 results remain complete scientific evidence with failed flags.

### Refits and confirmation

Refit each finalist once on all train rows and persist frozen choices before validation scoring. A failed/nonconverged refit fails completion; do not retune on validation. Evaluate both complete canonical validation batches with metric contract 1.0. Reload checks may repeat predictions solely to verify serialization, without changing choices.

Use `compare_candidate` for deltas against majority, canonical stratified and repeat mean, and independent strict-superiority, +0.01 material-improvement, recall >=0.50 and macro F1 >=0.45 flags. Keep CV and validation recall statuses distinct. Failed validation flags do not change the CV preference or restart search. SPEC-07 receives both finalists and the evidence for its decision.

## 8. Resource limits, failures and runtime evidence

Run trials/folds sequentially, with forest and numerical thread pools limited to one where supported in the installed environment. Record actual settings and dependency versions; use existing dependencies where possible. Progress logs contain family/trial/fold and aggregate counts, never row identities.

Malformed configs, incompatible provenance, invalid folds and unsafe outputs fail before fitting. Capture per-trial convergence warnings and fit exceptions. Unknown warnings remain visible; do not suppress all warnings. Remaining scheduled trials may finish for diagnosis after a failed trial, but completion requires a valid finalist from each family. Interruptions/resource exhaustion leave incomplete diagnostics without a completion manifest. Resume/retry and hard subprocess timeouts are deferred; enforce the fit-count budget and honor cancellation.

Record search, refit, prediction and serialization time separately. The current training config has a 30-second batch limit; assessing it requires an explicit benchmark definition. Optional diagnostic: after reload, score 30,000 rows assembled deterministically from development features through the public prediction path. Record CPU/thread settings, row construction, timed stages and batch size. Repeated development rows are only a throughput proxy and may underrepresent diversity; they do not independently establish a production SLA. Never load test rows for timing or use runtime observations to alter this protocol's selection.

## 9. APIs and file-level implementation

| File | Proposed responsibility |
| --- | --- |
| `configs/tuning.yaml` | Versioned grid, fold/ranking policy, budgets and isolated output roots. |
| `utils/config.py` | `validate_tuning_config(tuning, training)` and model parameter validation, including expanded fit count. |
| `models/train.py` | Fresh per-family pipelines with allowlisted overrides; preserve existing `build_candidates` callers. |
| `models/tune.py` | Trial expansion, fold checks, train-only execution, summaries and structured CV selection. |
| `models/training_artifacts.py` (new) | Safe artifact paths, hashes, completion manifest validation and trusted local bundle readback. |
| `pipelines/tuning_pipeline.py` (new) | `run_tuning(...)`: preflight, dependencies, search, train refits, confirmation and publication. |
| `pipelines/training_pipeline.py` | Share narrow finalist-evaluation/metadata helpers where useful; retain fixed-mode compatibility. |
| `scripts/train_model.py` | Optional `--tuning-config`, `--run-id`, output roots and `--dry-run`; tuning requires both manifests. |
| `scripts/verify_training.py` (new) | Focused/full/lint checks, two approved isolated searches, semantic/reload/privacy/immutability verification. |
| Tests, readme, SPEC-06 and this plan | Traceability, usage, limitations and evidence-linked completion. |

Data flow:

`verified development + frozen manifests -> train-only folds/trials -> frozen CV choices -> train refits -> validation + baseline flags -> staged artifacts -> verified manifest -> SPEC-07`

Return a structured result with trial summaries, finalist choices, CV preference, eligibility/fallback reasons, validation comparisons and artifact references. Distinguish execution status from scientific flags; do not overload legacy `threshold_met` to mean full readiness.

## 10. Artifact and provenance contract

Use `reports/training/<run_id>/` and `artifacts/models/training/<run_id>/`. Validate run IDs as safe local names and reject occupied destinations rather than overwrite them. Stage within the corresponding approved roots. Tuning mode must not write the default historical metrics/model paths or any SPEC-03/04/05 outputs.

| Artifact | Required content |
| --- | --- |
| `resolved_config.json` | Original config hashes, complete effective tuning settings, ordered schedule and fit budget. |
| `fold_manifest.json` | Training-only fold identities, algorithm/version/seed, row/group counts and class support. |
| `trial_metrics.json` | Every trial/fold fixed-label metric, effective parameters, status and sanitized diagnostics. |
| `cv_results.csv` | Trial aggregates, convergence, eligibility and stable ranks. |
| `selection.json` | Frozen finalists, CV preference, tie rule, fallback reasons and selection-input hashes. |
| `validation_metrics.json` | Both finalist metrics, evaluation identity and SPEC-05 comparison flags. |
| `timings.json` | Stage timings, environment and optional throughput protocol/results. |
| `training_report.md` | Schedule, results/failures, CV-versus-validation distinction, limitations and handoff. |
| `training_manifest.json` | Versions; source/split/feature/baseline/config/fold/selection hashes; code/environment; report/model hashes. Publish last. |
| `verification.json` | Actual commands/outcomes, fit counts, semantic/reload/privacy/immutability evidence. Written after verification and references the final manifest hash without a circular self-hash. |
| Model root: `<family>.joblib` and metadata JSON for both finalists | Complete train-fitted pipeline, original/effective model settings, frozen feature config, class order, transformed names, versions and provenance references. |

Preserve bundle schema 2.0 for engineered finalists unless a structural change requires a new version. Model version/run identity must distinguish this result instead of reusing one generic constant. Mark bundles `development_only`; do not update a production/latest pointer. Retaining both finalists lets SPEC-07 evaluate without repeating the search.

Use existing project-scoped privacy-safe hashing for fold membership/order and prediction audits. Public JSON/CSV/Markdown contain no raw client/content keys, row predictions, raw validation label vectors or per-row probabilities. Review categorical diagnostics and exception strings for accidental identity leakage.

Bind provenance to source SHA-256, split manifest/assignment hashes, ordered train/validation identity, original feature-study input and selected-config hashes, baseline manifest/config, effective tuning/model settings, code revision/dirty/source-tree hash and Python/library/runtime versions. Include all training/tuning modules and entry scripts in code fingerprints. Recheck source/split/feature/baseline bytes against preflight hashes before publication.

Hash report payloads and model bundles after writing. Verify allowlisted filenames and resolved paths including symlinks against the separate approved roots. Hash-check trusted local joblib bytes before deserialization; never deserialize arbitrary untrusted files. Publish the final manifest only after both roots are complete and readback succeeds. Interrupted multi-root writes may leave staging files but never valid completion evidence.

Semantic reproducibility requires equal fold identities, trial schedule/statuses, aggregate metrics, selected parameters, validation metrics and prediction audit hashes in the same recorded environment. Normalize run IDs, timestamps, physical paths, timings and hashes of timing-bearing files. Joblib bytes need not match across runs; verify each run's own hashes and equivalent reloaded behavior. Declare numerical tolerances explicitly and never use them to ignore changed winners, predictions or eligibility flags.

## 11. Delivery phases and exit criteria

### Phase 0 - Adopt the protocol

Replace SPEC-06 placeholders with requirements/tests and the search/selection rules. Adopt the 15-trial/47-fit budget, recall aggregation, tie policy and development-only boundary. Document previous validation/test exposure and unresolved cutoff evidence.

**Exit:** No normative placeholders remain; SPEC-07/08 boundaries and scientific versus execution status are explicit.

### Phase 1 - Configuration and preflight

Implement grid/model validation, trial IDs/budget expansion and read-only CLI preflight. Verify frozen manifests using the original config and reject unsafe outputs before any fit.

**Exit:** Dry run reports aggregate counts and 47 planned fits, with no model fit or artifact publication. Invalid inputs fail actionably.

### Phase 2 - Grouped fold execution

Implement shared fold identities, fresh pipelines with separate overrides, fixed-label scoring and timing/convergence diagnostics.

**Exit:** Synthetic spies prove every fitted stage sees only fit-fold rows, all trials share folds, and identifiers/outcomes never reach model features.

### Phase 3 - Selection and confirmation

Implement deterministic CV choices/fallbacks, train-only finalist refits, validation confirmation and SPEC-05 comparison. Preserve legacy fixed-training behavior.

**Exit:** Choices freeze before validation; failed flags remain explicit; legacy consumers still work.

### Phase 4 - Publication and readback

Implement isolated staging, full metadata/hashes, manifest-last completion and readback through the existing prediction contract.

**Exit:** Both bundles reproduce predictions; tampered/missing/escaped artifacts fail; interrupted runs cannot appear complete.

### Phase 5 - Real-data verification and handoff

Run focused/full tests and lint, then two isolated approved searches. Verify each run and compare semantic evidence. Record actual scores/warnings/runtime, exact commands and protected-input hashes. Update SPEC-06/readme/plan only with linked evidence.

**Exit:** Verified finalist artifacts and documented results are ready for SPEC-07, without requiring target achievement or deployment authorization.

## 12. Test matrix

| Test ID | Scenario | Expected result / location |
| --- | --- | --- |
| TRAIN-T-001 | Approved grid expansion | 3 logistic + 12 forest trials, 45 fold + 2 refit calls, stable IDs; `tests/unit/test_tuning_config.py`. |
| TRAIN-T-002 | Bad names/ranges/duplicates/booleans/nonfinite values/budget/version | Fail before fitting; no silently ignored parameters. |
| TRAIN-T-003 | Missing/stale source or outer split | Actionable failure; no regenerated assignments. |
| TRAIN-T-004 | Stale/tampered feature/baseline evidence | Reject before fitting, including row order/count/labels/metric mismatch. |
| TRAIN-T-005 | Original feature inputs intact; trial C/depth differs | Frozen features remain valid; override hash recorded; changed original inputs still fail. |
| TRAIN-T-006 | Canonical reorder with valid renewed source provenance | Deterministic folds; each train row scored once; clients disjoint. |
| TRAIN-T-007 | Too few clients/null groups/missing classes/duplicate identities | Preflight failure without new seed search or relaxed coverage. |
| TRAIN-T-008 | Spy on engineering/imputer/encoder/scaler/estimator fit | Only current fit-fold indices; fresh state; `tests/integration/test_tuning_pipeline.py`. |
| TRAIN-T-009 | Held-out-only extremes/categories and forbidden columns | No learned leakage; unknown categories handled; identifiers/outcomes rejected as features. |
| TRAIN-T-010 | Known fixed-label predictions | Exact metrics/support/matrix; one prediction batch per fold. |
| TRAIN-T-011 | Unequal fold sizes and known scores | Declared unweighted mean, ddof=0/min/max; pooled scores cannot alter ranks. |
| TRAIN-T-012 | Recall/tolerance boundaries and changed iteration order | Deterministic eligible/complexity/ID choices; `tests/unit/test_tuning.py`. |
| TRAIN-T-013 | No recall pass, or no valid trial for a family | Explicit diagnostic fallback for valid weak scores; incomplete status if a finalist is impossible. |
| TRAIN-T-014 | Validation/test access instrumentation | Tuner receives no validation/test; choices precede validation predict; no test scoring/refit. |
| TRAIN-T-015 | Finalist refit and reload | Both fit full train only; feature/class/config/predictions round-trip. |
| TRAIN-T-016 | One baseline beaten or margin/recall/target missed/equal | Independent SPEC-05 flags; validation cannot restart search. |
| TRAIN-T-017 | Timers and model parameter inspection | Exact fit-count budget, serial resources, separated stage timings. |
| TRAIN-T-018 | Convergence warning/fit exception/interruption | Diagnostics retained; no partial-average winner or false completion. |
| TRAIN-T-019 | Successful publication | Complete hashes/provenance/development metadata; `tests/contract/test_training_contract.py`. |
| TRAIN-T-020 | Missing/tampered model/path traversal/symlink | Fail before deserialization/handoff. |
| TRAIN-T-021 | Private synthetic identities and diagnostic strings | Only aggregates and safe hashes in public artifacts. |
| TRAIN-T-022 | Two synthetic and two approved real runs | Same folds/trials/choices/predictions/metrics; volatile timing/serialization normalized explicitly. |
| TRAIN-T-023 | Protected snapshots around success/failure | Source/split/feature/baseline/old model/report bytes unchanged. |
| TRAIN-T-024 | Legacy CLI/API and tuning dry run | Fixed mode compatible; dry run fits nothing and writes no results. |
| TRAIN-T-025 | Handoff review | Both finalists and flags supplied; prior exposure/CV optimism/cutoff/runtime limits explicit. |

Use small synthetic multi-client fixtures, controlled estimators and fit spies for routine checks. Never use the real test partition as a fixture. Real-data verification proves reproducibility/scale; it does not replace leakage or fault-injection tests.

## 13. Planned commands

The tuning options, new files and tests below are proposed deliverables and must not be run before implementation.

```powershell
# Read-only preflight; tuning requires both frozen dependencies.
.\.venv\Scripts\python.exe scripts/train_model.py --tuning-config configs/tuning.yaml --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --dry-run

# First isolated run; existing data/training/preprocessing/baseline configs are defaults.
.\.venv\Scripts\python.exe scripts/train_model.py --tuning-config configs/tuning.yaml --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --run-id spec06-reference

# Same protocol, separate output destination.
.\.venv\Scripts\python.exe scripts/train_model.py --tuning-config configs/tuning.yaml --feature-manifest reports/features/feature_manifest.json --baseline-manifest reports/baselines/baseline_manifest.json --run-id spec06-reproduction

.\.venv\Scripts\python.exe -m pytest tests/unit/test_tuning_config.py tests/unit/test_tuning.py tests/integration/test_tuning_pipeline.py tests/contract/test_training_contract.py tests/integration/test_training_pipeline.py tests/integration/test_baseline_pipeline.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

The verification script should orchestrate the two runs or verify those completed above, not inadvertently perform four full searches. Record commands, exit codes, environment, fit counts and normalized comparisons. Keep scratch output outside the active pytest base directory. This plan claims no new scores or completed checks.

## 14. Decisions and implementation risks

| Decision/risk | Proposed resolution |
| --- | --- |
| No approved SPEC-06 search space | Adopt the explicit grid at implementation; no unbounded search. |
| Feature fingerprints include original model settings | Verify originals first, then apply separate hashed tuning overrides. |
| Shared default features differ from frozen choices | Require estimator-specific feature manifest. |
| CV reuses feature-development evidence | Disclose non-nested selection optimism; no independent-generalization claim. |
| Validation used in prior confirmation | Freeze search choices before current validation; no adaptive follow-up grid. |
| Mean versus worst-fold recall | Mean >=0.50 for CV eligibility; show worst-fold recall and separate validation gate. |
| Legacy selection uses validation | Preserve fixed mode; tuning reports a CV preference and hands broader selection to SPEC-07. |
| Logistic convergence failures | Record/exclude; no silent iteration increase or partial-score ranking. |
| Forest compute cost | Predetermined fit budget and serial resources; report observed runtime. |
| Existing artifacts overwritten | Isolated roots and immutable dependency snapshots. |
| Serialization/timing vary | Per-run hash verification plus semantic/reload comparison. |
| Weak candidates | Complete evidence with failing flags; never lower target/margin or promote fallback. |
| Historical cutoff uncertainty | Carry limitation into reports/bundles; tuning cannot resolve it. |

## 15. Acceptance checklist and definition of done

- [x] SPEC-06 contains adopted requirements/tests/defaults without normative placeholders.
- [x] Frozen manifests and original input contracts are verified before fitting.
- [x] Exact 15-trial/47-fit schedule is validated and recorded.
- [x] All trials share training-only client-grouped folds with required class coverage.
- [x] Complete pipelines fit freshly inside folds without identity/outcome leakage.
- [x] Fixed-label metrics, convergence diagnostics and deterministic ranking are tested.
- [x] CV preference and finalist parameters freeze before validation scoring.
- [x] Both finalists remain train-fitted; no test scoring or train-plus-validation refit occurs.
- [x] Improvement/recall/target flags remain independent of development preference.
- [x] Legacy fixed-training callers remain compatible.
- [x] Isolated artifacts have full provenance, safe paths and verified hashes.
- [x] Reloaded finalists reproduce predictions and feature/class contracts.
- [x] Public outputs contain no raw sensitive identities or row-level prediction matrices.
- [x] Protected source/split/feature/baseline/old model/report bytes remain unchanged.
- [x] Focused/full tests, Ruff and two-run real-data checks pass with exact evidence.
- [x] SPEC-07/08 handoff includes both finalists, failing flags and scientific/runtime limitations.

Completion means the approved search is reproducible and its evidence can be safely consumed downstream. Target achievement, final holdout policy and promotion remain separate decisions. This planning request creates only this implementation plan.


## 16. Implementation record (2026-09-08)

The subsequent request to implement this plan adopted protocol 1.0. This record supersedes the
planning-only authorization language above. Configuration, the train-only grouped search,
CV selection, isolated publication, trusted readback, CLI preflight and verification tooling
are implemented. The optional throughput diagnostic remains explicitly out of the measured results.

The frozen feature-study contract remains intact: original inputs are verified before per-trial
parameters are applied. Existing fixed training retains its legacy selection behavior.
No source/split/feature/baseline artifacts are regenerated by tuning.

Verification completed: **51 focused tests, 150 full-suite tests and Ruff passed**.
Both real runs completed 47 fits each, with identical semantic evidence, selected parameters,
validation metrics and prediction fingerprints. All 15 trials in each run were valid. Both
bundles passed hash/readback checks; protected prerequisite and existing model/report bytes
were unchanged. A public JSON identity audit passed for both run directories.

Random forest is the frozen CV preference: depth=None, max_features=0.5, min_samples_leaf=5,
200 trees. Validation macro F1 is **0.430224**; logistic regression with C=0.1 scores **0.389920**.
Both pass material baseline improvement and down recall, but both fail the unchanged 0.45 target.
No test scoring, train-plus-validation refit, adaptive search or promotion occurred.

Evidence:

- [Reference report](../../reports/training/spec06-reference/training_report.md)
- [Reference completion manifest](../../reports/training/spec06-reference/training_manifest.json)
- [Reproduction manifest](../../reports/training/spec06-reproduction/training_manifest.json)
- [Exact commands and verification](../../reports/training/spec06-verification.json)

SPEC-07 can load both finalists with `load_training_run` using the matching report/model run
directories. Finalist bundles are local generated files excluded from Git; manifests/metadata
and aggregate evidence remain reviewable. The optional throughput diagnostic was not run and
no production SLA is claimed.
