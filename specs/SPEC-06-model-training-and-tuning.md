# SPEC-06 - Model Training and Tuning

**Status:** Implemented and verified for development on 2026-09-08.
**Owner:** Babar Ali Khan
**Protocol adopted:** 2026-09-08 under the request to implement the plan.

## Objective and scope

Tune logistic regression and random forest reproducibly using the existing training partition.
Verify SPEC-03 splits, SPEC-04 frozen estimator-specific features and SPEC-05 frozen baselines
before fitting. Search receives training rows only. Retain both train-fitted finalists for SPEC-07.
Do not change outer splits, feature choices, targets, baselines or existing model/report artifacts.
No test scoring, train-plus-validation refit, deployment or promotion belongs to this phase.

## Adopted protocol 1.0

`configs/tuning.yaml` owns a fixed 15-configuration search over three shared shuffled,
seed-42 StratifiedGroupKFold folds. Validate client/row disjointness, complete class coverage
and one scoring-fold assignment per training row. Canonically order development rows first.
Every trial fits a fresh complete pipeline inside its fit fold; no globally fitted preprocessing.

Logistic regression searches C=[0.1,1.0,10.0], with balanced weights, lbfgs, max_iter=1000
and seed 42. Random forest searches min_samples_leaf=[1,2,5], max_features=[sqrt,0.5],
max_depth=[null,20], with 200 trees, balanced_subsample weights and seed 42.
Run serially with estimator/numerical threads limited to one. The schedule contains 45 fold
fits and two train-only finalist refits, for 47 estimator fits per complete run.

The original feature-study contract includes original candidate settings. Validate those unchanged
inputs first and apply trial overrides separately on fresh pipelines. Never bypass this check.
Each family uses its exact frozen feature config; the shared default is not a substitute.

Score complete fold batches with SPEC-05 metric contract 1.0. Rank valid complete/converged
trials by unweighted mean fold macro F1, preferring mean down recall >=0.50. Within 0.001
of the maximum, prefer smaller C for logistic regression; for forest prefer bounded/smaller
depth, larger leaf size, then sqrt before 0.5; final tie is stable trial ID. Across families,
prefer logistic regression on a tie. Record worst-fold recall and population SD separately.
No eligible recall trial produces an explicit diagnostic fallback. No valid trial for either
family makes the run incomplete. Never rank partial-fold averages or silently change iterations.

Freeze family choices and a CV development preference before validation scoring. Refit each
finalist on all train rows only. Evaluate each on canonical validation once; reload predictions
may repeat only to verify serialization. Validation cannot change the grid or frozen preference.
Use SPEC-05 comparison helpers for strict superiority, +0.01 material improvement over majority,
canonical stratified and stratified mean, down recall >=0.50, and macro F1 >=0.45 independently.
Weak valid scores are completed scientific evidence; they are not execution failures or promotion.

## Requirements

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

## Test contract

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

## Artifacts and compatibility

Use isolated reports/training/<run_id> and artifacts/models/training/<run_id> directories.
Reject occupied/unsafe destinations. Publish aggregate config, fold, trial, selection, validation,
timing and report payloads, plus both complete schema-2.0 engineered finalist bundles.
Bundles carry unique model versions and development_only metadata. Preserve legacy fixed-training
CLI/API behavior. SPEC-07 uses load_training_run(report_dir, model_dir) for both finalists.

The completion manifest binds source/split/order/feature/baseline/config/code/environment hashes
to every report/model payload. Verify safe local paths and all hashes before trusted joblib loads.
Reload checks must pass before completion. Failed/interrupted runs retain diagnostics without a
completion manifest. Public artifacts contain no raw identities or row-level prediction vectors.

## Verification and limitations

Run python scripts/verify_training.py for focused/full tests, Ruff and two isolated real searches.
It records commands, 94 total real estimator fits, semantic results and protected-input snapshots
in reports/training/spec06-verification.json, with references in each run directory.
Compare metrics, fold/selection identity and prediction hashes; exclude timestamps, paths, timings
and joblib bytes while verifying each run's own integrity. Do not overwrite earlier evidence.

Features were selected using broader training evidence; CV is not nested evaluation of the full
selection procedure. Validation informed prior feature confirmation and historical test results
exist. Fold dispersion is not a confidence interval; these are development results, not independent
generalization estimates. Historical cutoff evidence remains unresolved. The optional 30,000-row
throughput diagnostic is deferred; fit/predict timings do not establish a production SLA.

Completion requires traceable tests, full regression/lint, two-run semantic/reload verification,
private outputs and protected-artifact immutability. Target achievement and promotion remain separate.


## Verified outcomes (2026-09-08)

- 51 focused checks passed; the full suite passed 150 tests; Ruff passed.
- Both approved real runs completed 47 fits each (94 total). All 15 trials per run were valid.
- Fold/trial evidence, selected parameters, validation metrics and prediction fingerprints matched exactly.
- Both finalist bundles passed hash verification and reload checks.
- Protected source/split/feature/baseline/existing model/report bytes remained unchanged.
- Public JSON identity audit passed for both runs. No test scoring or train-plus-validation refit ran.

| Finalist | Chosen settings | Validation macro F1 | Baseline improvement / recall | Project target |
| --- | --- | ---: | --- | --- |
| Logistic regression | C=0.1 | 0.389920 | Passed / passed | Failed |
| Random forest | depth=None, max_features=0.5, min_samples_leaf=5; 200 trees | 0.430224 | Passed / passed | Failed |

Random forest is the frozen CV development preference. Neither finalist reaches macro F1 0.45.
These failed target flags are retained for SPEC-07; no automatic retuning or promotion occurred.

Evidence: [reference report](../reports/training/spec06-reference/training_report.md),
[reference manifest](../reports/training/spec06-reference/training_manifest.json),
[reproduction manifest](../reports/training/spec06-reproduction/training_manifest.json),
[exact commands and verification](../reports/training/spec06-verification.json).
