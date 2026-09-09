# SPEC-07 - Evaluation and Error Analysis

**Status:** Implemented and verified for development on 2026-09-09.
**Owner:** Babar Ali Khan
**Protocol adopted:** 2026-09-09 under the request to implement the plan.

## Objective and scope

Audit both frozen SPEC-06 finalists on the same canonical development validation rows.
Verify source/split/config/feature/baseline/model/selection provenance, then require exact
prediction fingerprint, metric and comparison replay before publishing new diagnostics.
No training, tuning, calibration, feature selection, baseline generation, external dataset,
test scoring, deployment or production model pointer is permitted in this phase.
`load_development` validates full-source integrity and returns development partitions only.
Only approved validation features, truth and context reach analysis.

Validation informed earlier feature confirmation and SPEC-06 confirmation. Historical test
evaluation exists. These diagnostics reuse development evidence and are not independent
generalization estimates. Historical metadata cutoff evidence, a future final-test policy,
and production runtime certification remain unresolved review blockers.

## Adopted protocol 1.0

`configs/evaluation.yaml` owns the closed evaluation/artifact/metric version 1.0 configuration.
Both logistic regression and random forest are required. Predict once per complete canonical
validation batch with numerical threads limited to one and unchanged serial model settings.
Do not invoke fitting or probabilities. Dry run verifies prerequisites and context without
predicting, fitting or writing. Reject stale environments, identities and semantic references.

Metrics retain SPEC-05's ordered labels and zero-valued unsupported per-class metric records.
Diagnostic rates separately return null with numerator, denominator and reason when undefined.
Publish raw/row-normalized confusion, ranked off-diagonal counts, TP/FP/FN/TN and down workload,
misses and false-alert source classes. Paired correctness and true-down counts partition their
populations; the truth x logistic prediction x forest prediction table reconstructs both matrices.

The only slice dimensions are content_type, main_intent, age_tier, freshness_tier,
previous_impressions_bucket and numeric_missingness_bucket. Previous impressions bins are
missing, zero, (0,100], (100,1000], >1000. Numeric missingness counts raw configured numeric
columns before imputation and uses 0, 1-2, >=3. No outcome fields or cross-products are allowed.
Groups use validation frequencies, then stable category value order; retain up to 12 sufficiently
large categories and pool remaining nonmissing values. Missing and pooled sentinels are typed.
Publish anonymous frequency-ordered category aliases, never raw categorical names or client IDs.
Aliases are local to a dimension/population; they are not universal category identities.

Defaults: publish groups with at least 100 rows; require 20 truth examples per class for
class interpretation and a macro-F1 gap alert; require 30 true down examples for down-recall
alerts and 30 predicted down examples for precision interpretation. Published fixed-label macro
F1 includes all five labels; balanced accuracy averages observed truth classes. Unsupported
down interpretations are null. A supported macro-F1 gap >=0.05 or down recall <0.50 flags review.
Every dimension accounts for published/suppressed/pooled/missing rows. Overlapping dimensions
must not have their error counts added. Support thresholds are not a formal privacy guarantee.

Leave one whole client out at a time using existing predictions; no refitting or reprediction.
Publish aggregate row/score/recall/gap ranges, support-valid scenario counts, ordering reversals
and a source-scoped assignment hash, without per-client scenario tables. Macro summaries and
ordering flags require every truth class to meet the class support threshold; down summaries
require down support. No p-values, confidence intervals, fairness or causal claims.

## Decision and SPEC-08 handoff

Inherit macro F1 >=0.45 and down recall >=0.50 from training configuration and minimum
improvement 0.01 from baseline configuration; never introduce conflicting copies.
Eligibility requires all four SPEC-05 comparison flags, recomputed and matched to SPEC-06.
When neither finalist is eligible, recommended_model is null and recommendation_status is
no_candidate_meets_metric_requirements. Preserve the frozen CV development reference separately.
Among eligible models within 0.001 of the maximum validation macro F1, prefer the eligible
CV reference, then logistic regression, then stable family name. Review flags cannot select
an ineligible model. Unresolved cutoff/holdout/runtime evidence keeps an eligible recommendation
pending review. production_ready is always false. Packaging acceptance is owned by SPEC-08.

Handoff decision.json includes model/report/selection/feature/baseline hashes, failing eligibility
reasons, review flags and limitations. A null recommendation hands off rejection evidence and
research references without a production model pointer. Future experiment hypotheses do not
authorize changing the frozen models, splits, slices or thresholds.

## Requirements and tests

| Requirement | Contract | Verification |
| --- | --- | --- |
| SPEC-07-REQ-001 | Closed configuration; validation-only safe isolated output | test_evaluation_config.py; test_evaluation_contract.py |
| SPEC-07-REQ-002 | Fresh source/split identity and frozen provenance compatibility | test_evaluation_pipeline.py; read-only real preflight |
| SPEC-07-REQ-003 | Zero fits and exactly one canonical validation prediction per finalist | audit_inference; synthetic and real verify_runs |
| SPEC-07-REQ-004 | Exact stored prediction/metric/comparison replay before publication | replay and drift rejection integration tests |
| SPEC-07-REQ-005 | Fixed-label class/down arithmetic and undefined denominators | test_error_analysis.py |
| SPEC-07-REQ-006 | Paired counts with both confusion marginals | test_error_arithmetic_and_joint_marginals |
| SPEC-07-REQ-007 | Shared deterministic support-aware slices and complete coverage | membership, collision, privacy and support boundary tests |
| SPEC-07-REQ-008 | Aggregate supported client sensitivity with no refitting | client sensitivity fixture and inference audit |
| SPEC-07-REQ-009 | Deterministic eligibility/null recommendation separate from review/readiness | parametrized decision tests |
| SPEC-07-REQ-010 | Aggregate-only private outputs and plots with undefined markers | privacy scan and plot data/manifest checks |
| SPEC-07-REQ-011 | Complete safe manifest-last payload publication and provenance | evaluation artifact contract and tamper tests |
| SPEC-07-REQ-012 | Semantic reproduction and protected input immutability | two synthetic runs; two frozen real replay runs |
| SPEC-07-REQ-013 | Compatible legacy display and explicit research/rejection handoff | legacy CLI test; report and decision review |

## Artifacts and acceptance

Write exclusively into a fresh reports/evaluation/<run_id>. Required payloads are enumerated in
evaluation_artifacts.PAYLOADS, including class/subgroup CSV, error/confusion/paired/coverage/
sensitivity/decision JSON, plot source data, five PNGs, report, configuration and timings.
The completion manifest binds all payloads to input/config/model/code/environment provenance
and is written last after protected input hashes are rechecked. Reader rejects missing, altered,
escaped or incomplete payloads. Existing runs are never overwritten.
verification.json is a subsequent attestation referencing both completion manifest hashes,
avoiding a circular hash dependency. Semantic reproduction excludes run paths/IDs and timings;
each run independently verifies PNG hashes, without requiring renderer-independent PNG bytes.

Run `python scripts/verify_evaluation.py` for full tests, Ruff, a dry run and two instrumented
real replay runs. The real runs perform four predictions and zero fits in total; synthetic
fixture setup may train only its tiny synthetic models. Subsequent runs require fresh run IDs.
Completion requires passing checks, linked verification evidence, immutable prerequisites,
and an honest recommendation or rejection. Independent holdout success and deployment are
separate outcomes.

## Recorded verification and handoff

`reports/evaluation/spec07-reference/verification.json` records the exact verifier command,
198 passing tests, Ruff success, four canonical validation prediction calls, zero estimator/
feature-transformer fits and zero probability calls. Metric-local sklearn LabelEncoder
bookkeeping is explicitly excluded from the fit audit. Both real runs reproduce the saved
SPEC-06 prediction fingerprints, metrics and comparison flags exactly; semantic artifacts
match and protected source/split/feature/baseline/training/model hashes remain unchanged.

- [Evaluation report](../reports/evaluation/spec07-reference/evaluation_report.md)
- [Frozen decision and SPEC-08 handoff](../reports/evaluation/spec07-reference/decision.json)
- [Reference verification](../reports/evaluation/spec07-reference/verification.json)
- [Reproduction verification](../reports/evaluation/spec07-reproduction/verification.json)

Logistic regression macro F1 is 0.389920; random forest macro F1 is 0.430224.
Both fail the unchanged 0.45 project target, so recommended_model is null. Random forest
remains the CV development reference. There are 1,518 paired disagreements, five supported
slice alerts, and a macro-F1 ordering reversal under client omission. These are descriptive
review findings. No model is promoted and production_ready remains false.
