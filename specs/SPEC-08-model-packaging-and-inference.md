# SPEC-08 - Model Packaging and Inference

**Status:** Complete for research packaging; verified 2026-09-10
**Owner:** Babar Ali Khan
**Adopted:** 2026-09-10
**Outcome:** Reproducible research packages and strict local inference, preserving the frozen SPEC-07 rejection decision.

## Scope

Package either or both frozen SPEC-06 finalists using verified SPEC-07 evidence. Support explicit research purpose only, packaging/package schema 1.0 and inference contract 2.0. Preserve the legacy bundle API and six-field record serialization. The adopted detailed contract is sections 4-12 of the [implementation plan](../plan/SPEC-08-model-packaging-and-inference/implementation-plan.md).

No training, tuning, feature selection, final-test scoring, model promotion, remote loading, API deployment or automatic content changes are authorized by this phase. Both current finalists remain below the unchanged macro-F1 target; recommended_model is null and production_ready is false.

## Requirements

| Requirement | Behavior | Tests |
| --- | --- | --- |
| SPEC-08-REQ-001 | Validate packaging/inference versions, purpose, resource limits and safe destinations before loading/scoring | PKG-T-001/002 |
| SPEC-08-REQ-002 | Verify SPEC-07 completion, decision, references and exact SPEC-06 model bytes | PKG-T-003/004 |
| SPEC-08-REQ-003 | Preserve research-only/null-recommendation semantics and refuse promotion aliases | PKG-T-005 |
| SPEC-08-REQ-004 | Publish immutable self-contained packages with complete provenance and manifests | PKG-T-006/007 |
| SPEC-08-REQ-005 | Verify all payloads, schema compatibility and environment before deserialization | PKG-T-008/009 |
| SPEC-08-REQ-006 | Enforce typed, bounded, cutoff-safe input contracts and ID preservation | PKG-T-010/011/012 |
| SPEC-08-REQ-007 | Return valid ordered labels and explicitly optional uncalibrated probabilities | PKG-T-013/014 |
| SPEC-08-REQ-008 | Preserve deterministic row order and full/chunked batch equivalence without fitting | PKG-T-015/016 |
| SPEC-08-REQ-009 | Keep request-level identifiers private and publish no source-row examples | PKG-T-017 |
| SPEC-08-REQ-010 | Support explicit local CLI/Python interfaces and existing legacy consumers | PKG-T-018/019 |
| SPEC-08-REQ-011 | Verify source/package prediction parity and reproducible package semantics | PKG-T-020/021 |
| SPEC-08-REQ-012 | Record truthful resource measurements and immutable upstream evidence | PKG-T-022/023 |
| SPEC-08-REQ-013 | Hand off a versioned inference contract and unresolved readiness boundaries to SPEC-09 | PKG-T-024 |

## Contract decisions

- Copy original model bytes; verify exact payload inventory, hashes, decision/model references, schemas and runtime before trusted deserialization. A caller may pin the manifest digest. Hashes do not authenticate attacker-controlled packages.
- Publish completion manifests last; reject occupied destinations and protected paths. Interrupted family builds have no complete run marker.
- Require content_id and exactly the approved raw features. Preserve unique string IDs, including leading zeros and literal NA. Reject duplicate columns, extras, invalid types, nonfinite numbers and negative constrained values before any scoring. Missing cells use fitted imputation; unknown categories use the frozen encoder.
- Python numeric strings are rejected. CSV is UTF-8 with the configured delimiter and explicit missing tokens applied only to feature cells. Response IDs remain unchanged.
- Maximum request size is 30,000 rows; default chunks are 5,000 and numerical threads are one. Configured limits may be stricter.
- Default label-only mode never calls predict_proba. Explicit probability mode validates actual estimator class mapping, shape, finite [0,1] scores and row sums with absolute tolerance 1e-9. Scores are uncalibrated and non-causal. Stable review ranking requires probability mode.
- JSON is the supported private batch output, with the full versioned envelope. No CSV output or metadata sidecar is implemented. Validate the entire bounded request before chunking and atomically publish only complete responses, without overwrite.
- Runtime support requires exact CPython version, OS, architecture, dependency versions and project source digest declared in environment.json. Build the project wheel offline. Isolated verification installs the wheel locally and reuses existing dependency files with site/editable hooks disabled and upstream access denied. It does not certify a clean dependency installation or other platforms.
- Two builds reuse one verified wheel. Compare semantic package digests and source model bytes. Real parity uses verified development validation only; capacity measurements use invented synthetic rows. Record observed timings without an SLA claim.

## Tests

| Test ID | Scenario | Expected outcome / proposed location |
| --- | --- | --- |
| PKG-T-001 | Valid fixed packaging/inference configuration | Deterministic resolution; `tests/unit/test_packaging_config.py` |
| PKG-T-002 | Unknown keys, unsafe names, conflicting roots, occupied output, invalid row/chunk/tolerance values | Fail before loading/scoring/writing |
| PKG-T-003 | Completed SPEC-07 decision bound to matching SPEC-06 finalist | Valid family/model/selection/config references; `tests/integration/test_packaging_pipeline.py` |
| PKG-T-004 | Tampered/missing/stale evaluation or training evidence; file hash confused with semantic hash | Reject before trusted model loading as appropriate |
| PKG-T-005 | Null recommendation, production purpose, recommended alias, explicit research family | Research succeeds with rejection preserved; promotion requests fail |
| PKG-T-006 | Successful two-family copy-only publication | Original model hashes unchanged; exact required payloads and complete manifests |
| PKG-T-007 | Interrupted publication, one-family partial build, escaped symlink/junction | No false complete-run marker; existing complete packages preserved |
| PKG-T-008 | Missing/tampered/extra/escaped payload; forged metadata relationship | Reject before unpickling or immediately on embedded semantic mismatch; `tests/contract/test_package_contract.py` |
| PKG-T-009 | Unsupported runtime version; isolated supported runtime with upstream paths inaccessible | Fail before unpickling for mismatch; supported standalone inference succeeds |
| PKG-T-010 | Missing/duplicate/extra/outcome columns, empty/oversized frame | Whole-request rejection; `tests/unit/test_inference_validation.py` |
| PKG-T-011 | Null/duplicate/non-string/whitespace IDs, leading zeros, literal NA, duplicate CSV header | Invalid IDs fail; valid identifiers preserved byte-for-byte |
| PKG-T-012 | Numeric booleans/text/infinity/negative constrained values; missing cells; unknown categories | Strict invalid-value rejection, frozen imputation/unknown-category behavior |
| PKG-T-013 | Labels/output shape/class set invalid; model classes in another order | Reject malformed outputs; correct canonical response mapping |
| PKG-T-014 | Label-only, explicit probabilities, missing capability, bad shape/sum/nonfinite values | Correct call counts/null fields; valid down mapping or explicit failure |
| PKG-T-015 | Reordered dataframe columns/index, non-divisible chunks, ties in queue | Positional order and full/chunk parity; stable probability ranking |
| PKG-T-016 | Fit/fit_transform/partial_fit/builders/probability calls instrumented | No estimator or feature-transformer fits; no probability calls in label mode |
| PKG-T-017 | Seeded private IDs/categories; CLI/report/error output inspection | Authorized response echoes IDs; package/report/log artifacts leak no source rows |
| PKG-T-018 | CLI valid input, invalid later chunk, existing destination and output/input alias | Atomic private output or no final output; `tests/integration/test_packaged_inference.py` |
| PKG-T-019 | Existing legacy bundles, record keys and batch callable | Existing contract tests remain green; v2 requires explicit interface |
| PKG-T-020 | Source versus packaged validation predictions/probabilities | Exact labels/audit fingerprint; declared probability parity; no test scoring |
| PKG-T-021 | Two synthetic and two real research packaging runs | Identical semantic package content and independently valid payload manifests |
| PKG-T-022 | Synthetic 30,000-row CPU workload and row-limit boundary | Completes in bounded chunks with measured timings; no invented deadline/pass claim |
| PKG-T-023 | Protected snapshots across dry run/success/failure/verification | Existing source, splits, features, baselines, models and evaluation reports unchanged |
| PKG-T-024 | Package card, schema examples and SPEC-09 handoff | Research status, null recommendation, probabilities and historical exposure stated accurately |

The packaging integration test is tests/integration/test_packaging_pipeline.py. It covers synthetic two-build reproduction, zero-fit/zero-predict dry runs, byte identity, CLI parity, private global validation, forged metadata, environment mismatch, tampering, occupied destinations and interrupted publication. Existing contract/unit tests cover request/output boundaries and legacy compatibility. scripts/verify_packaging.py owns the real parity, isolated runtime, privacy, immutability and synthetic workload evidence.

## Acceptance and handoff

Completion requires the implementation plan acceptance checklist, passing focused/full tests and Ruff, and actual verification evidence. SPEC-09 must choose a concrete research package path, enforce its input/output contract and preserve unresolved cutoff, historical holdout exposure, model eligibility and deployment limitations. See the [handoff](../reports/packaging/SPEC-09-handoff.md) for concrete artifacts and results.

## Remaining product decisions

Production acceptance criteria, independent final-test policy, cutoff review, numeric runtime SLA, deployment, application access controls and retention belong to subsequent specifications. They do not block research packaging and are not resolved by this implementation.
