# Implementation Plan - Model Packaging and Inference

**Source:** `specs/SPEC-08-model-packaging-and-inference.md`
**Status:** Proposed; this request authorizes this planning document only
**Prepared:** 2026-09-09
**Owner:** Babar Ali Khan
**Outcome:** Package the frozen development finalists reproducibly and define validated, deterministic local inference without implying model acceptance or production readiness.

## 1. Objective and specification status

SPEC-08 currently contains placeholder scope, requirements, tests and decisions. This plan proposes concrete defaults based on the implemented repository and the SPEC-07 handoff. These defaults become normative when adopted during a separately authorized implementation. Creating this plan does not modify specifications, package models, score data, change inference code, or publish artifacts.

Deliver two related capabilities:

1. A safe, versioned package format containing an unchanged fitted pipeline, its inference contract, provenance, dependency requirements and research limitations.
2. A local Python/batch inference interface that validates raw inputs, preserves row identity and order, returns validated predictions, and refuses unsupported packages or malformed requests.

Scientific eligibility and engineering packaging success are separate outcomes. The current project can complete research packaging mechanics while having no model eligible for production use. SPEC-09 can later build an application on this contract; API endpoints and deployment are outside this phase.

## 2. Current state and authoritative evidence

Paths under `src/machine_learning_project/` below are package-relative.

| Area | Existing behavior | Planned treatment |
| --- | --- | --- |
| SPEC-08 | Draft with `TBD` placeholders | Adopt concrete requirements and tests during implementation, not this planning task. |
| `models/training_artifacts.py::load_training_run` | Verifies both finalists' report/model hashes before trusted joblib loading; checks metadata and evaluation identity | Reuse for packaging preflight; additionally bind the chosen bytes to the verified SPEC-07 decision. |
| `models/evaluation_artifacts.py::load_evaluation` | Verifies completed evaluation payload inventory, hashes, identity and semantic references | Use as the entry point for consuming the decision; do not read an unverified standalone decision JSON. |
| `inference/predictor.py::Predictor.load` | Loads legacy bundle versions 1.0/2.0 directly through joblib; version 2.0 checks feature configuration | Preserve legacy compatibility and introduce an explicitly verified package loader. Do not describe the legacy method as a secure package verification boundary. |
| `Predictor.predict` | Selects configured features; calls `predict` and, when available, `predict_proba`; echoes content ID and model version | Add a separate versioned interface with explicit probability mode and response validation. Reuse fitted transforms without fitting. |
| `inference/validation.py` | Requires feature columns and unique non-null content IDs; rejects empty frames | Expand the new contract to validate types, finite values, bounds, duplicate columns, extra columns, resource limits and ID normalization policy. |
| `features/selection.py` | Defines ordered approved features and converts numeric strings; excludes configured ID/drop/target fields | Reuse approved feature ordering; define stricter request parsing before calling it. Do not mutate shared training behavior implicitly. |
| `inference/schemas.py::PredictionRecord` | Dataclass with six fields and a human-review warning; no runtime probability validation | Preserve its legacy serialization; define a versioned envelope and validated records for the new interface. |
| `pipelines/inference_pipeline.py` | Reads a CSV using pandas defaults, loads a raw bundle and returns a dataframe | Introduce package-aware batch orchestration, identifier-safe CSV parsing and atomic private output. |
| `app/main.py` | Health endpoint only | Leave API implementation to SPEC-09. |
| Tests | A six-field prediction schema test, feature-bundle checks and existing round-trip tests | Add meaningful contract, integrity, compatibility, resource and no-fit tests. |

### Frozen SPEC-07 handoff

Consume `reports/evaluation/spec07-reference/evaluation_manifest.json` and its verified payloads. Related source artifacts are `reports/training/spec06-reference/`, `artifacts/models/training/spec06-reference/`, `reports/features/feature_manifest.json` and `reports/baselines/baseline_manifest.json`.

The recorded decision has:

- `execution_status=complete` and `recommended_model=null`.
- `metric_eligible_models=[]`; both finalists fail `project_macro_f1_target_met`.
- `development_reference=random_forest` and `production_ready=false`.
- Validation macro F1 approximately 0.389920 for logistic regression and 0.430224 for random forest, both below the unchanged 0.45 target.
- Unresolved cutoff, final-test and runtime evidence; supported subgroup concerns and a client-composition ordering reversal.

The implementation must read full-precision values and hashes from verified artifacts, never hard-code the rounded scores above. SPEC-07 verification records 198 passing tests and two reproducible real analyses. Its reproduction is a repeat of the same evidence, not an additional independent evaluation population.

## 3. Scope and non-goals

### In scope

- Explicit research packaging of either or both frozen finalists; default packaging selection is both, not a fallback recommendation.
- Manifest-first verification before trusted deserialization and manifest-last publication.
- Self-contained offline inference assets, approved raw feature schema and environment compatibility policy.
- Strict request validation and stable prediction/probability response contracts.
- Deterministic local batch prediction, bounded chunking and a private output contract.
- Synthetic and validation-only parity verification, no-fit instrumentation and immutable prerequisite checks.
- Documentation, model card, examples with invented IDs, CLI commands and a SPEC-09 handoff.

### Out of scope

- Training, retuning, recalibration, threshold changes, feature reselection, new models or validation-driven model selection.
- Train-plus-validation fitting, new splits, final-test scoring or claiming an untouched historical holdout.
- A production pointer, automatic promotion, registry upload, remote downloads, API deployment, authentication or monitoring infrastructure.
- ONNX conversion, cross-language runtimes, containers or alternate serialization formats in the initial implementation.
- Formal probability calibration, causal refresh benefit, subgroup-specific routing or automatic content changes.
- An arbitrary wall-clock SLA or production hardware certification. SPEC-00 specifies 30,000 items on a standard CPU but does not specify a numeric deadline or hardware profile.

## 4. Packaging and readiness policy

Version 1.0 of the packaging protocol supports `purpose=research` only. Require this purpose explicitly in the packaging CLI so the resulting artifact's intended use is concrete. No additional interactive confirmation is needed for an already authorized implementation or research run.

Research packaging may use either frozen finalist despite the null recommendation, provided the package preserves the rejection decision and limitations. Each package must record `research_only=true`, `production_ready=false`, `recommended_model=null` from the current handoff, the development reference, its own eligibility flags and the decision hash. The fact that a package was successfully built cannot change any of these fields.

Reject `purpose=production`, a requested `recommended` alias when the recommendation is null, unknown families, and attempts to override eligibility through a CLI flag. Even a future non-null SPEC-07 recommendation does not authorize production promotion under this initial protocol. A later acceptance contract must define independent performance evidence, cutoff review and operational readiness.

Packaging both families creates two named packages and a packaging-run manifest. It does not create a `latest`, `best`, `production` or automatically selected serving pointer. Consumer code chooses a concrete package ID/path.

## 5. Proposed requirements and traceability

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

## 6. Configuration and versioning

Add `configs/packaging.yaml` and `configs/inference.yaml`, with closed top-level mappings. Reject unknown fields, unsupported versions, duplicate family entries, nonfinite values and booleans masquerading as integer limits.

| Setting | Proposed default / policy |
| --- | --- |
| `packaging_contract_version`, `package_schema_version` | `1.0` |
| `inference_contract_version` | `2.0`; explicit new interface, preserving legacy record behavior |
| Embedded `bundle_schema_version` | Preserve existing `2.0`; do not relabel or reserialize the source model |
| `purpose` | `research`, explicitly supplied at packaging entry point |
| `families` | Both frozen finalists, or an explicit nonempty subset of those two |
| Package root | `artifacts/packages/<package_id>/` |
| Packaging report root | `reports/packaging/<run_id>/` |
| `overwrite`, `resume`, remote loading | Disabled |
| `include_probabilities` | False by default in inference v2; explicit opt-in |
| `unknown_categories` | Existing fitted encoder's `ignore` behavior, recorded; never extend vocabulary |
| `extra_columns` | Reject in strict v2; legacy behavior stays in the legacy interface |
| `maximum_batch_rows` | 30,000, inclusive; reject larger requests in the initial protocol |
| `chunk_rows` | 5,000, configurable positive integer no greater than the batch limit |
| Numerical threads / fitted estimator jobs | One; reject incompatible fitted settings rather than mutating the model |
| Probability validation tolerance | Absolute 1e-9 for row sums; finite values within [0,1] required |
| Numeric-string coercion | Disabled in the Python v2 interface; explicitly parsed numeric CSV columns only |
| Operational output | Explicit local destination, private, no overwrite; no default row-level artifact publication |

Package identity and model identity serve different purposes. Keep the original SPEC-06 `model_version` unchanged in records. Add the package ID, package schema version and inference contract version at the response-envelope level. Compute a semantic package digest from canonical contract/config/model/decision/environment hashes, excluding physical output paths, timestamps and verification timings. A caller-supplied safe package ID is a human-readable immutable locator; conflicting or occupied IDs fail.

Record the exact installed package versions needed by the fitted pipeline, including Python, numpy, pandas, scipy, scikit-learn, joblib, threadpoolctl and this project's distribution/code digest. Distinguish training versions actually recorded upstream from packaging environment observations. Do not invent missing training-version history. Initial loading requires the declared supported environment, with explicit failure before unpickling when compatibility is not established. Define the supported Python/platform/version checks in code and tests rather than relying on a vague compatibility statement.

## 7. Package preflight, construction and loading

### Build preflight

1. Validate settings, safe run/package IDs, local paths and isolated unoccupied destinations. Resolve symlinks/junctions and reject overlap with source data, existing models, feature/baseline/evaluation reports and other package destinations.
2. Call `load_evaluation` on the selected evaluation directory. Check the decision, evaluation identity, model/report/selection hashes, configuration references and package family's membership in the frozen finalists.
3. Verify the referenced SPEC-06 manifest and all required report/model payloads before trusted loading. Compare the SPEC-07 handoff's model bytes, selection hash, family, feature references and original/effective parameter metadata with the training evidence.
4. Check semantic and byte-hash conventions explicitly: a file SHA-256 and `fingerprint(parsed_json)` are different fields. Do not compare or substitute them accidentally.
5. Validate the full pipeline's expected steps, class set/order, fitted feature configuration, preprocessing behavior and input feature order. Check environment support and fitted resource settings.
6. Snapshot every protected input. `--dry-run` completes these checks and reports intended package paths, families and purpose with zero fits, predictions or writes.

### Construction and publication

- Copy the original joblib bytes exactly into `model.joblib`; do not call `joblib.dump`, refit, mutate metadata inside the model, or rebuild estimators.
- Export the approved inference-only configuration and schema separately. Do not require a full source CSV, split assignment file or training output directory during normal package inference.
- Include a frozen decision snapshot and a model card with eligibility, limitations and provenance. Retain hashes/identities of upstream artifacts as audit references; do not copy source datasets, row predictions or client examples.
- Use a newly created staging directory within the approved root. Write payloads exclusively, validate their hashes and their semantic consistency, and recheck protected inputs.
- Publish the package manifest last. Readers ignore/reject any directory without a valid completion manifest. If a two-family run is interrupted after one package completes, that package may remain individually valid, but the packaging-run manifest must remain incomplete/absent; do not claim the entire run completed.
- Existing completed packages and runs are never overwritten. A failure can retain local staging diagnostics without a valid completion marker.

### Verified runtime loading

Introduce `PackagedPredictor.load(package_dir)` (or an equivalently explicit API) rather than silently changing `Predictor.load` semantics.

Before any joblib deserialization, validate the manifest status/version, exact required payload inventory, safe relative paths, resolved containment, all payload hashes, inference schema and declared runtime compatibility. Reject missing, altered, escaped, duplicate or unsupported payload records. After loading, compare embedded model metadata, class order, feature/preprocessor configuration and model version with the package manifest.

Joblib is a trusted-local-artifact format, not a sandbox. Hashes detect inconsistency against a trusted manifest; they do not prove authenticity if an attacker controls both model and manifest. Accept packages only from the local trusted build process or a separately pinned trusted manifest digest. Do not implement user-uploaded pickle loading or remote URL loading in this phase.

Normal inference must remain usable when original training/evaluation/data paths are unavailable. Package verification should use embedded contract material and trusted manifest hashes; source provenance checks requiring the original repository belong to package construction and audit, not every prediction request.

## 8. Input contract

The strict v2 request contains `content_id` plus the exact approved raw feature columns from the frozen data/feature contract. `client_id`, target labels, `trend_pct`, outcome-window measures, sensitive fields and unrecognized extras are rejected in this interface. IDs are echoed only as request-response correlation keys and never enter fitted transforms.

Input rules:

- Require a dataframe/tabular request with unique column names, at least one row and no more than the configured batch limit. Column order may vary; select features in canonical contract order.
- Require nonempty string content IDs, unique across the entire batch. Reject non-string IDs, null IDs, whitespace-only IDs and leading/trailing whitespace rather than silently trimming or stringifying them. Preserve legitimate leading zeros and literal strings such as `NA`.
- Numeric feature values must be real finite numbers or supported missing values. Reject booleans, nested objects, infinity and invalid numeric text. Enforce configured nonnegative constraints; do not invent new numeric bounds for otherwise valid fields.
- Require all feature columns even when their cells are missing. Impute missing values only through the already-fitted pipeline. Do not infer missing columns or fit statistics from a request.
- Categorical values are strings or supported missing values. Unknown string categories use the frozen encoder policy. Reject lists/dicts and other invalid value types; no adaptive vocabulary or silent case normalization.
- Preserve source row order and do not rely on dataframe index labels being unique or consecutive. Prediction alignment is positional after validation.
- Invalid requests fail as a whole before scoring. Error messages identify safe column names and violation counts/codes, without including raw IDs, feature values or complete rows.

CSV loading must inspect the header for duplicate names before pandas can rename them. Read IDs as strings with an explicit missing-value policy that preserves literal identifiers; parse numeric and categorical feature cells according to the exported schema. Specify encoding/delimiter and missing tokens rather than relying on pandas defaults. Reject ambiguous or malformed CSVs before publishing any output.

## 9. Prediction, probabilities and ranking

### Response contract

The new Python interface returns a versioned result object/envelope with:

- `inference_contract_version`, `package_id`, `package_manifest_sha256` and `model_version`.
- `purpose=research`, `production_ready=false`, row count and probability mode.
- Ordered records preserving the existing fields: `content_id`, `predicted_trend`, `probability_down`, `probabilities`, `model_version`, `warning`.

Validate that output length equals input length, each output ID matches the corresponding input ID, labels belong to the frozen five-class vocabulary, and version/purpose fields agree with the loaded package. Convert numpy scalars to JSON-compatible Python values and reject nonfinite output. Do not allow partially valid output records to escape.

### Optional probabilities

Label-only mode calls `pipeline.predict` once per chunk and never calls `predict_proba`. Set both probability fields to null in this mode.

When explicitly requested, use the existing pipeline's `predict_proba` once per chunk in addition to its label prediction. Validate matrix shape, finite values, [0,1] bounds, normalized row sums within the declared tolerance and correspondence with the actual estimator `classes_`. Build response dictionaries in the exported canonical label order using an explicit class-to-column mapping; never assume `down` is column zero. `probability_down` must equal the mapped `probabilities['down']` value.

Do not recompute labels from probabilities, change thresholds, renormalize invalid outputs or silently fall back when a requested capability is absent. Call failures or malformed outputs fail the entire request. Record that probabilities are model scores without established calibration and do not establish causal need for a refresh. Probability consistency tests are engineering checks, not new model eligibility evidence.

### Review queue

If the new interface exposes review ranking, require explicit probability mode. Sort descending by `probability_down` with a stable sort preserving original row position for ties. Label-only ranking raises a capability error rather than sorting null values. Keep normal `predict` responses in original order. This queue supports authorized research/human review and cannot trigger content changes automatically.

## 10. Batch orchestration and operational privacy

Implement a package-aware batch pipeline and `scripts/predict_batch.py`. Require explicit input, concrete package path, research purpose and output destination. Reject output aliases/symlinks that overlap the input, package or protected reports. Load and validate the complete bounded batch first, including global duplicate-ID checks, then score chunks in original order with a single loaded package.

At the proposed maximum of 30,000 rows, whole-request validation is bounded and avoids reporting partial success before a later invalid chunk. Chunk size controls prediction memory, not permission to bypass the total row limit. Full-batch and chunked output must agree in labels/order and probability values under the declared numerical tolerance.

Stage the full private response and publish it atomically only after every chunk and output record passes. No partial file appears at the requested final destination on failure. Existing outputs are rejected. Prefer JSON envelopes for complete provenance; if CSV is supported, define the fixed column order and required metadata sidecar, including how five class-probability columns and nulls are represented.

Operational output may echo IDs supplied by the authorized caller, as allowed by SPEC-00. This does not authorize placing source validation IDs or downloadable examples in package cards, verification reports, public logs or version control. Synthetic examples use invented IDs. CLI stdout contains aggregate completion/error information and local output paths only; no row dumps. Access control, retention and remote delivery for an application remain SPEC-09/10 concerns.

## 11. Proposed files and APIs

| File | Responsibility |
| --- | --- |
| `configs/packaging.yaml` | Package versions, family/purpose policy, roots and environment requirements |
| `configs/inference.yaml` | Strict v2 parsing, resource/chunk limits and probability settings |
| `utils/config.py` | Closed packaging and inference validators |
| `models/package_artifacts.py` (new) | Safe paths, manifest/payload verification, semantic identity and copy-only publication |
| `inference/contracts.py` (new) | Exported input schema, versioned response envelope and runtime output validation |
| `inference/validation.py` | New strict validator/parser with clearly separate legacy behavior |
| `inference/packaged_predictor.py` (new) | Verified package loading, label/probability modes, stable ranking and bounded inference |
| `inference/predictor.py`, `schemas.py` | Preserve legacy behavior; share compatible helpers only after caller review |
| `pipelines/packaging_pipeline.py` (new) | Verified handoff, research packaging preflight and orchestration |
| `pipelines/inference_pipeline.py` | Explicit package-aware batch entry point while retaining legacy callable compatibility |
| `scripts/package_model.py` (new) | Packaging CLI and read-only dry run |
| `scripts/predict_batch.py` (new) | Strict private operational batch CLI |
| `scripts/verify_packaging.py` (new) | Focused/full/lint checks, two builds, parity, privacy and immutable input verification |
| Tests / readme / SPEC-08 / this plan | Traceable tests, usage, adopted protocol and actual completion evidence |

Suggested interfaces:

```python
run_packaging(packaging_config, inference_config, evaluation_dir,
              training_report_dir, model_dir, run_id, purpose, dry_run=False)
load_package_manifest(package_dir, expected_manifest_sha256=None)
PackagedPredictor.load(package_dir, *, purpose="research",
                      expected_manifest_sha256=None)
predictor.predict(frame, *, include_probabilities=False)
predictor.rank_review_queue(frame, *, include_probabilities=True)
run_packaged_batch(input_path, package_dir, output_path, *, purpose,
                   inference_config, include_probabilities=False)
```

Do not expose a training dataframe, fit function or estimator builder in the packaging/inference APIs. Offline parity verification may independently use the verified development loader; keep that data dependency out of normal serving.

## 12. Artifact layout and reproducibility

Each `artifacts/packages/<package_id>/` contains a fixed inventory:

| Artifact | Contents |
| --- | --- |
| `model.joblib` | Exact unchanged source finalist bytes |
| `model_metadata.json` | Source model metadata, checked against embedded metadata |
| `input_schema.json` | Ordered required features, IDs, types, null/unknown-category policies and constraints |
| `output_schema.json` | Envelope/record versions, labels, probability mode and errors |
| `inference_config.json` | Effective bounded inference defaults and contract fingerprint |
| `decision.json` | Frozen verified research/rejection handoff snapshot |
| `environment.json` | Required versions and observed build environment, with compatibility policy |
| `runtime/` | Exactly one project wheel with a validated distribution/version filename, plus `requirements.txt` containing the supported pinned runtime dependencies |
| `MODEL_CARD.md` | Intended research use, metrics, limitations, provenance and prohibited promotion inference |
| `package_manifest.json` | Completion status, semantic digest, versions, exact payload hashes and source references; written last |

Each `reports/packaging/<run_id>/` contains resolved configuration, a package inventory with manifest hashes, a packaging report and a completion manifest. `verification.json` is a later attestation referencing completion hashes without circular hashing. The verification report contains aggregate parity checks and resource observations, never row predictions.

The installed project code is part of the runtime dependency because the joblib pipeline contains custom feature-engineering classes. Build a project wheel using the existing Python build configuration and record its digest, distribution version and source-tree digest. Copy that wheel and pinned dependency requirements into each package's `runtime/` directory so package directories do not depend on sibling build folders. Validate the actual wheel filename against the declared project distribution/version and include it in the exact manifest inventory. Freeze one verified wheel for both reproducibility builds; rebuilding a wheel with variable archive timestamps must not be confused with model or contract drift. This layout must pass an offline isolated-environment test without importing code from the development checkout. Dependency installation must use already available/local wheel assets or an explicitly authorized dependency setup; packaging must not silently download arbitrary code.

Keep the initial runtime platform/environment support narrow. Do not claim the project wheel alone supplies numpy/scikit-learn dependencies. Document the supported environment and how to install the exact required runtime. If a dependency cannot be provisioned locally, report the isolated-install check as blocked instead of claiming portable verification passed.

Semantic two-build verification compares copied model hashes, inference schemas/configuration, decision hashes, dependency requirements and package content digests. Exclude physical package IDs/paths, build times and measured latencies. Independently validate each package's own manifest and payload hashes. Never allow broad tolerance to hide label/class-order/configuration drift; probability tolerances apply only where explicitly declared.

## 13. Delivery phases and exit criteria

### Phase 0 - Adopt scope, versions and readiness policy

Replace SPEC-08 placeholders with concrete requirements, tests and the research-only outcome. Adopt v2 inference compatibility boundaries, input rules, probability defaults and private operational output policy. Link the SPEC-07 null recommendation and unresolved evidence.

**Exit:** No normative placeholders; packaging success cannot be mistaken for a production recommendation.

### Phase 1 - Package schema and frozen preflight

Implement configuration validators, semantic package identity, safe inventory/path handling, runtime asset layout and read-only source/decision checks. Write failure tests before allowing deserialization of unchecked bytes.

**Exit:** Invalid purpose, family, environment, stale references or unsafe/occupied destinations fail before inference/publication; valid dry run performs no fitting, prediction or writes.

### Phase 2 - Copy-only package publication and loader

Build research packages from original bytes, export schemas, include decision/card/environment references, produce the project wheel and publish manifests last. Implement verified standalone loading and legacy compatibility tests.

**Exit:** Packages load from the supported installed runtime without upstream data/report paths; missing/tampered/escaped/incomplete packages fail, and all original artifact hashes remain unchanged.

### Phase 3 - Strict inference contract

Implement request parsing/validation, label-only and optional probability modes, output checks, positional alignment and stable ranking. Add no-fit and capability spies. Keep legacy six-field record serialization intact.

**Exit:** Accepted requests produce valid, deterministic outputs; invalid requests and malformed estimator output fail without partial results or silent coercion.

### Phase 4 - Private bounded batch workflow

Implement package-aware CLI, CSV header/ID handling, total-row and chunk limits, safe local output staging and aggregate logging. Document JSON/optional CSV provenance behavior.

**Exit:** CLI and Python outputs agree; chunking preserves predictions/order, duplicate IDs across chunks fail globally, and interrupted runs expose no final partial output.

### Phase 5 - Verification and handoff

Run focused/full tests and Ruff, create two isolated research packaging runs from the same frozen inputs, check model-byte identity and package semantic reproduction, run validation parity and isolated runtime checks, and measure the 30,000-row synthetic CPU workload. Update documentation/status with actual evidence only.

**Exit:** SPEC-09 receives concrete package manifests, versioned schemas, example invocation, compatibility evidence and explicit research/readiness limitations. No final-test success or deployment is claimed.

## 14. Verification protocol and test matrix

Use synthetic fixtures for boundary and failure tests. Where a fitted fixture is necessary, fit only small synthetic models before enabling fit spies. Real verification consumes frozen models and never re-enters SPEC-06 search or baseline generation.

For real parity, load and canonically order verified development validation through the established loaders. Form approved inference features with private IDs kept in memory. Compare original pipeline and package outputs on exactly these rows, require exact class labels and the SPEC-06 scoped prediction fingerprint, and validate optional probabilities against the same frozen source pipeline. Publish only hashes, aggregate outcomes and capability/resource counts. Do not score test or repeat eligibility selection.

Use invented IDs and synthetic feature rows for the 30,000-item workload, including representative missing/unknown-category conditions. Do not score the full 30,000-row source dataset, since it includes test rows. Record hardware/environment, cold load, validation, per-mode prediction, serialization time, chunk size and observed resource limits. Separate observations from a future agreed SLA.

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

The no-fit audit should target fitted estimator/feature-transformer operations. If shared sklearn metrics perform internal LabelEncoder bookkeeping during parity checks, document that distinction as SPEC-07 does; do not mistake label-index construction for model training. Instrument prediction populations, call counts and requested probability mode separately.

## 15. Planned commands

These files and flags are deliverables. The commands are illustrative until implementation adopts and provides them.

```powershell
# Read-only verification and intended research package inventory.
.\.venv\Scripts\python.exe scripts/package_model.py --packaging-config configs/packaging.yaml --inference-config configs/inference.yaml --evaluation-dir reports/evaluation/spec07-reference --training-report-dir reports/training/spec06-reference --model-dir artifacts/models/training/spec06-reference --purpose research --run-id spec08-reference --dry-run

# Build both research packages in fresh, isolated destinations.
.\.venv\Scripts\python.exe scripts/package_model.py --packaging-config configs/packaging.yaml --inference-config configs/inference.yaml --evaluation-dir reports/evaluation/spec07-reference --training-report-dir reports/training/spec06-reference --model-dir artifacts/models/training/spec06-reference --purpose research --run-id spec08-reference

# Explicit private inference; the CSV below must be an invented or authorized request fixture.
.\.venv\Scripts\python.exe scripts/predict_batch.py --package-dir artifacts/packages/spec08-reference-random_forest --purpose research --input-path data/examples/synthetic_inference.csv --output-path reports/private-inference/example.json

# Optional probability output is explicit.
.\.venv\Scripts\python.exe scripts/predict_batch.py --package-dir artifacts/packages/spec08-reference-random_forest --purpose research --input-path data/examples/synthetic_inference.csv --output-path reports/private-inference/example-probabilities.json --include-probabilities

# Verifier owns two builds; use fresh IDs if a manual reference build already exists.
.\.venv\Scripts\python.exe scripts/verify_packaging.py --run-prefix spec08-verification

.\.venv\Scripts\python.exe -m pytest tests/unit/test_packaging_config.py tests/unit/test_inference_validation.py tests/contract/test_package_contract.py tests/contract/test_prediction_contract.py tests/integration/test_packaging_pipeline.py tests/integration/test_packaged_inference.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src pipelines scripts tests
```

The verifier must record actual commands, return codes, package/manifest hashes, fit/predict/probability counts, parity scope, privacy scan, runtime-install result, synthetic workload measurements and before/after protected hashes. Avoid overwriting manually created completed runs. Keep scratch outside pytest's active base directory. Do not silently install dependencies, rebuild real models or invoke the SPEC-06 verifier.

## 16. Risks and implementation decisions

| Risk or ambiguity | Proposed resolution |
| --- | --- |
| Packaging interpreted as model acceptance | Support research purpose only, preserve null recommendation and never create a production pointer |
| Raw joblib loader mistaken for verified package loading | Separate explicit loader; validate complete trusted manifest before deserialization |
| Custom feature classes unavailable outside checkout | Build/hash project wheel and validate installed-runtime inference |
| Dependencies or historical versions incompletely recorded | Record observed vs upstream-known versions separately; report unsupported portability honestly |
| Training schema contains outcome fields and IDs | Export approved inference-only schema; reject extras in v2 |
| Strict validation breaks legacy notebooks/tests | Preserve explicit legacy API and six-field records; migrate consumers deliberately |
| CSV parsing changes identifiers or hides duplicate columns | Explicit ID/missing-token policy and raw-header inspection |
| Probability columns mapped in the wrong order | Map through actual estimator classes and validate canonical output mapping |
| Uncalibrated scores mistaken for reliable refresh probabilities | Explicit optional mode and documented uncalibrated/non-causal interpretation |
| Chunking allows duplicate IDs or partial output | Validate the full bounded request first and publish the complete output atomically |
| Arbitrary hardware or an unstated deadline produces false SLA success | Record synthetic workload measurements; defer SLA certification |
| Tests or parity accidentally touch test rows | Use verified validation-only parity and synthetic scale fixtures; instrument exact inference batches |
| Private operational rows appear in shareable evidence | Separate explicit private responses from aggregate verification/card/report artifacts |

## 17. Acceptance checklist and definition of done

- [ ] SPEC-08 adopts concrete scope, requirements, tests and defaults without normative placeholders.
- [ ] Current null recommendation and research-only packaging policy are explicit and enforced.
- [ ] Packaging consumes verified evaluation/training references and unchanged finalist bytes.
- [ ] Dry run performs zero fitting, predictions and writes.
- [ ] Packages have complete, safe, versioned manifests and standalone inference contracts.
- [ ] Runtime compatibility is checked before trusted model loading; integrity failures reject packages.
- [ ] Project runtime assets and supported isolated installation are verified or any external blocker is explicitly recorded.
- [ ] Strict input/ID/numeric/categorical/resource policies and output validation are covered by tests.
- [ ] Label-only mode makes no probability calls; optional probabilities map correctly and remain explicitly uncalibrated.
- [ ] Full/chunked prediction and stable review ranking behave deterministically without model/transformer fitting.
- [ ] Private batch outputs publish atomically; logs/reports/packages contain no source-row examples or private identifiers.
- [ ] Legacy bundle loading, prediction record serialization and existing tests remain compatible.
- [ ] Two builds reproduce semantic package contents and exact source prediction parity on verified validation only.
- [ ] The synthetic 30,000-row workload has measured evidence without an invented SLA.
- [ ] All protected source/split/feature/baseline/training/evaluation/model inputs remain unchanged.
- [ ] Focused/full tests, Ruff, privacy/integrity checks and verification evidence are recorded with actual outcomes.
- [ ] SPEC-09 receives concrete package references, input/output contracts, usage and unresolved readiness limitations.

Completion means reproducible research packaging and a dependable local inference contract. It does not mean that either current finalist qualifies for production, that historical holdout exposure has been resolved, or that deployment is authorized. This request creates only this implementation plan.
