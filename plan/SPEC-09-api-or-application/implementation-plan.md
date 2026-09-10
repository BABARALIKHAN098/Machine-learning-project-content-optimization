# Implementation Plan - API or Application

**Source:** `specs/SPEC-09-api-or-application.md`  
**Status:** Proposed; planning document only  
**Prepared:** 2026-09-10  
**Owner:** Babar Ali Khan  
**Outcome:** A tested local HTTP prediction workflow around a verified, explicitly selected research package, with clear input/output contracts and private request handling.

## 1. Objective and authorization boundary

SPEC-09 currently contains placeholder scope, requirements, tests and decisions. This plan proposes concrete implementation defaults using the existing FastAPI scaffold, the approved SPEC-00 problem definition and the completed SPEC-08 handoff. A later implementation should adopt these requirements in SPEC-09 and record any deliberate deviations.

This task creates this plan only. It does not change the specification or application, install dependencies, start a server, load or score models, rebuild packages, deploy a service or change model acceptance.

Deliver an API-first local application: an operator starts a pinned research model, inspects its contract, submits a bounded JSON batch, and receives the existing versioned prediction envelope. Interactive API documentation and a synthetic client example provide the initial user-facing workflow. A separate dashboard or frontend framework is not required for this phase.

Application readiness and scientific model acceptance remain separate. The current recommendation is null; both finalists remain research-only and `production_ready=false`. An HTTP 200 means a request was processed successfully, not that a model meets the project's macro-F1 acceptance target.

## 2. Repository baseline and authoritative evidence

| Area | Observed state | Planned treatment |
| --- | --- | --- |
| `specs/SPEC-09-api-or-application.md` | Draft with `TBD` placeholders | Adopt concrete requirements/tests during implementation |
| `app/main.py` | FastAPI app with `GET /health` returning `{"status":"healthy"}` | Preserve that liveness response; add application factory and lifecycle |
| `app/routes.py`, `app/schemas.py` | Placeholder modules | Implement HTTP routes and strict transport schemas |
| `pyproject.toml` | API extra includes FastAPI and Uvicorn; setuptools discovers only `src` | Inspect installed versions, add explicit API test dependencies; document that `app/` is a checkout application, not part of the SPEC-08 wheel |
| `PackagedPredictor.load` | Requires explicit research purpose; verifies manifest, payloads, runtime and embedded metadata | Sole model-loading entry point; supply a trusted pinned manifest hash |
| `PackagedPredictor.predict` | Validates the full frame, predicts in bounded chunks, returns inference v2 envelope | Delegate unchanged; no HTTP-specific training or feature logic |
| `rank_review_queue` | Stable descending score order; requires probabilities | Preserve Python API; no separate HTTP ranking endpoint initially |
| `inference/contracts.py` | Strict dataframe/CSV contracts and synthetic request generator | Reuse request semantics; add strict JSON handling in `app/` before dataframe conversion |
| `scripts/predict_batch.py` | Private local CSV-to-JSON batch CLI | Preserve existing behavior; HTTP clients submit JSON |
| SPEC-08 verification | 259 tests and Ruff passed; two reproducible builds; zero fitting | Regression baseline, not a result claimed for SPEC-09 |
| SPEC-10 / SPEC-11 | Deployment and monitoring specifications remain drafts | Hand off operational boundaries without implementing remote deployment or drift monitoring |

Primary local sources:

- [SPEC-00](../../specs/SPEC-00-problem-definition.md).
- [SPEC-08 contract](../../specs/SPEC-08-model-packaging-and-inference.md).
- [SPEC-09 handoff](../../reports/packaging/SPEC-09-handoff.md).
- [Reference package inventory](../../reports/packaging/spec08-final-reference/package_inventory.json).
- [SPEC-08 verification](../../reports/packaging/spec08-final-reference/verification.json).

### Frozen research packages

| Family | Concrete package ID | Trusted manifest SHA-256 recorded by SPEC-08 |
| --- | --- | --- |
| Logistic regression | `spec08-final-reference-logistic_regression` | `9bd1ae7043ea49339bbc21f8311e9b1e296043b83db8f90c25e220a0741e658a` |
| Random forest | `spec08-final-reference-random_forest` | `17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6` |

Read authoritative fields from verified manifests during implementation; do not silently derive a new trusted pin from whatever file happens to occupy a configured path. The operator chooses one concrete package per server process. Random forest remains the development reference, not an automatic fallback or production recommendation.

SPEC-08 verified exact labels on 5,857 development-validation rows per family and measured invented 30,000-row workloads. SPEC-09 should use invented requests for real HTTP parity, not repeat validation scoring or claim new scientific evidence. The supported observed runtime is CPython 3.12.0 on Windows AMD64 with exact versions in each package's `environment.json`.

## 3. Scope and proposed product decisions

### In scope

- One local FastAPI process, one selected verified package and explicit research purpose.
- Strict JSON batch prediction, default label-only output and explicit optional probabilities.
- Liveness, operational readiness, public-safe model metadata and inference schema discovery.
- Loopback-only supported launcher, bearer-token protection for model/schema/prediction endpoints, bounded requests and concurrency.
- Interactive API documentation, a synthetic request generator/client example and clear research limitations.
- Sanitized errors, aggregate logging, no request persistence and complete non-streamed responses.
- Synthetic unit/integration/contract tests, real HTTP-to-Python parity, no-fit auditing and a verification report.

### Out of scope

- Training, retuning, recalibration, model reselection, final-test scoring or changing the null recommendation.
- Remote/public hosting, TLS termination, cloud infrastructure, containers, CI release pipelines and production authentication/authorization systems.
- Multi-user accounts, client tenancy, stored request history, a database, background jobs, polling or durable queues.
- Package upload/download, arbitrary filesystem inputs, remote model URLs, hot reload or a model registry.
- Separate React/Next.js/dashboard UI, HTTP CSV upload, bulk file export, review-queue mutation or automated content changes.
- Drift monitoring, production performance certification or an invented numeric SLA.

The bearer token is a minimal local access control, not user identity or tenant isolation. A later multi-user application must define those capabilities separately. The token protects use of the running local service; trusted package selection remains an operator responsibility.

## 4. Proposed requirements and traceability

| Requirement | Required behavior | Tests |
| --- | --- | --- |
| SPEC-09-REQ-001 | Validate closed configuration, explicit research purpose and a concrete pinned package before serving | API-T-001/002 |
| SPEC-09-REQ-002 | Load exactly once per lifespan through the verified loader; fail closed on startup errors | API-T-003/004 |
| SPEC-09-REQ-003 | Preserve null recommendation, research-only status and original model/package identities | API-T-005 |
| SPEC-09-REQ-004 | Distinguish process liveness from loaded-model readiness | API-T-006 |
| SPEC-09-REQ-005 | Expose safe metadata/schema and useful OpenAPI documentation without private paths or source data | API-T-007/008 |
| SPEC-09-REQ-006 | Validate JSON shape, every row's keys, IDs, types and finite numbers before scoring | API-T-009/010/011 |
| SPEC-09-REQ-007 | Enforce byte, row, string, chunk and concurrency limits with bounded memory | API-T-012/013/014 |
| SPEC-09-REQ-008 | Preserve inference v2 response semantics, deterministic order and optional probability behavior | API-T-015/016 |
| SPEC-09-REQ-009 | Authenticate protected endpoints and reject unsupported transport/origin behavior | API-T-017/018 |
| SPEC-09-REQ-010 | Return stable sanitized errors and never publish a partially valid response | API-T-019/020 |
| SPEC-09-REQ-011 | Keep requests, IDs, feature values, credentials and responses out of persistent logs/artifacts | API-T-021 |
| SPEC-09-REQ-012 | Provide a repeatable local launch/client workflow without dependency downloads at startup | API-T-022 |
| SPEC-09-REQ-013 | Verify HTTP/Python parity, no fitting and unchanged upstream evidence | API-T-023/024 |
| SPEC-09-REQ-014 | Document measured local behavior and explicit SPEC-10/11 readiness boundaries | API-T-025 |

## 5. Architecture, ownership and compatibility

```text
Local client / interactive API documentation
    -> transport admission: host, auth, media type, byte limits
    -> bounded strict JSON decoder and per-record transport validation
    -> single admitted inference task
    -> dataframe adapter preserving raw scalar types and row order
    -> PackagedPredictor.predict (existing full-request validation + chunking)
    -> response contract validation + bounded complete JSON serialization
    -> HTTP response with private/no-cache headers
```

Use `create_app(settings=None, predictor_loader=None)` for testable construction. Application import must not deserialize models or read source datasets. The lifespan initializes the selected predictor once and stores an application service in `app.state`. Use dependency injection for synthetic tests rather than loading real artifacts during test collection.

Keep HTTP configuration, schemas, exceptions, concurrency and adapters in `app/`. The predictor remains the single authority for fitted transformations, feature order, labels, score mapping and inference versions. Do not copy model logic into endpoint handlers.

### Critical SPEC-08 source-digest constraint

Existing packages require an exact digest of all Python files under `src/machine_learning_project`. Even adding an otherwise unused module there changes the supported runtime digest. Therefore implement the initial application outside that tree and test existing packages after dependency setup.

If a necessary inference defect requires a source change, document the change, rerun relevant tests and build new immutable research packages using fresh IDs and fresh pins. Preserve old packages and their manifests. Do not disable source verification, edit completed manifests or claim an old package supports changed code. This conditional repackaging is a separate recorded implementation decision, not a default step of the plan.

The current wheel includes `src` only. For this phase, the API runs from the repository checkout using the installed compatible inference runtime; do not claim the bundled model wheel also installs `app.main`. Application distribution belongs to the deployment handoff unless explicitly added during implementation.

## 6. Configuration and environment

Add `configs/api.yaml` with a single closed `api` mapping; reject unknown keys both at the document root and inside the mapping. Reject duplicate YAML keys, booleans as integer limits, nonfinite limits and unsupported versions. Keep API configuration validation in `app/config.py` to avoid changing the packaged source digest.

| Setting | Proposed default / rule |
| --- | --- |
| `api_contract_version` | `1.0`; HTTP transport version, separate from inference `2.0` |
| `purpose` | Required explicit `research`; production unsupported |
| `package_dir` | Required concrete local package path; no automatic selection |
| `expected_manifest_sha256` | Required 64-character hexadecimal trusted pin |
| `host`, `port` | `127.0.0.1`, `8000`; supported launcher rejects non-loopback hosts |
| Workers / reload | One worker; reload disabled for verification and normal research use |
| `maximum_request_bytes` | 33,554,432 bytes (32 MiB), including the complete JSON body |
| `maximum_response_bytes` | 67,108,864 bytes (64 MiB); preflight serialized response before sending |
| `maximum_batch_rows` | At most 30,000 and no greater than the loaded package limit |
| `chunk_rows` | Default 5,000; positive and at most effective batch limit |
| `maximum_content_id_bytes` | 256 UTF-8 bytes; HTTP-only restriction |
| `maximum_category_bytes` | 1,024 UTF-8 bytes per string; HTTP-only restriction |
| `maximum_active_predictions` | Fixed at one initially; no unbounded pending prediction queue |
| Request body deadline | 30 seconds for body receipt, distinct from an inference SLA |
| Token source | Environment variable `CONTENT_TREND_API_TOKEN`; nonempty operator-generated secret of at least 32 characters |
| CORS | Disabled; same-origin docs/local clients only |
| Persistence | None; no request/response files or response cache |
| Documentation | Enabled for supported loopback use; only synthetic examples |

These transport limits are proposed resource policies, not existing SPEC-08 guarantees. A 30,000-row request can still exceed the byte limit if its strings are large. Document that distinction instead of promising every possible maximum-row request fits.

Inspect installed FastAPI, Uvicorn, Pydantic, HTTPX and test client compatibility before coding against version-specific APIs. Add an explicit HTTP test dependency group or equivalent project-supported declaration. Dependency provisioning must preserve the exact inference versions; avoid an unconstrained upgrade of the model runtime. Record actual API dependency versions used in verification. No install or network access occurs on application startup.

## 7. Lifecycle, concurrency and readiness

Startup sequence:

1. Validate settings and token presence without printing the token or resolved configuration secrets.
2. Resolve the selected local package path and enforce the trusted digest configuration.
3. Call `PackagedPredictor.load(..., purpose="research", expected_manifest_sha256=...)` once.
4. Check the package's supported versions, resource settings and research metadata. Derive effective HTTP row/chunk limits bounded by the package configuration.
5. Build request/documentation schemas from the loaded approved feature schema and initialize bounded concurrency.
6. Set ready only after every startup step succeeds. Perform no fitting or prediction, including warm-up predictions, during startup.

Startup verification failure aborts startup with a sanitized diagnostic code and a nonzero launcher exit. There is no legacy raw-joblib fallback, alternate-family fallback or automatic artifact regeneration. Startup should not require source CSVs or upstream report directories once the concrete package exists.

CPU inference must run away from the async event loop so health requests remain responsive. Use one dedicated worker or an equivalently bounded offload mechanism. Admission capacity must cover the complete prediction operation, including body handling and serialization, with prompt `429` rejection when busy. A semaphore around an unbounded background queue is insufficient.

Python cancellation of an HTTP task does not prove the underlying estimator stopped. On disconnect, retain the admission lease until the worker actually completes; discard its response if the client is gone. Never release the slot immediately and allow another task to overlap the still-running estimator. Avoid a hard inference timeout that pretends to cancel a running thread. Document that hard interruption requires process lifecycle management, deferred to deployment.

During shutdown, mark unready, reject new predictions and drain admitted work before releasing the predictor. Tests must control/block a fake worker to prove shutdown and disconnect behavior. Do not mutate estimator `n_jobs`, fitted state or global model selection while serving.

## 8. HTTP endpoint contract

| Method / path | Access | Success behavior |
| --- | --- | --- |
| `GET /health` | Loopback, no token | Preserve `200 {"status":"healthy"}` as liveness only |
| `GET /ready` | Loopback, no token | `200 {"status":"ready","purpose":"research","production_ready":false}` when operational; `503` otherwise |
| `GET /v1/model` | Bearer token | Allowlisted model/package identity, research status, null recommendation, limitations and active resource limits |
| `GET /v1/schema` | Bearer token | Approved input schema, inference output schema and HTTP request/limit descriptions |
| `POST /v1/predictions` | Bearer token | Complete existing inference v2 prediction envelope |
| `GET /docs`, `GET /openapi.json` | Loopback, no token | Documentation with bearer authorization support and synthetic examples only |

Readiness means a verified research model is operational. It must not reuse the name `production_ready` for service health. Ordinary validation failures do not poison readiness. A failed model-output invariant marks the service unready until restart; subsequent predictions return `503`. Unexpected internal faults should use a documented fail-closed service policy.

`/v1/model` must not return absolute filesystem paths, environment secrets, arbitrary embedded metadata, training IDs or raw provenance documents. Allowlist fields instead of returning `predictor.metadata` wholesale. The manifest hash and package ID are public-safe operational identities, not secrets.

The API version lives in `/v1` and an `X-API-Contract-Version: 1.0` response header. Keep `inference_contract_version="2.0"` and `package_schema_version="1.0"` unchanged in prediction envelopes. Add a server-generated `X-Request-ID` header for correlation without changing record identity or accepting attacker-controlled correlation text into logs.

## 9. Strict JSON request handling

Request envelope:

```json
{
  "purpose": "research",
  "include_probabilities": false,
  "records": [
    {
      "content_id": "example-0001",
      "search_volume": 1200,
      "competition": 0.4,
      "cpc": 1.5,
      "word_count": 1000,
      "char_count": 6000,
      "impressions_prev_30d": 500,
      "clicks_prev_30d": 25,
      "sessions_prev_30d": 30,
      "content_age_days": 180,
      "age_tier_order": 2,
      "days_since_last_update": 60,
      "competition_level": "medium",
      "content_type": "article",
      "main_intent": "informational",
      "model_used": "synthetic-example",
      "age_tier": "synthetic-age",
      "freshness_tier": "synthetic-freshness",
      "word_count_tier": "synthetic-length",
      "char_count_tier": "synthetic-length"
    }
  ]
}
```

All values above are invented, including category labels. Unknown categories follow the frozen encoder policy; these are not exported vocabulary examples. Generate the delivered JSON fixture from the current verified schema and an invented request so it cannot silently diverge from the actual contract.

Transport and validation order:

1. Validate host policy, token and supported content type before reading/scoring a prediction request. Accept JSON with UTF-8 semantics; reject compressed request bodies and unsupported content encodings initially.
2. Enforce actual streamed byte count, including requests without `Content-Length`; do not trust that header as the only bound. Bound body receipt time and handle early disconnect.
3. Decode strict UTF-8 and JSON. Reject duplicate keys at every object level, trailing content, nonstandard NaN/Infinity literals, nonfinite parsed numbers such as `1e999`, excessive nesting and malformed values. Do not echo decoder excerpts.
4. Require exactly the allowed envelope fields. `purpose` is required and equals `research`; `include_probabilities` defaults to false and accepts only a JSON boolean. Reject request-supplied model paths, hashes, family aliases, chunk overrides or output paths.
5. Require `records` to be a nonempty list within the effective row limit. For **each individual record**, require exactly `content_id` plus every approved raw feature key. Checking only the dataframe's union of columns would hide a missing field in one row behind another row's key.
6. Validate raw scalar types before dataframe construction. Mixed booleans/numbers or numeric-looking strings must not be silently converted by dataframe or validation-library coercion. Use a type-preserving adapter; do not rely on inferred dataframe dtypes to recover original JSON types.
7. IDs are nonempty unique strings without surrounding whitespace; preserve `001` and literal `NA`. Apply UTF-8 string byte limits. Duplicate IDs anywhere in the batch fail globally.
8. Numeric features accept finite JSON numbers or null, with configured nonnegative constraints. Reject booleans and text. Categorical features accept strings or null. In JSON, strings such as `NA`, `null` or an empty string remain strings; the CSV missing-token policy does not apply. Null represents a missing cell, not permission to omit a field.
9. Reject `client_id`, `trend_pct`, outcomes and unapproved extras; do not accept HTTP parameters that re-enable excluded features.
10. Delegate the complete validated dataframe to the existing predictor, which performs its own canonical contract checks before scoring chunks.

Do not invent new numeric feature ranges or normalize category/ID strings. Schema discovery should describe required-versus-nullable fields, approved order, unknown-category handling, class vocabulary and HTTP-only byte limits explicitly.

## 10. Prediction response and failure contract

Return the predictor's complete envelope with package ID/hash, original model version, research purpose, row count, probability mode and ordered records. Validate these identities against the loaded package before sending; a response serializer must not strip existing fields or silently coerce an invalid result into a valid-looking response.

- Default mode calls `predict` once per chunk and never `predict_proba`; both probability fields are null.
- Explicit probability mode preserves actual-estimator class mapping and canonical label dictionary order. Scores remain uncalibrated and non-causal; do not derive new labels from them.
- Input row order and IDs remain exact. Normal prediction does not sort or prioritize results.
- JSON output must be finite and fully serialized within the response byte limit before response headers/body are sent. Do not stream partial row results in this phase.
- Prediction, model/schema metadata and error responses include `Cache-Control: no-store`; do not configure response caching or browser persistence.

Error envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Request does not match the inference contract.",
    "request_id": "server-generated-id"
  }
}
```

| HTTP status | Stable code examples | Trigger |
| --- | --- | --- |
| 400 | `invalid_json` | Malformed/duplicate-key/nonstandard JSON |
| 401 | `unauthorized` | Missing or invalid bearer token; include appropriate authentication challenge |
| 408 | `request_body_timeout` | Body not received within the declared transport deadline |
| 413 | `request_too_large` | Actual body bytes exceed limit |
| 415 | `unsupported_media_type` | Unsupported content type or content encoding |
| 422 | `invalid_request` | Wrong envelope/row types, missing/extra keys, IDs, row limit or numeric constraints |
| 429 | `prediction_capacity_exceeded` | Another admitted prediction is still active; bounded retry guidance |
| 500 | `prediction_failed`, `response_limit_exceeded` | Estimator/output/serialization failure; no partial success |
| 503 | `service_not_ready` | Predictor unavailable, shutting down or fail-closed after model fault |

Install sanitized validation and exception handlers. Framework default validation errors can contain raw `input` fields; never return them directly. Safe details may include known schema field names and violation counts, but not submitted keys, IDs, values, token text, traceback excerpts or package paths. Do not parse the existing predictor's exception strings to distinguish client errors from model errors: validate client inputs before invocation and treat a subsequent predictor/output failure as an internal failure.

## 11. Local access, privacy and logging

Require constant-time comparison against the configured bearer secret for protected endpoints. Fail startup when it is absent; do not ship a working default token or accept it in query strings. Documentation should show authorization without saving credentials to localStorage or enabling persisted authorization in API docs.

The supported launcher binds loopback only, configures allowed hosts, disables proxy-header trust and leaves CORS disabled. Reject foreign browser origins on protected operations; test same-origin docs and direct non-browser clients separately. No network-facing trust claim follows from a host header check: binding is the launcher/server responsibility. Document direct Uvicorn overrides and remote reverse proxies as unsupported operation in this phase.

Log only generated request ID, fixed route template, status/error code, duration, row count after validation, probability mode and public-safe package identity. Never log authorization headers, raw URLs/query strings, request bodies, IDs, category values or complete responses. Avoid framework access logging that includes attacker-controlled paths/query strings; configure or replace it deliberately. Disable debug traceback pages.

No request rows, predictions, uploads or tokens are written to disk. Release request references after completion and retain no response cache. Do not claim cryptographic memory erasure. Verification artifacts may contain invented examples, aggregate timings and hashes; seeded private test identifiers must not appear in logs or shareable reports. Future access control, retention and multi-user delivery policies belong to SPEC-10/11 or a separately approved application extension.

## 12. Proposed files and responsibilities

| File | Responsibility |
| --- | --- |
| `configs/api.yaml` | Closed local API settings and explicit package/pin fields |
| `app/config.py` | Config parsing, environment secret resolution and startup policy |
| `app/main.py` | App factory, lifespan and dependency wiring; preserve health response |
| `app/routes.py` | Health/readiness/model/schema/prediction endpoints |
| `app/schemas.py` | Strict transport/envelope and response/OpenAPI schemas derived from the approved contract |
| `app/service.py` | Single loaded predictor, bounded worker admission and request-to-dataframe adapter |
| `app/middleware.py` | Body/host/origin limits, generated request IDs and no-cache behavior |
| `app/errors.py` | Stable error mapping and sanitization |
| `app/logging.py` | Allowlisted aggregate event formatting, if substantial enough for a separate module |
| `scripts/serve_api.py` | Supported loopback, one-worker launcher; no implicit installs |
| `scripts/create_api_example.py` | Create an invented schema-compatible JSON request at an explicit fresh path |
| `scripts/verify_api.py` | Full verification orchestration, owned server lifecycle and aggregate evidence |
| `data/examples/synthetic_api_request.json` | Invented complete JSON request; no source rows |
| `tests/unit/test_api_config.py`, `test_api_validation.py` | Config and strict adapter boundaries |
| `tests/contract/test_api_contract.py` | Routes, schemas, response/error identity, privacy and OpenAPI |
| `tests/integration/test_api_workflow.py` | Lifecycle, auth, concurrency, disconnect/shutdown and client behavior |
| `pyproject.toml` | Explicit API test dependency declaration, only as necessary |
| SPEC-09 / README / this plan | Adopted protocol, local usage, actual completion evidence |
| `reports/application/<run_id>/` | Aggregate verification JSON/report and completion manifest; no operational rows |

Module boundaries may be consolidated when small; the responsibilities and tests are required, not an arbitrary file count. Existing inference source, package directories and batch APIs remain immutable unless the source-digest decision in section 5 is explicitly exercised.

## 13. Delivery phases and exit criteria

### Phase 0 - Adopt product scope and environment

Replace normative SPEC-09 placeholders with the API-first research scope, endpoint contract, defaults and requirement/test IDs. Confirm one concrete package and trusted pin through configuration. Inventory installed API dependencies and exact model environment; resolve any local dependency setup explicitly.

**Exit:** Adopted specification; no implied production acceptance; supported launch path and dependency requirements documented.

### Phase 1 - App lifecycle and verified model service

Build the factory, config parser, lifespan, loader injection, single-worker service and readiness state. Preserve `/health`. Validate startup failures and one-load/no-fit behavior using synthetic fakes and loader spies.

**Exit:** A valid lifecycle becomes research-ready without predictions; invalid package/config/runtime fails before accepting requests; importing modules has no model-loading side effects.

### Phase 2 - Strict transport and prediction endpoint

Implement bounded JSON decoding, per-record keys/types, full-batch validation, dataframe adaptation, prediction delegation, output validation and complete serialization. Add negative tests before exposing success paths for malformed requests.

**Exit:** Valid JSON matches Python results; invalid later rows trigger zero scoring; label-only mode triggers zero probability calls; no partial response on model/output failure.

### Phase 3 - Local access and operational behavior

Add bearer auth, host/origin/media policy, no-cache responses, safe logs/errors, concurrency admission and shutdown/disconnect handling. Keep the event loop responsive with a deliberately blocked fake model test.

**Exit:** Unauthorized/oversized/busy requests fail predictably; no overlapping model work or leaked private values; worker capacity is released only when computation ends.

### Phase 4 - User-facing contract and examples

Add safe `/v1/model`, `/v1/schema`, OpenAPI docs, the synthetic JSON fixture, example generator and launcher. Document research warnings, null probabilities, input timing, private local use and actionable sanitized errors. Ensure interactive docs can authorize and submit the same invented request.

**Exit:** A local user can start the service, inspect schema, authorize and submit a request using only documented steps; legacy CLI and health behavior remain compatible.

### Phase 5 - Verification and handoff

Run focused tests, the full suite and lint including `app/`; start a real owned loopback server with a temporary token. Verify both frozen packages in separate server lifecycles using synthetic requests and direct Python parity. Measure the complete 30,000-row HTTP workflow, audit fits/calls/populations, inspect logs and compare protected hashes.

**Exit:** Actual verification artifacts identify code/dependency versions, package pins, commands, results, workload observations and unresolved deployment boundaries. Mark checklist items only from observed evidence.

## 14. Test matrix

| Test ID | Scenario | Expected outcome |
| --- | --- | --- |
| API-T-001 | Valid config with each concrete research package | Deterministic effective limits and package selection |
| API-T-002 | Unknown/duplicate fields, missing secret/pin, unsafe remote path, production purpose, bool limits, non-loopback launcher | Startup rejection; no model scoring or secret output |
| API-T-003 | Missing/tampered/incomplete package, bad trusted pin, incompatible runtime | Loader rejects; startup does not serve predictions; no legacy fallback |
| API-T-004 | Module import, lifespan entry/exit, repeated requests | No import-time loading; exactly one load per lifespan; zero startup fits/predictions |
| API-T-005 | Model info and response identities | Research purpose, false production readiness and frozen recommendation/model/package values preserved |
| API-T-006 | Healthy, loading/unavailable, busy, model-fault and shutdown states | Liveness compatibility; truthful readiness; busy alone does not mean model failure |
| API-T-007 | Metadata/schema responses | Approved fields only, no filesystem paths, secrets, raw upstream documents or rows |
| API-T-008 | OpenAPI and interactive-doc request schema | Required/nullable features, auth, defaults, probability semantics and errors accurately described |
| API-T-009 | Malformed UTF-8/JSON, duplicate keys, NaN/Infinity, overflow float, nesting | Sanitized rejection before predictor calls |
| API-T-010 | Missing key in only one row; extras, outcome fields, nested values, mixed bool/number or numeric text | No dataframe coercion/union-of-columns loophole; whole batch rejected |
| API-T-011 | IDs with leading zeros/literal NA, null/duplicate/non-string/whitespace IDs; JSON category null versus NA string | Exact preservation of valid inputs and global invalid-ID rejection |
| API-T-012 | Empty/30,000/30,001 rows; configured stricter limits; multibyte string boundaries | Inclusive effective limits; transport restrictions documented |
| API-T-013 | Oversized body with/without honest Content-Length, compressed input, oversized response, slow body | Bounded reading/serialization; stable status; no final partial response |
| API-T-014 | Concurrent requests, disconnect during running worker, shutdown, live health check | One real computation at a time; prompt busy rejection; capacity/drain behavior correct |
| API-T-015 | Label-only versus explicit probability mode and absent probability capability | Correct call counts and null fields; explicit failures, no hidden probability work |
| API-T-016 | Class-order mapping, non-divisible chunks, bad output labels/scores/length/identity | HTTP/Python parity or sanitized whole-response failure; no renormalization |
| API-T-017 | Missing/wrong/valid token and malformed auth | Protected routes enforce access; token never echoed/logged; health remains available |
| API-T-018 | Foreign origin/host, unsupported content type/encoding and unknown routes | Declared local transport policy and sanitized errors |
| API-T-019 | Default validation errors and estimator exception containing seeded private values | Error body and logs contain no raw inputs or traceback details |
| API-T-020 | Prediction/serialization failure after an earlier valid chunk | No success headers or partial records; readiness follows defined fault policy |
| API-T-021 | Seeded private IDs/categories/token across logs, docs, response cache and reports | Only authorized prediction response echoes IDs; no persistence or private artifact leaks |
| API-T-022 | Documented launcher + synthetic generator + HTTP client + shutdown | Repeatable local workflow, fresh output handling, no dependency download |
| API-T-023 | Each real frozen package, synthetic full/chunked label/probability requests | Exact labels/order/IDs, probability tolerance 1e-9, original package/model identity |
| API-T-024 | No-fit spies, exact synthetic population, protected snapshots, legacy regressions | Zero real fitting/test scoring, unchanged upstream files and passing prior APIs/tests |
| API-T-025 | Synthetic 30,000-row HTTP workload and verification report | Actual hardware/timings/resources/statuses recorded; no production/SLA claim |

Use small fitted synthetic fixtures only when a fake cannot test the required behavior. Enable model/transformer fit spies after fixture construction. Do not invoke tuning/baseline builders on real data. Spy at the estimator/predictor boundary rather than relying solely on route mock call counts.

Tests for byte admission and disconnects must exercise streamed ASGI input or a real HTTP server; an in-process test client that buffers the entire body cannot prove those properties alone. Browser documentation smoke testing supplements, rather than replaces, HTTP and OpenAPI contract tests.

## 15. Verification protocol and artifacts

`scripts/verify_api.py` should own a fresh run ID, temporary secret, loopback port and server process. It must detect server startup failure and terminate only its own process on success/failure. Use a supported hidden background process on Windows. Avoid pytest's active base directory for verifier scratch.

Verification order:

1. Snapshot hashes for selected completed packages and protected source/split/feature/baseline/training/evaluation/model inputs. Include API code/configuration and dependency versions in the report identity, excluding the secret.
2. Run focused API tests, full repository tests and Ruff over `src pipelines scripts tests app`. Save actual commands, exit codes and sanitized output.
3. Start a server with one configured pinned package, wait for operational readiness, and prove `/health` remains compatible.
4. Submit only invented rows in label-only and probability modes. Compare decoded HTTP envelopes to direct `PackagedPredictor.predict`; exclude transport request-ID headers from equality, but do not exclude package/model fields.
5. Repeat for the other concrete package in a fresh lifecycle. Do not switch packages through an HTTP request.
6. Measure 30,000 invented rows in both modes, within actual byte limits. Record startup/load time, request bytes, validation/prediction/serialization timing where instrumented, end-to-end HTTP latency, response bytes, chunks, thread/worker settings and peak memory where available. Separate warm/cold observations and avoid calling them production benchmarks.
7. Exercise invalid later rows, auth failures, overload/disconnect, output failure and no-persistence checks. Record fit/predict/probability counts and ensure the real request population is synthetic only.
8. Stop the owned server; verify protected hashes are unchanged and scan logs/reports for seeded private identifiers, values and token text.
9. Write aggregate `verification.json`, `verification_report.md` and a completion manifest last under `reports/application/<run_id>/`. Reject occupied completed run IDs. Failed verification must not produce a complete marker.

The completed report records `purpose=research`, `production_ready=false`, `recommended_model=null`, API/inference/package versions, trusted package hashes, actual test counts, privacy/immutability outcomes and any external blockers. Request/response bodies from private fixtures are not verification payloads. Do not claim the previous 259-test count as the new final count.

Dependency setup, unavailable browser automation or inability to bind/start a local process must be reported accurately if encountered. Complete unaffected checks; do not fabricate a live-server or fresh-install result. No production deployment is part of this verifier.

## 16. Planned commands and local workflow

These commands describe deliverables and are illustrative until implementation provides them. Paths are relative to the repository root. In Bash on this Windows checkout, activate with `source .venv/Scripts/activate` or call `.venv/Scripts/python.exe` directly.

```bash
# The shell environment must contain an operator-generated CONTENT_TREND_API_TOKEN.
# Do not put a literal credential in committed config or example commands.

python scripts/serve_api.py --config configs/api.yaml --purpose research --package-dir artifacts/packages/spec08-final-reference-random_forest --expected-manifest-sha256 17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6 --host 127.0.0.1 --port 8000

# In another terminal, create an invented request at a fresh explicit destination.
python scripts/create_api_example.py --package-dir artifacts/packages/spec08-final-reference-random_forest --expected-manifest-sha256 17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6 --rows 5 --output-path data/examples/local_api_request.json

python -m pytest tests/unit/test_api_config.py tests/unit/test_api_validation.py tests/contract/test_api_contract.py tests/integration/test_api_workflow.py
python -m pytest
python -m ruff check src pipelines scripts tests app
python scripts/verify_api.py --run-id spec09-reference
```

Use `/docs` at `http://127.0.0.1:8000/docs` for an interactive request after authorization. A delivered Python client example should read the bearer token from the environment, submit the synthetic JSON to `/v1/predictions`, check status/content type, and inspect only an aggregate response summary by default. Do not encourage copying real source rows into documentation.

Define configuration precedence explicitly: declared CLI overrides for package/pin/purpose/host/port, then YAML; token from the environment only. No implicit current-directory model discovery or `latest` alias. Treat all commands here as proposed interfaces and align them with final `--help` output before completion.

## 17. Risks and decisions to preserve

| Risk / ambiguity | Proposed resolution |
| --- | --- |
| API availability interpreted as production model acceptance | Distinct readiness status; immutable research metadata and null recommendation |
| Placeholder specification encourages uncontrolled UI/deployment scope | API-first local workflow with docs; separate dashboard and remote hosting excluded |
| API additions invalidate exact inference source digest | Keep HTTP code in `app/`; explicit fresh-package procedure if a source change is necessary |
| Installing API tools upgrades model dependencies | Inspect/preserve pinned inference runtime; record API versions and actual setup |
| Dataframe construction masks per-row omissions/coerces booleans | Validate each raw JSON record and scalar before type-preserving adaptation |
| Framework validation exposes private input values | Sanitized handlers and seeded privacy tests for every failure path |
| Body-size check trusts Content-Length or runs after buffering | Bound streamed bytes before decoding; test absent/misleading headers |
| Async cancellation releases capacity while sklearn still runs | Worker-owned lease through actual completion; no pretend thread cancellation |
| Global numerical thread settings collide across requests | One worker/active model task and fixed resource policy |
| Model-path request field permits arbitrary deserialization | Package selection only at trusted startup; request schema rejects paths/aliases |
| Current wheel mistaken for a complete installed API | Document checkout-based `app/`; deployment distribution remains separate |
| Local access token mistaken for tenant authorization | Explicit single-operator local boundary; multi-user controls deferred |
| HTTP workload measurements treated as an SLA | Record hardware, sizes, modes and observed timings only |
| Report publishing leaks data or overwrites evidence | Aggregate-only reports, privacy scan and fresh manifest-last publication |

## 18. Acceptance checklist and definition of done

- [ ] SPEC-09 adopts concrete research API scope, requirements, tests and transport defaults.
- [ ] Existing package pins/runtime compatibility are enforced; no unsafe loading or fallback.
- [ ] Import/startup perform no fitting or predictions; one predictor loads per lifespan.
- [ ] Health compatibility, honest readiness and fail-closed startup/shutdown behavior are tested.
- [ ] Every row is validated before dataframe coercion and before inference; IDs/order are preserved.
- [ ] Byte/row/string/chunk/concurrency bounds apply to real transport behavior.
- [ ] Default label-only mode makes no probability calls; explicit scores retain validated mapping and warnings.
- [ ] Complete HTTP envelopes match the underlying v2 contract; errors/serialization cannot leak partial results.
- [ ] Protected endpoints authenticate; supported launch remains loopback-only with no token leakage.
- [ ] Request values, source rows, credentials and responses are absent from persistent logs/reports/caches.
- [ ] Metadata, schema, interactive docs and invented client examples are accurate and usable.
- [ ] Disconnect/shutdown do not permit overlapping inference or abandoned admission capacity.
- [ ] Existing legacy inference and package verification tests remain green.
- [ ] Focused/full tests, lint including `app/`, and real local HTTP smoke/parity checks pass.
- [ ] Both frozen packages have synthetic HTTP/Python parity with zero real fitting/test scoring.
- [ ] Synthetic 30,000-row HTTP workload evidence and before/after protected hashes are recorded.
- [ ] Verification artifacts are complete, immutable, aggregate-only and reference actual package/code identities.
- [ ] SPEC-10/11 receive launch/configuration, dependencies, lifecycle/resource constraints and unresolved readiness limits.

Completion means a dependable local research HTTP workflow. It does not mean either finalist is accepted for production, historical holdout/cutoff issues are resolved, a fresh machine installation is certified, or remote deployment is authorized. All checklist items remain open until implementation and verification occur.
