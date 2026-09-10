# SPEC-09 — API or Application

**Status:** Verified  
**Owner:** Babar Ali Khan  
**Outcome:** A tested local HTTP prediction workflow around a verified, explicitly selected research package, with clear input/output contracts and private request handling.

## Purpose

Expose a local FastAPI application that allows an operator to start a pinned research model, inspect its contract, submit a bounded JSON batch of requests, and receive a versioned prediction envelope. This provides a standardized HTTP interface for the research models developed in SPEC-08.

## In Scope

- One local FastAPI process, one selected verified package and explicit research purpose.
- Strict JSON batch prediction, default label-only output and explicit optional probabilities.
- Liveness, operational readiness, public-safe model metadata and inference schema discovery.
- Loopback-only supported launcher, bearer-token protection for model/schema/prediction endpoints, bounded requests and concurrency.
- Interactive API documentation, a synthetic request generator/client example and clear research limitations.
- Sanitized errors, aggregate logging, no request persistence and complete non-streamed responses.
- Synthetic unit/integration/contract tests, real HTTP-to-Python parity, no-fit auditing and a verification report.

## Out of Scope

- Training, retuning, recalibration, model reselection, final-test scoring or changing the null recommendation.
- Remote/public hosting, TLS termination, cloud infrastructure, containers, CI release pipelines and production authentication/authorization systems.
- Multi-user accounts, client tenancy, stored request history, a database, background jobs, polling or durable queues.
- Package upload/download, arbitrary filesystem inputs, remote model URLs, hot reload or a model registry.
- Separate React/Next.js/dashboard UI, HTTP CSV upload, bulk file export, review-queue mutation or automated content changes.
- Drift monitoring, production performance certification or an invented numeric SLA.

## Requirements

| ID | Requirement | Expected Behavior | Tests |
| --- | --- | --- | --- |
| SPEC-09-REQ-001 | Configuration Validation | Validate closed configuration, explicit research purpose and a concrete pinned package before serving | API-T-001/002 |
| SPEC-09-REQ-002 | Lifespan Loading | Load exactly once per lifespan through the verified loader; fail closed on startup errors | API-T-003/004 |
| SPEC-09-REQ-003 | Research Status | Preserve null recommendation, research-only status and original model/package identities | API-T-005 |
| SPEC-09-REQ-004 | Readiness Check | Distinguish process liveness from loaded-model readiness | API-T-006 |
| SPEC-09-REQ-005 | Safe Discovery | Expose safe metadata/schema and useful OpenAPI documentation without private paths or source data | API-T-007/008 |
| SPEC-09-REQ-006 | Input Validation | Validate JSON shape, every row's keys, IDs, types and finite numbers before scoring | API-T-009/010/011 |
| SPEC-09-REQ-007 | Transport Limits | Enforce byte, row, string, chunk and concurrency limits with bounded memory | API-T-012/013/014 |
| SPEC-09-REQ-008 | Response Semantics | Preserve inference v2 response semantics, deterministic order and optional probability behavior | API-T-015/016 |
| SPEC-09-REQ-009 | Authentication | Authenticate protected endpoints and reject unsupported transport/origin behavior | API-T-017/018 |
| SPEC-09-REQ-010 | Error Handling | Return stable sanitized errors and never publish a partially valid response | API-T-019/020 |
| SPEC-09-REQ-011 | Privacy | Keep requests, IDs, feature values, credentials and responses out of persistent logs/artifacts | API-T-021 |
| SPEC-09-REQ-012 | Local Workflow | Provide a repeatable local launch/client workflow without dependency downloads at startup | API-T-022 |
| SPEC-09-REQ-013 | Parity Verification | Verify HTTP/Python parity, no fitting and unchanged upstream evidence | API-T-023/024 |
| SPEC-09-REQ-014 | Handoff Documentation | Document measured local behavior and explicit SPEC-10/11 readiness boundaries | API-T-025 |

## Tests

| Test ID | Scenario | Expected Outcome |
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

## Acceptance Criteria

- [x] SPEC-09 adopts concrete research API scope, requirements, tests and transport defaults.
- [x] Existing package pins/runtime compatibility are enforced; no unsafe loading or fallback.
- [x] Import/startup perform no fitting or predictions; one predictor loads per lifespan.
- [x] Health compatibility, honest readiness and fail-closed startup/shutdown behavior are tested.
- [x] Every row is validated before dataframe coercion and before inference; IDs/order are preserved.
- [x] Byte/row/string/chunk/concurrency bounds apply to real transport behavior.
- [x] Default label-only mode makes no probability calls; explicit scores retain validated mapping and warnings.
- [x] Complete HTTP envelopes match the underlying v2 contract; errors/serialization cannot leak partial results.
- [x] Protected endpoints authenticate; supported launch remains loopback-only with no token leakage.
- [x] Request values, source rows, credentials and responses are absent from persistent logs/reports/caches.
- [x] Metadata, schema, interactive docs and invented client examples are accurate and usable.
- [x] Disconnect/shutdown do not permit overlapping inference or abandoned admission capacity.
- [x] Existing legacy inference and package verification tests remain green.
- [x] Focused/full tests, lint including `app/`, and real local HTTP smoke/parity checks pass.
- [x] Both frozen packages have synthetic HTTP/Python parity with zero real fitting/test scoring.
- [x] Synthetic 30,000-row HTTP workload evidence and before/after protected hashes are recorded.
- [x] Verification artifacts are complete, immutable, aggregate-only and reference actual package/code identities.
- [x] SPEC-10/11 receive launch/configuration, dependencies, lifecycle/resource constraints and unresolved readiness limits.

## Definition of Done

This phase is complete when every acceptance criterion is verified and the final verification report is published.
