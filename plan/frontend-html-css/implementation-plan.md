# Content Trend Research Workspace: frontend implementation plan

Date: 2026-09-11  
Status: Implementation-ready proposal; frontend code has not been changed.  
Stack: semantic HTML, plain CSS, and vanilla JavaScript ES modules.  
Scope: a local, single-operator interface for the existing research API.

## 1. Intended outcome

Build a responsive workspace where an operator connects to the running API, enters content metrics or imports a JSON batch, validates the inputs, runs classification, reviews results, and explicitly downloads results when needed. The application predicts the labels `down`, `stable`, `up`, `new`, and `flat`. It does not generate content, recommend edits, forecast traffic amounts, or prove the benefit of refreshing a page.

HTML supplies structure and native controls; CSS supplies layout, responsive behavior, and presentation. A small JavaScript layer is necessary for authentication headers, schema-driven controls, JSON validation, API requests, filtering, pagination, and downloads. No React, Next.js, CSS framework, bundler, database, or frontend hosting service is required.

Deliver the frontend at `/` from the existing FastAPI application. Preserve `/docs` as the technical request console. Keep API contracts, frozen models, training pipelines, datasets, and existing plans unchanged.

## 2. Evidence examined

The following repository files are the source of truth for this proposal:

- [Routes](../../app/routes.py): health, readiness, model, schema, predictions, and documentation.
- [Transport middleware](../../app/middleware.py): bearer authorization, exact origin and host checks, CSP, response headers, and logging.
- [HTTP schemas](../../app/schemas.py): strict JSON decoding, field and byte limits, request and response validation.
- [Prediction service](../../app/service.py): single active request, package-derived row limit, worker lifecycle, and readiness failure behavior.
- [Error vocabulary](../../app/errors.py) and [configuration](../../configs/api.yaml).
- [Inference contracts](../../src/machine_learning_project/inference/contracts.py) and [prediction record](../../src/machine_learning_project/inference/schemas.py).
- [Reference input schema](../../artifacts/packages/spec08-final-reference-logistic_regression/input_schema.json): current 19-feature contract. Read the live schema at runtime; do not bind the UI to this package.
- [Existing docs script](../../app/static/docs.js), [API contract tests](../../tests/contract/test_api_contract.py), and [API workflow tests](../../tests/integration/test_api_workflow.py).
- [Previous API implementation plan](../SPEC-09-api-or-application/implementation-plan.md) and [recorded API verification](../../reports/application/spec09-final/verification_report.md).

These findings come from source inspection. Existing verification reports describe prior checks; this planning task did not start a server or rerun inference. No existing Figma design URL was supplied, so the visual specifications below are proposed design decisions, not measurements from an existing design.

## 3. Figma planning artifact and design handoff

An editable FigJam workflow was requested through the connected Figma tool with the title **Content Trend Frontend — Research Workflow**. The tool requires team/organization selection in its widget before it can create the board. At the time this document was written, no board URL was returned. This dependency affects the hosted Figma artifact only; the complete workflow source is saved in [frontend-workflow.mmd](frontend-workflow.mmd).

When the widget completes generation, add the returned FigJam URL here. Do not describe the board as created until that link is available.

### Proposed Figma screen work during implementation

Create a design file named `Content Trend Research Workspace`, with these pages:

1. **Foundations**: color, typography, spacing, radius, focus treatment, and layout grids.
2. **Components**: buttons, labeled inputs, missing-value controls, connection status, notices, tabs, trend badges, tables, and pagination.
3. **Desktop**: 1440 px frames for disconnected, connected/empty, single-record input, batch preview, validation errors, submitting, results, and service errors.
4. **Responsive**: 768 px tablet and 390 px mobile frames for input, errors, and results; verify the implementation additionally at 320 px.
5. **Handoff**: component-to-file mapping, endpoint annotations, keyboard behavior, and fixture names.

Use auto layout for related groups. Define button states (default, hover, focus, disabled, busy), input states (default, focused, invalid, missing), and status states (disconnected, checking, connected, unavailable). Annotate intended HTML elements and CSS token names. Use invented content only. Record final file links and frame IDs in this folder when created.

The FigJam flow supports interaction planning. It is not a pixel-accurate UI mockup. The screen specifications below provide the implementation baseline until the proposed Figma screen work is complete.

## 4. Verified API-to-interface mapping

| Endpoint | Access | Response/use | Frontend behavior |
| --- | --- | --- | --- |
| `GET /health` | No token | `{ "status": "healthy" }` | Optional connection diagnostics; never treat as prediction readiness |
| `GET /ready` | No token | Ready status, research purpose, `production_ready: false`; otherwise 503 | Check on connection and after service failures |
| `GET /v1/model` | Bearer token | Package/model identity, family, limitations, research flags, limits | Populate model panel and research notice |
| `GET /v1/schema` | Bearer token | `input`, `output`, `request`, `limits`, synthetic `example` | Build controls and validation from the running package |
| `POST /v1/predictions` | Bearer token | Complete inference v2 JSON envelope | Submit one validated request and render results |
| `GET /docs` | No token | Existing interactive technical console | Open from secondary navigation |
| `GET /openapi.json` | No token | OpenAPI document | Developer reference; not needed for the main workflow |

Send `Authorization: Bearer <session token>` only to protected same-origin endpoints. Send prediction bodies as uncompressed `application/json`. Do not set `Host`, `Origin`, or `Content-Length` manually in browser code.

The API has no CSV upload, stored history, accounts, model-switching, prediction progress, cancellation, explanation, or content-editing endpoint. JSON import and JSON export are browser operations, not new API routes. Defer CSV input and CSV export to a separately specified enhancement because CSV missing-value and escaping rules differ from JSON.

### Runtime limits

Read effective limits from `/v1/schema` and `/v1/model`; do not use these defaults as permanent frontend constants.

| Setting | Configured default | UI consequence |
| --- | --- | --- |
| `maximum_batch_rows` | 30,000 | Reject empty and oversized batches before submission |
| `maximum_request_bytes` | 33,554,432 bytes (32 MiB) | Measure the actual serialized UTF-8 request |
| `maximum_response_bytes` | 67,108,864 bytes (64 MiB) | Avoid rendering or copying the complete result into the DOM |
| `maximum_content_id_bytes` | 256 | Measure IDs with `TextEncoder`, not character count |
| `maximum_category_bytes` | 1,024 | Apply the same UTF-8 measurement to categories |
| `maximum_active_predictions` | 1 | Disable duplicate submissions; another tab can still cause 429 |
| `chunk_rows` | 5,000 | Server implementation detail, not browser pagination or progress |
| `body_timeout_seconds` | 30 | Upload/body-receipt deadline, not a total inference timeout |

The service can lower the effective row limit using package configuration. A 30,000-row batch must remain a single HTTP request unless the operator explicitly chooses separate runs; never silently split requests and combine their results.

## 5. Information architecture and screens

Use one document with anchor navigation: **Analyze**, **Results**, and **Model details**, plus a `/docs` link. The form and results remain in the same session; no client-side router is necessary.

### A. Application shell and connection

- Header: `Content Trend`, subtitle `Research workspace`, service status, and `Clear session`.
- Permanent compact notice: `Research use only. Human review required; do not automate content changes.`
- Connection card: masked token input, reveal/hide control, `Connect` button, and concise inline connection status. Do not render the token elsewhere.
- After successful connection, show model family/version and collapse the token card into a connection summary with a `Reconnect` action.
- Missing token or failed connection keeps prediction controls unavailable. A reachable health endpoint alone does not unlock them.
- `Clear session` immediately clears credentials, input, results, filter state, file controls, and outstanding response handlers. Clearly label that behavior next to the action.

### B. Analyze content

- Page title: `Analyze content trends`; support text describes submitting existing content metrics.
- Two accessible input tabs: `Single content` and `JSON batch`.
- Desktop: main input panel plus a 280–320 px sidebar for model summary, input guidance, and current limits. Mobile: the sidebar content follows the main panel.
- Shared options: `Include model scores` checkbox, unchecked initially; helper text `Scores are uncalibrated and do not estimate the benefit of editing content.`
- Shared footer: record count, request-size estimate, `Validate inputs`, and primary `Run prediction`.
- Add `Load synthetic example` using the live schema's example. Mark the data as invented; do not load repository source rows into the browser.

### C. Single-content form

Render fieldsets for identity, search metrics, content size, previous 30-day traffic, age/freshness, and categories. Use visible labels and helper text; expose each exact API field name in secondary text. Preserve every required key even when its value is missing.

| Group | API fields | Control and behavior |
| --- | --- | --- |
| Identity | `content_id` | Required text; preserve exact string, reject surrounding whitespace and empty values |
| Search | `search_volume`, `competition`, `cpc` | Number inputs; finite nonnegative values or explicit missing/null |
| Size | `word_count`, `char_count` | Number inputs; finite nonnegative values or missing/null |
| Previous traffic | `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d` | Number inputs; finite nonnegative values or missing/null |
| Age | `content_age_days`, `age_tier_order`, `days_since_last_update` | Number inputs; finite nonnegative values or missing/null |
| Categories | `competition_level`, `content_type`, `main_intent`, `model_used`, `age_tier`, `freshness_tier`, `word_count_tier`, `char_count_tier` | Text inputs plus explicit missing controls |

Current total: 20 keys per record, consisting of one identifier and 19 features (11 numeric, eight categorical).

Do not invent integer-only requirements, currency units, maximum competition values, category enumerations, or automatic tier derivations. The contract accepts nonnegative fractional numeric values and unrestricted category strings. Use `step="any"` for numeric controls. Derive minimum constraints from `non_negative_columns`.

For each feature, provide an explicit `Missing` control. When selected, disable its editor and serialize `null`. Otherwise require valid numeric entry or preserve categorical text exactly, including an empty string. Strings such as `NA` and `null` remain strings. Do not silently trim, normalize case, or replace categories. Explain that missing cells use the model's fitted imputation.

Drive available fields from the live schema; use the table for display grouping. Place unfamiliar future feature names in `Additional fields`. Fail visibly on unsupported schema structures rather than submitting incomplete records.

### D. JSON batch input and validation

- Support a local `.json` file picker and a labeled text area for small pasted request envelopes.
- Require the actual envelope shape: `purpose`, optional `include_probabilities`, and `records`. Do not accept a CSV file or an arbitrary dataset table.
- Imported probability settings synchronize the checkbox; an absent setting means false. Checkbox edits deliberately update that one envelope value.
- Read files as strict UTF-8 using a fatal decoder. Validate file size before allocating a full text copy, then validate the final request size as well.
- Validate strict JSON syntax, duplicate object keys, nonfinite numbers, and nesting beyond eight levels. A dedicated small JSON tokenizer/parser must detect duplicate keys before ordinary object construction; `JSON.parse` alone would silently keep the last duplicate.
- Then validate envelope keys, all record keys, types, null semantics, ID uniqueness, bounds, and byte limits. Reject extra keys such as outcome/target columns.
- Show totals and the first 50 rows in a paginated, horizontally scrollable preview; avoid 30,000 editable row controls.
- Show an error summary with row number, known field name, rule, and a link/focus target. Limit visible errors to 100 while retaining a total count. Do not print full submitted rows in error banners.
- Correct imported batches by editing/reimporting the source. A spreadsheet editor and column mapper are outside initial scope.

### E. Submission and results

- Before sending, snapshot validated records, options, and package identity; freeze input during submission.
- Show an indeterminate `Analyzing N records…` status. The API provides no percentage progress.
- A successful result replaces the previous run atomically. If input changes afterward, clearly mark existing results as belonging to the previous submission.
- Summary: total records and counts for all five labels, including zero-count labels. These are current-batch counts, not historical performance or accuracy metrics.
- Results table: input position, content ID, predicted trend, optional down score, model version, and `Details` action.
- Keep original API order initially; provide content-ID search and label filtering. Show 50 rows per page, with a choice of 25/50/100. Filter counts and pagination apply to the in-memory result, without extra API calls.
- Details expand inline with the record warning and, only when requested, all five probability values in canonical schema order. Optional CSS bars supplement readable numbers.
- With scores disabled, both `probability_down` and `probabilities` are null; show `Not requested`, never zero or a confidence estimate. Display rounded percentages only in the UI; preserve original numbers in exports.
- Use the returned `predicted_trend`; do not recalculate it from the largest displayed score or introduce a decision threshold.
- `Download results JSON` exports the complete original response envelope, including package provenance and warnings. Label it as the complete run even if a filter is active. Create downloads only after an explicit click; revoke object URLs afterward.
- Empty filtered results show `No matching results` with `Clear filters`; this differs from the initial no-run state.

### F. Model details

Show family, version, package ID, purpose, production status, development reference, recommendation, and limitations from `/v1/model`. Put the manifest hash in an expandable technical details block. A null recommendation is `No model recommended for production`, not a missing-data error. There is no model selection dropdown because switching models requires an operator-managed service restart.

## 6. Visual system and responsive CSS

Proposed direction: a calm, light research workspace with white panels, dark readable text, and blue primary actions. Avoid stock photos and decorative charts; the form and results are the main content.

| Token | Initial value/use |
| --- | --- |
| `--color-bg` | `#F6F8FC`, application background |
| `--color-surface` | `#FFFFFF`, panels |
| `--color-text` | `#172033`, primary text |
| `--color-muted` | `#475569`, secondary text |
| `--color-border` | `#CBD5E1`, panel dividers; verify stronger control boundaries as needed |
| `--color-primary` | `#1D4ED8`, primary buttons and links |
| `--color-focus` | `#2563EB`, visible focus ring |
| `--color-danger` | `#B91C1C`, invalid fields/down badge text |
| `--color-success` | `#166534`, ready/up badge text |
| `--color-warning` | `#92400E`, research/busy notice text |
| Spacing | 4, 8, 12, 16, 24, 32, 48 px |
| Radius | 8 px controls, 12 px panels, pill badges |
| Type | System sans-serif; 16 px body, 14 px labels, 24–32 px page title; line-height 1.5 |
| Layout | 1,280 px maximum content width, 24–32 px desktop gutters |

Give `stable`, `new`, and `flat` distinct labeled badges with muted blue, violet, and slate treatments. Color is supplemental; always show the label in text. Test actual text/background pairs before finalizing Figma tokens.

- At 1,024 px and above: two-column workspace and two-column field groups.
- At 640–1,023 px: stack the model panel below inputs; retain paired form fields where they fit.
- Below 640 px: single-column fields, wrapping header/navigation, full-width primary action, 16 px gutters.
- Use CSS Grid for the workspace and form groups; Flexbox for action rows and badges. Use `minmax(0, 1fr)` and `min-width: 0` to prevent content overflow.
- Keep wide tables inside a labeled horizontal-scroll region; never force the whole page to scroll sideways. Long IDs wrap in details and truncate visually in cells with an accessible full value.
- Prefer native controls, external CSS, and text/CSS visuals. No CDN fonts, remote scripts, or inline event handlers.
- Respect `prefers-reduced-motion`; avoid animation beyond an optional small loading indicator. Use explicit focus outlines and comfortable 44 px interaction targets.

## 7. Technical integration

### Serving strategy

Add a `GET /` HTML route and mount only `app/static/frontend/` at `/assets`. Use absolute asset URLs, such as `/assets/css/main.css` and `/assets/js/main.js`. The index HTML may live in that folder, but no broader directory should be mounted. Retain `/docs`, `/docs.js`, `/openapi.json`, and every existing API route.

The middleware allows only the configured loopback host/port and requires the exact same HTTP origin on protected requests. Therefore open the frontend at the API's own origin, normally `http://127.0.0.1:8000/`. A file opened with `file://`, a separate Live Server port, a mismatched `localhost`/`127.0.0.1` origin, or a hosted frontend will not work under this contract. Do not loosen CORS or host checks as a frontend shortcut.

Update middleware route logging to recognize `/` and a fixed set of frontend asset paths without logging arbitrary paths. Preserve `no-store`, request IDs, `nosniff`, origin enforcement, and CSP. External ES modules and CSS fit the current same-origin policy. If worker-based batch validation becomes necessary, verify worker loading under the actual CSP and use a narrowly scoped same-origin worker policy if required.

Use the supported `scripts/serve_api.py` launcher with an existing verified package, trusted manifest pin, research purpose, and `CONTENT_TREND_API_TOKEN` configured by the operator. Configuration defaults do not select a package or pin. The legacy Dockerfile binds a different host and is not the baseline for this local frontend plan. Do not alter frozen package sources merely to serve static assets.

### Proposed file map

```text
app/
  main.py                         # Mount frontend assets
  routes.py                       # Add the root HTML route
  middleware.py                   # Recognize known frontend paths in sanitized logs
  static/frontend/
    index.html                    # Semantic shell, form regions, templates, noscript
    css/
      tokens.css                  # Design tokens
      main.css                    # Layout, components, responsive rules, states
    js/
      main.js                     # Startup and event wiring
      api.js                      # Same-origin requests and normalized API errors
      state.js                    # Session state and generation/request guards
      schema-form.js              # Schema-driven field rendering and serialization
      strict-json.js              # Strict parser with duplicate/depth checks
      validation.js               # Contract, row, type, and byte validation
      batch-input.js              # File/paste import and preview
      results.js                  # Summary, filtering, pagination, record details
      download.js                 # Explicit complete-envelope JSON download
tests/
  contract/test_frontend_routes.py
  frontend/                       # Small JS unit and browser workflow suite
plan/frontend-html-css/
  implementation-plan.md
  frontend-workflow.mmd
```

No runtime Node toolchain is necessary. If automated browser tests use Node tooling, isolate it as development-only configuration with pinned dependencies and a lockfile.

### State and request lifecycle

Use explicit states: `disconnected`, `connecting`, `ready`, `validating`, `invalid`, `submitting`, `results`, and `error`. Keep connection readiness separate from current result availability.

Store the token only in current-page memory and its input control. Do not use cookies, browser storage, IndexedDB, URL parameters, service-worker caches, analytics, or logs for tokens, inputs, or outputs. Explicit downloads are the only planned persistence of user results.

On Connect, check readiness and fetch model/schema with the token. Accept the session only when all required calls succeed and the contract/version/research flags are supported. When reconnecting to a changed package or schema, invalidate the old form validation and label previous results as belonging to their original package.

Give each connection/submission a monotonically increasing generation ID. Ignore late responses after Clear session, reconnect, or replacement. Use `AbortController` for browser requests, while explaining that stopping a browser wait does not cancel the server's prediction worker. Never automatically resubmit an uncertain POST; a timed-out or interrupted request may still be computing.

Validate response envelope shape, supported versions, research flags, row count, expected IDs/order, record fields, labels, and score types before rendering. Unexpected or malformed responses become a contract error. Retain `X-Request-ID` or the error body's `request_id` for a support reference. Do not log request bodies or raw server exceptions.

## 8. Error and recovery behavior

The current API returns `{ "error": { "code", "message", "request_id" } }`. It does not return field-specific validation details. Field-level hints must come from frontend validation; do not invent server diagnostics.

| Status/code | UI behavior | Recovery |
| --- | --- | --- |
| Network failure | Connection error; keep editable input | Check running local service, then reconnect |
| 400 `invalid_json` | Batch syntax/encoding error | Correct source and validate again |
| 422 `invalid_request` | Contract mismatch summary | Review required fields/types/limits; rerun client validation |
| 401 `unauthorized` | Clear invalid token and disable submission | Enter a valid token and reconnect |
| 408 `request_body_timeout` | Upload receipt timed out | Keep input; explicit retry after checking service |
| 413 `request_too_large` | Show active byte/row limits | Reduce batch; no automatic splitting |
| 415 `unsupported_media_type` | Request-format error | Correct client to uncompressed JSON |
| 429 `prediction_capacity_exceeded` | Busy status, no loss of input | Respect `Retry-After` before enabling manual retry |
| 500 `prediction_failed` | Failure notice and request ID | Refresh readiness; operator may need service restart |
| 500 `response_limit_exceeded` | Result exceeded limit | Refresh readiness; restart if needed, then smaller batch/fewer scores |
| 503 `service_not_ready` | Unavailable state | Disable prediction until readiness and metadata are restored |
| 400 `invalid_host` / 403 `invalid_origin` | Unsupported connection address | Open the UI at the configured API origin |
| 404 / 405 | Endpoint integration error | Check paths/methods; no blind retry |
| Non-JSON or unexpected envelope | Invalid server response | Keep input, show safe message and request ID when present |

An output/prediction failure can mark this service unready while `/health` remains healthy. Do not suggest repeated submission as the only remedy. When a new request fails, retain any old successful results only with a clear `Previous run` label.

## 9. Accessibility, large batches, and data handling

- Use `header`, `nav`, `main`, sections, fieldsets, legends, explicit labels, and a skip link. Provide a `noscript` explanation that prediction requires JavaScript.
- Implement proper tab roles, selected states, panel relationships, and arrow-key behavior if using tabs. Native navigation or radio-based switching is an acceptable simpler implementation.
- Link error summaries to affected controls; use `aria-invalid` and `aria-describedby`. Focus the error summary after failed validation and the result heading after success.
- Announce connection, validation, and submission changes in a polite live region; avoid announcing each table row. Use an alert region for actionable failures.
- Use table captions, header scopes, accessible pagination, and visible button labels. Expanded details must expose their expanded state.
- Verify text contrast, control boundaries, focus visibility, keyboard use, 200% zoom, narrow reflow, and reduced motion.
- Render API/user strings through `textContent`, never interpolated `innerHTML`. Restrict dynamic properties to known schema data and safe DOM APIs.
- Keep at most 100 result rows and 50 preview rows in the DOM; keep only necessary data copies in memory. Avoid pretty-printing a full large response on screen.
- For large files, perform validation in scheduled chunks; consider a same-origin Web Worker only if profiling shows blocking. Show a real validation count if available, but never fake inference progress.
- Build fixtures from invented records and synthetic API examples. Never include real datasets, credentials, source-row screenshots, or private prediction outputs in Figma or committed test artifacts.

## 10. Ordered implementation work packages

| Phase | Work | Depends on | Completion evidence |
| --- | --- | --- | --- |
| 1. Design and contract fixtures | Finish FigJam selection; create proposed Figma frames; record tokens and component mapping; capture invented fixtures matching live schema | This plan | Linked board/frames and fixtures covering all five labels, null scores, and errors |
| 2. Static shell | Add root route and narrow asset mount; implement HTML landmarks, tokens, shell, responsive layout | 1 | `/` and assets load from the API origin; `/docs` still works |
| 3. Connection | Implement token control, API client, state guards, readiness and metadata loading, session clearing | 2 | Missing/wrong/valid token workflows and same-origin failures verified |
| 4. Single input | Build schema-driven controls, explicit missing states, numeric and byte validation, synthetic example action | 3 | Exact valid one-record envelope; invalid values blocked locally |
| 5. Batch input | Implement strict JSON parsing, import/paste, batch checks, limited preview, validation summary | 4 | Duplicates/extra fields/UTF-8/size limits covered; large files remain usable |
| 6. Prediction/results | Add guarded submission, safe response handling, summary, table, details, filters, pagination, JSON download | 5 | Scores on/off and multiple-record order preserved; full export matches response |
| 7. Recovery/accessibility | Complete error matrix, stale-result states, keyboard flows, responsive tuning, data clearing | 6 | Error and accessibility checks pass with no credential persistence |
| 8. Verification/handoff | Browser checks against both frozen model families in separate runs; regression tests; update readme and Figma mapping | 7 | Evidence documented, screenshots use invented data, no model/data changes |

Implement one vertical slice first: connect → single synthetic record → label-only result. Then add batch input, optional scores, and result exploration. Do not build decorative dashboard features before that slice works.

## 11. Verification plan

These are checks for the future implementation, not checks claimed as completed by this document.

| ID | Scenario | Required result |
| --- | --- | --- |
| FE-01 | Root/assets/docs routes and wrong asset path | Correct HTML/CSS/JS types, existing docs intact, unknown paths fail safely |
| FE-02 | CSP and origin restrictions | Modules/CSS work on API origin; foreign origins remain rejected |
| FE-03 | Valid/invalid token and clear session | Correct access gating; no storage/log/URL leaks; late responses ignored |
| FE-04 | Schema-driven one-record input | Exactly required keys and JSON types; zero remains zero; missing becomes null |
| FE-05 | Empty, whitespace, duplicate, and multibyte IDs | Backend-compatible rejection, including UTF-8 byte boundaries |
| FE-06 | Numeric strings, booleans, negative values, overflow | Invalid numeric values rejected without silent coercion |
| FE-07 | Unknown categories, empty string, literal `NA`/`null`, actual null | Strings preserved; null is the sole JSON missing marker |
| FE-08 | Duplicate JSON keys, malformed UTF-8, deep nesting, extra/missing keys | Strict import errors, no silent repair or discarded fields |
| FE-09 | Empty batch, active max rows/bytes, boundary plus one | Limits use server values and exact encoded payload size |
| FE-10 | Probabilities off/on | Null values shown as not requested; optional scores mapped to canonical labels |
| FE-11 | IDs/order, summary, filtering, pagination, export | Initial order retained; counts correct; export preserves complete original envelope |
| FE-12 | Double click, second-tab 429, interrupted request | One submission per UI; busy/retry and uncertain completion handled honestly |
| FE-13 | 500/503 with healthy liveness | Prediction disabled until readiness restores; useful recovery message |
| FE-14 | Reconnect or Clear during request | Stale response cannot repopulate cleared or changed session |
| FE-15 | Malicious-looking ID/category strings | Displayed as inert text; no HTML/script execution |
| FE-16 | Keyboard, zoom, 320/390/768/1440 px | Reachable controls, readable errors, visible focus, no page-wide overflow |
| FE-17 | Synthetic 30,000-row label and score results | Bounded DOM, usable pagination/filtering, no fake progress |
| FE-18 | Both frozen model families | UI adapts to returned identity/schema and preserves inference response semantics |

Use focused JS unit tests for the strict parser, contract validation, and result transformations; browser tests for the full operator workflow; Python tests for static routes, response headers, and API regression. Test real large workloads once per needed configuration and record actual durations instead of promising an inference SLA.

Existing backend regression commands, from a provisioned compatible environment:

```text
python -m pytest tests/contract/test_api_contract.py tests/unit/test_api_validation.py tests/unit/test_api_config.py tests/integration/test_api_workflow.py
python -m ruff check app tests scripts
```

Run new frontend tests and a final full `python -m pytest` after backend route changes. Do not rerun training, evaluation, or packaging for a frontend-only delivery. Add the browser-test command to readme after selecting its development-only runner. Verify current browsers using actual installed versions and record which were exercised.

## 12. Acceptance checklist and handoff

- [ ] Frontend is served at the API's local origin with responsive HTML/CSS and vanilla JS.
- [ ] Figma board generation is completed and its URL is recorded; proposed screen frames are linked when implemented.
- [ ] Connection, readiness, active package details, and token clearing work.
- [ ] Every live-schema feature is represented and exact input semantics are preserved.
- [ ] Single-content and strict JSON-batch workflows succeed without private fixture data.
- [ ] Prediction labels, optional scores, provenance, and warnings are faithfully displayed.
- [ ] API errors map to actionable states; cancellation and progress are not overstated.
- [ ] Results can be searched, filtered, paginated, and explicitly exported as full JSON.
- [ ] Keyboard, screen-reader announcements, contrast, reflow, and large-batch behavior are verified.
- [ ] Existing APIs and docs pass regression checks; frozen sources and artifacts are untouched.
- [ ] Readme explains local launch, connection, supported inputs, missing values, downloads, and research limitations.

The only external dependency currently unresolved is the Figma widget's team/organization selection. Visual tokens and layouts are explicitly proposed defaults. CSV handling, stored history, cloud hosting, account authentication, live analytics, model explanations, and content-edit automation are future scope requiring their own contracts.
