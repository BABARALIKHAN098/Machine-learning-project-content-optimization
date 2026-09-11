# Frontend implementation handoff

The workspace is implemented in `app/static/frontend/` and served at `/` by FastAPI.
The existing API routes and local-origin policy are preserved. No model, package,
training, feature, or dataset files were modified.

## Implemented

- Responsive light workspace, connection panel, model context, research notices,
  schema-driven inputs, batch import, results, and session clearing.
- All live-schema fields, explicit null controls, unknown categories, strict ID/type/byte
  validation, duplicate JSON-key detection, and chunked large-batch validation.
- Same-origin bearer requests; current-page credentials; request generation guards;
  explicit retry after capacity errors; readiness recovery and safe error references.
- Five-label summaries, optional score details, content-ID search, label filters,
  25/50/100-row pagination, original input order, and full-envelope JSON downloads.
- Bounded preview/results DOM, inert rendering of submitted strings, keyboard controls,
  visible focus, reduced-motion support, and narrow-screen table scrolling.

## File ownership

| Files | Responsibility |
| --- | --- |
| `app/routes.py`, `app/main.py` | Root HTML route and narrow frontend static mount |
| `app/middleware.py` | Fixed frontend asset names in sanitized route logging |
| `app/static/frontend/index.html` | Semantic document and application regions |
| `app/static/frontend/css/` | Tokens, layout, components, and responsive states |
| `app/static/frontend/js/main.js`, `state.js`, `api.js` | Session lifecycle and API orchestration |
| `schema-form.js`, `strict-json.js`, `validation.js`, `batch-input.js` | Input semantics and import validation |
| `results.js`, `download.js` | Result exploration and explicit export |
| `tests/contract/test_frontend_routes.py` | Static serving, transport headers, access boundaries |
| `tests/frontend/contracts.test.mjs` | Parser, schema, validation, results, session unit checks |
| `tests/frontend/browser.mjs` | Actual browser-to-model integration and responsive checks |

## Design handoff

The UI follows the proposed visual direction and workflow saved in this folder. The
Figma team-selection widget from the planning task remains unresolved; no hosted FigJam
board or Figma screen file is claimed as created. The local Mermaid workflow remains
available in `frontend-workflow.mmd`. Browser screenshots provide implementation references
in `.frontend-work/`; they contain only invented records.

## Operating boundary

Run the frontend on the same origin as the existing local API. No accounts, model switching,
stored history, CSV conversion, cloud deployment, causal recommendations, or automatic
content edits have been introduced. The readme contains launch and test instructions.

## Verification completed

Verified on 2026-09-11:

| Check | Result |
| --- | --- |
| Full Python suite | 310 passed; 29 dependency/metric-fixture warnings |
| Focused frontend/API regression | 51 passed |
| Frontend JavaScript contract suite | 12 passed |
| Ruff (`app tests scripts`) | Passed |
| Browser | Installed Chrome 152.0.7977.83 through the available Playwright module |
| Both frozen research packages | Single-record labels, optional five-class scores, and full JSON export passed |
| Large batch | 30,000 invented records with scores for each model family; 50 visible result/preview rows |
| Random-forest large workflow | 27.7 seconds observed from Run prediction through browser completion |
| Logistic-regression large workflow | 24.9 seconds observed from Run prediction through browser completion |
| Responsive layout | 1440, 768, 390, and 320 px without page-wide overflow |
| Accessibility checks | Native radio keyboard navigation, error focus/links, reduced motion, 200% CSS zoom, sampled text/button contrast |
| Recovery and data lifecycle | Invalid token, 429 delay/manual retry, 503 reconnection, and clearing a delayed response passed |
| Browser data handling | No browser storage/cookies; submitted HTML-like ID rendered as text; no JavaScript page errors |

Timings are observations on this machine, not an inference SLA. The 429/503 and delayed
response scenarios were controlled at the browser network boundary; successful prediction
and download flows used the real local API and frozen packages. Backend tests separately
cover actual server error semantics.

The initial sandboxed Python run hit a Windows DLL-loading restriction. The authorized
test run outside that restriction passed. The `agent-browser` CLI was unavailable in the
local cache, so the installed Playwright module and Chrome were used instead.

Aggregate evidence: `.frontend-work/verification.json` and
`.frontend-work/accessibility.json`. Screenshots: `.frontend-work/desktop-empty.png`,
`workspace-1440.png`, `workspace-768.png`, `workspace-390.png`, `workspace-320.png`, and
`workspace-zoom-200.png`. These local artifacts are ignored by Git. The browser runner
stopped its own servers and browser contexts after verification.

Screen-reader testing with assistive software, Firefox/Safari testing, and hosted Figma
screen creation remain unverified. No cloud deployment was performed.
