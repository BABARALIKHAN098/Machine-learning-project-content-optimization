import {apiRequest, APIError} from './api.js';
import {createSession} from './state.js';
import {SchemaForm, element} from './schema-form.js';
import {BatchInput} from './batch-input.js';
import {Results} from './results.js';
import {assertContract, validateRequest, assertResponse} from './validation.js';

const $ = id => document.getElementById(id);
const session = createSession();
const form = new SchemaForm($('schema-fields'));
const batch = new BatchInput();
const results = new Results();
const waiting = $('schema-fields').cloneNode(true);
let mode = 'single', revision = 0, scoresOverride = false, retryTimer;
function status(message) { $('live-status').textContent = message; }
function controls() {
  $('input-controls').disabled = !session.ready || session.busy;
  $('connect').disabled = session.busy;
  $('reconnect').disabled = session.busy;
  $('predict').disabled = Date.now() < session.retryUntil;
  $('token').disabled = session.busy;
  document.body.classList.toggle('is-busy', session.busy);
  $('analyze').setAttribute('aria-busy', String(session.busy));
  $('connection-status').classList.toggle('connected', session.ready);
  $('connection-label').textContent = session.ready ? 'Connected' : session.busy ? 'Connecting' : 'Disconnected';
}
function clearErrors() { $('error-summary').hidden = true; $('error-list').replaceChildren(); $('error-message').textContent = ''; form.clearErrors(); $('batch-text').removeAttribute('aria-invalid'); }
function showError(message, issues = [], count = issues.length) {
  clearErrors(); $('error-summary').hidden = false;
  $('error-title').textContent = count ? `${count.toLocaleString()} input issue${count === 1 ? '' : 's'}` : 'Unable to continue';
  $('error-message').textContent = message;
  for (const issue of issues) {
    const target = mode === 'single' && issue.field ? form.mark(issue.field) : 'batch-text';
    const text = `${issue.row === null ? '' : `Row ${issue.row + 1} · `}${issue.field ? `${issue.field}: ` : ''}${issue.message}`;
    const item = element('li');
    if (target) { const link = element('a', text); link.href = `#${target}`; link.addEventListener('click', event => { event.preventDefault(); const control = $(target); (control?.disabled ? control.parentElement.querySelector('input[type=checkbox]') : control)?.focus(); }); item.append(link); }
    else item.textContent = text;
    $('error-list').append(item);
  }
  if (count > 100) $('error-list').append(element('li', 'Showing the first 100 issues. Correct the source and validate again.'));
  if (mode === 'batch') $('batch-text').setAttribute('aria-invalid', 'true');
  $('error-summary').focus();
}
function modelDetails(model, contract) {
  $('model-empty').hidden = true;
  $('model-title').textContent = model.family.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
  const facts = {Version: model.model_version, Purpose: 'Research only · not production ready', 'Batch limit': `${contract.limits.maximum_batch_rows.toLocaleString()} records`, 'Request limit': `${(contract.limits.maximum_request_bytes / 1048576).toFixed(1)} MiB`, Recommendation: model.recommended_model === null ? 'No model recommended for production' : String(model.recommended_model), 'Development reference': String(model.development_reference)};
  $('model-facts').replaceChildren(...Object.entries(facts).flatMap(([key, value]) => [element('dt', key), element('dd', value)]));
  $('model-provenance').replaceChildren(element('dt', 'Package ID'), element('dd', model.package_id), element('dt', 'Manifest SHA-256'), element('dd', model.package_manifest_sha256));
  $('model-technical').hidden = false;
  $('model-limitations').replaceChildren(...model.limitations.map(text => element('li', text)));
}
function dirty() { revision++; clearErrors(); results.stale(); $('input-summary').textContent = 'Inputs changed · validate before running'; }
function disconnect() {
  session.ready = false; session.token = ''; $('token').value = '';
  $('connect-form').hidden = false; $('connected-summary').hidden = true;
  $('connection-description').textContent = 'Use the token from your running local API. It stays in this tab.';
}
async function handleError(error, generation) {
  if (generation !== session.generation || error.name === 'AbortError') return;
  if (error.status === 401) disconnect();
  if (error.status === 429) {
    session.retryUntil = Date.now() + Math.max(1000, error.retry);
    clearTimeout(retryTimer); retryTimer = setTimeout(() => { if (generation === session.generation) controls(); }, Math.max(1000, error.retry));
  }
  if (!error.status || error.status >= 500) {
    disconnect();
    // Readiness can be false while health remains healthy. Never unlock from health.
    try { await apiRequest('/ready', {signal: session.controller.signal}); } catch { /* Reconnection remains explicit. */ }
    if (generation !== session.generation) return;
  }
  results.stale();
  showError(`${error.message}${error.requestId ? ` Request ID: ${error.requestId}` : ''}`);
  status('Your inputs are retained. Review the message above before continuing.');
}

$('connect-form').addEventListener('submit', async event => {
  event.preventDefault(); if (session.busy) return;
  const token = $('token').value;
  if (!/^[\x21-\x7E]{32,}$/.test(token)) { showError('Enter the local API token: at least 32 printable ASCII characters without spaces.'); return; }
  const generation = session.invalidate();
  session.ready = false; session.busy = true; clearErrors(); controls(); status('Checking readiness and loading the model contract…');
  try {
    const signal = session.controller.signal;
    const ready = await apiRequest('/ready', {signal});
    if (ready.status !== 'ready' || ready.purpose !== 'research' || ready.production_ready !== false) throw new Error('The service is not ready for research inference.');
    const [model, contract] = await Promise.all([apiRequest('/v1/model', {token, signal}), apiRequest('/v1/schema', {token, signal})]);
    if (generation !== session.generation) return;
    assertContract(contract, model);
    const changed = !session.model || session.model.package_manifest_sha256 !== model.package_manifest_sha256 || JSON.stringify(session.contract.input) !== JSON.stringify(contract.input);
    if (changed) { form.render(contract.input); batch.clear(); scoresOverride = false; $('include-scores').checked = false; }
    session.model = model; session.contract = contract; session.token = token; session.ready = true;
    modelDetails(model, contract); results.stale(); revision++;
    $('token').value = ''; $('token').type = 'password'; $('reveal-token').textContent = 'Show'; $('reveal-token').setAttribute('aria-pressed', 'false');
    $('connect-form').hidden = true; $('connected-summary').hidden = false; $('connected-model').textContent = `${model.family.replaceAll('_', ' ')} · ${model.model_version}`;
    $('connection-description').textContent = 'Connected to your local research API. Credentials remain in this tab only.';
    $('input-summary').textContent = `${contract.input.features.length} features · ${contract.limits.maximum_batch_rows.toLocaleString()} record limit`;
    status('Connected. Add content metrics or load an invented example to begin.');
  } catch (error) { await handleError(error, generation); }
  finally { if (generation === session.generation) { session.busy = false; controls(); } }
});
$('reveal-token').addEventListener('click', () => { const reveal = $('token').type === 'password'; $('token').type = reveal ? 'text' : 'password'; $('reveal-token').textContent = reveal ? 'Hide' : 'Show'; $('reveal-token').setAttribute('aria-pressed', String(reveal)); });
$('reconnect').addEventListener('click', () => { session.invalidate(); disconnect(); results.stale(); controls(); $('token').focus(); status('Enter your token to reconnect. Existing inputs are retained if the model contract is unchanged.'); });
$('clear-session').addEventListener('click', () => {
  const wasBusy = session.busy;
  session.clear(); revision++; scoresOverride = false; clearTimeout(retryTimer); batch.clear(); results.clear(); clearErrors();
  form.fields.clear(); $('schema-fields').replaceChildren(...waiting.cloneNode(true).childNodes);
  $('model-title').textContent = 'Your research model'; $('model-empty').hidden = false; $('model-facts').replaceChildren(); $('model-provenance').replaceChildren(); $('model-limitations').replaceChildren(); $('model-technical').hidden = true; $('model-technical').open = false;
  $('token').type = 'password'; $('reveal-token').textContent = 'Show'; $('reveal-token').setAttribute('aria-pressed', 'false'); $('include-scores').checked = false;
  mode = 'single'; document.querySelector('input[name=input-mode][value=single]').checked = true; $('single-panel').hidden = false; $('batch-panel').hidden = true;
  $('connected-model').textContent = ''; disconnect(); controls(); $('input-summary').textContent = 'No validated input yet';
  status(wasBusy ? 'Session cleared. A submitted prediction may still be running on the local service.' : 'Session cleared. Connect to start a new workspace.'); $('token').focus();
});
for (const radio of document.querySelectorAll('input[name=input-mode]')) radio.addEventListener('change', () => { mode = radio.value; $('single-panel').hidden = mode !== 'single'; $('batch-panel').hidden = mode !== 'batch'; dirty(); });
$('schema-fields').addEventListener('input', dirty);
$('batch-text').addEventListener('input', () => { scoresOverride = false; dirty(); });
$('include-scores').addEventListener('change', () => { scoresOverride = true; dirty(); });
$('batch-file').addEventListener('change', async () => {
  if (!session.ready || session.busy) return;
  const generation = session.generation; const currentRevision = ++revision;
  results.stale(); clearErrors(); session.busy = true; controls(); status('Reading JSON file…');
  try {
    const parsed = await batch.loadFile(session.contract.limits.maximum_request_bytes, () => generation === session.generation && currentRevision === revision);
    if (!parsed || generation !== session.generation) return;
    scoresOverride = false; if (typeof parsed.include_probabilities === 'boolean' || parsed.include_probabilities === undefined) $('include-scores').checked = parsed.include_probabilities ?? false;
    status('JSON file loaded. Validate its records before running.'); $('input-summary').textContent = 'Imported batch · awaiting validation';
  } catch (error) { if (generation === session.generation && error.name !== 'AbortError') { showError(error.message); status('File was not imported. Choose a valid JSON request.'); } }
  finally { if (generation === session.generation) { session.busy = false; controls(); } }
});
$('load-example').addEventListener('click', () => {
  if (!session.ready || session.busy) return;
  const example = session.contract.example;
  if (mode === 'single') form.fill(example.records[0]); else batch.set(example);
  $('include-scores').checked = example.include_probabilities ?? false; scoresOverride = false; dirty(); status('Invented example loaded. These are synthetic values, not source data.');
});
async function run(predict) {
  if (!session.ready || session.busy || (predict && Date.now() < session.retryUntil)) return;
  const generation = session.generation; const currentRevision = revision;
  const alive = () => generation === session.generation && revision === currentRevision;
  session.busy = true; controls(); clearErrors(); status('Validating required fields, values, and request size…');
  try {
    let document;
    if (mode === 'single') document = {purpose: 'research', include_probabilities: $('include-scores').checked, records: [form.read()]};
    else {
      document = batch.read(session.contract.limits.maximum_request_bytes);
      // Do not repair invalid imported types, even if the checkbox was changed.
      if (document && typeof document === 'object' && !Array.isArray(document) && (document.include_probabilities === undefined || typeof document.include_probabilities === 'boolean')) {
        if (scoresOverride) document.include_probabilities = $('include-scores').checked;
        else $('include-scores').checked = document.include_probabilities ?? false;
      }
    }
    const validation = await validateRequest(document, session.contract, alive);
    if (!alive()) return;
    if (validation.count) { showError('Correct these issues and validate again.', validation.issues, validation.count); status('Validation found issues. Nothing was submitted.'); return; }
    $('input-summary').textContent = `${document.records.length.toLocaleString()} record${document.records.length === 1 ? '' : 's'} · ${(validation.size / 1024).toFixed(1)} KiB · validated`;
    if (mode === 'batch') batch.showPreview(document);
    if (!predict) { status('Inputs are valid. Run prediction when you are ready.'); return; }
    results.stale(); status(`Analyzing ${document.records.length.toLocaleString()} records… The API does not report percentage progress.`);
    const response = await apiRequest('/v1/predictions', {token: session.token, body: validation.body, signal: session.controller.signal, maxBytes: session.contract.limits.maximum_response_bytes});
    if (!alive()) return;
    try { assertResponse(response, document, session.contract, session.model); } catch (error) { throw new APIError(error.message); }
    results.set(response, session.contract.input.labels); status(`Completed ${response.row_count.toLocaleString()} predictions. Review the results before making any content decisions.`);
  } catch (error) {
    if (!alive() || error.name === 'AbortError') return;
    if (error instanceof APIError) await handleError(error, generation);
    else { showError(error.message); status('Nothing was submitted. Check the input format.'); }
  } finally { if (generation === session.generation) { session.busy = false; controls(); } }
}
$('validate').addEventListener('click', () => run(false));
$('predict').addEventListener('click', () => run(true));
controls();
