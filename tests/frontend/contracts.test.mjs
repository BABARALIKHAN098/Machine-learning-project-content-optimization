import test from 'node:test';
import assert from 'node:assert/strict';
import {parseStrictJSON} from '../../app/static/frontend/js/strict-json.js';
import {validateRequest, assertContract, assertResponse, bytes} from '../../app/static/frontend/js/validation.js';
import {summarize, filterRecords} from '../../app/static/frontend/js/results.js';
import {createSession} from '../../app/static/frontend/js/state.js';

const labels = ['down', 'stable', 'up', 'new', 'flat'];
const contract = {
  input: {features: ['number', 'category'], numeric_columns: ['number'], categorical_columns: ['category'], non_negative_columns: ['number'], labels, id_column: 'content_id', extra_columns: 'reject', missing_columns: 'reject', id_policy: 'unique_nonempty_strings_no_surrounding_whitespace', inference_contract_version: '2.0'},
  output: {inference_contract_version: '2.0'},
  limits: {maximum_batch_rows: 30000, maximum_request_bytes: 33554432, maximum_response_bytes: 67108864, maximum_content_id_bytes: 256, maximum_category_bytes: 1024},
};
const model = {purpose: 'research', production_ready: false, research_only: true, package_id: 'synthetic', model_version: 'frozen', package_manifest_sha256: '1'.repeat(64), family: 'synthetic', limitations: [], limits: contract.limits};
const request = () => ({purpose: 'research', records: [{content_id: 'example', number: 0, category: ''}]});

test('strict JSON preserves values, escaped strings, and safe object keys', () => {
  for (const value of [request(), {text: 'escaped " \\ \n', category: 'NA', literal: 'null'}, {a: [null, false, 1.2e3]}, [0], 1, 'text', null]) assert.deepEqual(parseStrictJSON(JSON.stringify(value)), value);
  const value = parseStrictJSON('{"__proto__":{"x":1}}');
  assert.equal(Object.hasOwn(value, '__proto__'), true);
  assert.equal({}.x, undefined);
});
test('strict JSON rejects duplicate escaped keys, overflow, invalid syntax, and excessive depth', () => {
  for (const raw of ['{"a":1,"a":2}', '{"a":1,"\\u0061":2}', '{"x":{"a":0,"a":1}}', '{"x":NaN}', '1e400', '[1,]', '{"a":1,}', '{"a" 1}', '01', 'true false', '\ufeff{}', '"\n"', '['.repeat(9) + '0' + ']'.repeat(9), '']) assert.throws(() => parseStrictJSON(raw), /Invalid strict JSON/);
  assert.doesNotThrow(() => parseStrictJSON('['.repeat(8) + '0' + ']'.repeat(8)));
});
test('metadata must describe a supported research contract', () => {
  assert.doesNotThrow(() => assertContract(contract, model));
  assert.throws(() => assertContract(contract, {...model, production_ready: true}));
  assert.throws(() => assertContract({...contract, input: {...contract.input, features: ['number']}}, model));
});
test('zero, null, unknown categories and empty strings retain their meaning', async () => {
  for (const value of ['', 'NA', 'null', 'unknown', null]) {
    const input = request(); input.records[0].category = value;
    const result = await validateRequest(input, contract);
    assert.equal(result.count, 0); assert.equal(JSON.parse(result.body).records[0].number, 0);
    assert.equal(JSON.parse(result.body).records[0].category, value);
  }
});
test('invalid numeric values are rejected before serialization can turn them into null', async () => {
  for (const value of [NaN, Infinity, -1, true, '2', undefined]) {
    const input = request(); input.records[0].number = value;
    assert.ok((await validateRequest(input, contract)).count);
  }
});
test('IDs, UTF-8 byte boundaries and unpaired surrogates match transport rules', async () => {
  for (const id of ['', ' leading', 'trailing\n', '\u0085id', 'é'.repeat(129), '\ud800']) {
    const input = request(); input.records[0].content_id = id;
    assert.ok((await validateRequest(input, contract)).count);
  }
  const input = request(); input.records[0].content_id = 'é'.repeat(128);
  assert.equal((await validateRequest(input, contract)).count, 0);
  input.records.push({...input.records[0]}); assert.ok((await validateRequest(input, contract)).count);
});
test('envelope, required keys, batch limits and exact serialized byte limit', async () => {
  for (const input of [{...request(), purpose: 'production'}, {...request(), include_probabilities: 'true'}, {...request(), target: 'down'}, {...request(), records: []}, {...request(), records: [{content_id: 'x', number: 1}]}, {...request(), records: [{...request().records[0], extra: 1}]}]) assert.ok((await validateRequest(input, contract)).count);
  const input = request(); const size = bytes(JSON.stringify(input));
  assert.equal((await validateRequest(input, {...contract, limits: {...contract.limits, maximum_request_bytes: size}})).count, 0);
  assert.equal((await validateRequest(input, {...contract, limits: {...contract.limits, maximum_request_bytes: size - 1}})).count, 1);
});
test('30,000 rows validate, over-limit fails, and cancellation stops chunked work', async () => {
  const input = request(); input.records = Array.from({length: 30000}, (_, i) => ({content_id: `synthetic-${i}`, number: i, category: null}));
  assert.equal((await validateRequest(input, contract)).count, 0);
  input.records.push({content_id: 'overflow', number: 1, category: null});
  assert.equal((await validateRequest(input, contract)).count, 1);
  await assert.rejects(validateRequest(request(), contract, () => false), {name: 'AbortError'});
});
test('errors are counted but only the first 100 are retained', async () => {
  const input = request(); input.records = Array.from({length: 200}, () => ({}));
  const result = await validateRequest(input, contract);
  assert.equal(result.count, 200); assert.equal(result.issues.length, 100);
});
test('response provenance, row order, null scores, and probability mapping are checked', () => {
  const input = request();
  const response = {inference_contract_version: '2.0', package_schema_version: '1.0', package_id: model.package_id, package_manifest_sha256: model.package_manifest_sha256, model_version: model.model_version, purpose: 'research', production_ready: false, row_count: 1, include_probabilities: false, probability_interpretation: 'not_requested', records: [{content_id: 'example', predicted_trend: 'down', probability_down: null, probabilities: null, model_version: 'frozen', warning: 'Human review required; do not automate content changes.'}]};
  assert.doesNotThrow(() => assertResponse(response, input, contract, model));
  assert.throws(() => assertResponse({...response, package_id: 'wrong'}, input, contract, model));
  response.records[0].content_id = 'wrong'; assert.throws(() => assertResponse(response, input, contract, model)); response.records[0].content_id = 'example';
  input.include_probabilities = true; response.include_probabilities = true; response.probability_interpretation = 'uncalibrated_noncausal';
  response.records[0].probabilities = {down: .6, stable: .1, up: .1, new: .1, flat: .1}; response.records[0].probability_down = .6;
  assert.doesNotThrow(() => assertResponse(response, input, contract, model));
  response.records[0].probability_down = .5; assert.throws(() => assertResponse(response, input, contract, model));
});
test('filtering preserves original position and summary includes all labels', () => {
  const records = [{content_id: 'B', predicted_trend: 'down'}, {content_id: 'A', predicted_trend: 'up'}];
  assert.equal(summarize(records, labels).flat, 0);
  assert.deepEqual(filterRecords(records, 'a', 'up').map(({index}) => index), [1]);
  assert.equal(filterRecords(records, '', '').length, 2);
});
test('clear aborts the previous session and removes credentials', () => {
  const state = createSession(); const signal = state.controller.signal; state.token = 'secret'; state.ready = true;
  state.clear(); assert.equal(signal.aborted, true); assert.equal(state.token, ''); assert.equal(state.ready, false); assert.equal(state.generation, 1);
});
