export const bytes = value => new TextEncoder().encode(value).byteLength;
export const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const sameKeys = (value, keys) => isObject(value) && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
const unicode = value => !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value);
// Python str.strip also recognizes the four information separators and U+0085.
const trimID = value => value.replace(/^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/gu, '');

export function assertContract(document, model) {
  const schema = document?.input;
  const fail = () => { throw new Error('Unsupported model or input contract. Reconnect to a compatible research API.'); };
  if (!isObject(schema) || !isObject(document.limits) || model?.purpose !== 'research' || model.production_ready !== false || model.research_only !== true || schema.inference_contract_version !== '2.0' || document.output?.inference_contract_version !== '2.0') fail();
  for (const key of ['features', 'numeric_columns', 'categorical_columns', 'non_negative_columns', 'labels']) {
    if (!Array.isArray(schema[key]) || !schema[key].every(v => typeof v === 'string') || new Set(schema[key]).size !== schema[key].length) fail();
  }
  const roles = [...schema.numeric_columns, ...schema.categorical_columns];
  if (!schema.features.length || roles.length !== schema.features.length || new Set(roles).size !== roles.length || roles.some(k => !schema.features.includes(k)) || schema.features.includes('content_id') || schema.non_negative_columns.some(k => !schema.numeric_columns.includes(k)) || schema.labels.length !== 5 || ['down', 'stable', 'up', 'new', 'flat'].some(k => !schema.labels.includes(k))) fail();
  if (schema.id_column !== 'content_id' || schema.extra_columns !== 'reject' || schema.missing_columns !== 'reject' || schema.id_policy !== 'unique_nonempty_strings_no_surrounding_whitespace') fail();
  for (const key of ['maximum_batch_rows', 'maximum_request_bytes', 'maximum_response_bytes', 'maximum_content_id_bytes', 'maximum_category_bytes']) {
    if (!Number.isSafeInteger(document.limits[key]) || document.limits[key] <= 0 || model.limits?.[key] !== document.limits[key]) fail();
  }
  for (const key of ['package_id', 'model_version', 'package_manifest_sha256', 'family']) if (typeof model[key] !== 'string') fail();
  if (!Array.isArray(model.limitations) || !model.limitations.every(v => typeof v === 'string')) fail();
}

export async function validateRequest(document, contract, alive = () => true) {
  const {input: schema, limits} = contract;
  const issues = [];
  let count = 0;
  const add = (message, row = null, field = null) => { count++; if (issues.length < 100) issues.push({message, row, field}); };
  if (!isObject(document) || Object.keys(document).some(k => !['purpose', 'records', 'include_probabilities'].includes(k)) || document.purpose !== 'research' || (Object.hasOwn(document, 'include_probabilities') && typeof document.include_probabilities !== 'boolean')) {
    add('Use a research request envelope with only purpose, records, and an optional boolean include_probabilities.');
    return {issues, count};
  }
  if (!Array.isArray(document.records) || document.records.length < 1 || document.records.length > limits.maximum_batch_rows) {
    add(`Provide between 1 and ${limits.maximum_batch_rows.toLocaleString()} records.`);
    return {issues, count};
  }
  const expected = ['content_id', ...schema.features];
  const ids = new Set();
  for (let index = 0; index < document.records.length; index++) {
    if (index % 500 === 0) { await new Promise(resolve => setTimeout(resolve, 0)); if (!alive()) throw new DOMException('Cancelled', 'AbortError'); }
    const row = document.records[index];
    if (!sameKeys(row, expected)) { add('Every record must have exactly the required fields; missing or extra fields were found.', index); continue; }
    const id = row.content_id;
    if (typeof id !== 'string' || !id || id !== trimID(id) || !unicode(id) || bytes(id) > limits.maximum_content_id_bytes) add(`Use a nonempty ID without surrounding whitespace, at most ${limits.maximum_content_id_bytes} UTF-8 bytes.`, index, 'content_id');
    else if (ids.has(id)) add('Content IDs must be unique within this batch.', index, 'content_id');
    else ids.add(id);
    for (const key of schema.numeric_columns) {
      const v = row[key];
      if (v !== null && (typeof v !== 'number' || !Number.isFinite(v) || (schema.non_negative_columns.includes(key) && v < 0))) add('Enter a finite number within the field bounds, or select Missing.', index, key);
    }
    for (const key of schema.categorical_columns) {
      const v = row[key];
      if (v !== null && (typeof v !== 'string' || !unicode(v) || bytes(v) > limits.maximum_category_bytes)) add(`Use text of at most ${limits.maximum_category_bytes} UTF-8 bytes, or select Missing.`, index, key);
    }
  }
  if (count) return {issues, count};
  const body = JSON.stringify(document);
  const size = bytes(body);
  if (size > limits.maximum_request_bytes) add(`Request exceeds the ${limits.maximum_request_bytes.toLocaleString()} byte limit. Reduce the batch.`);
  return {issues, count, body, size};
}

export function assertResponse(result, request, contract, model) {
  const scores = request.include_probabilities ?? false;
  const fail = () => { throw new Error('The API returned an unexpected prediction contract. Results were not displayed. Reconnect before trying again.'); };
  const fixed = {inference_contract_version: '2.0', package_schema_version: '1.0', package_id: model.package_id, package_manifest_sha256: model.package_manifest_sha256, model_version: model.model_version, purpose: 'research', production_ready: false, row_count: request.records.length, include_probabilities: scores, probability_interpretation: scores ? 'uncalibrated_noncausal' : 'not_requested'};
  if (!sameKeys(result, [...Object.keys(fixed), 'records']) || Object.entries(fixed).some(([k, v]) => result[k] !== v) || !Array.isArray(result.records) || result.records.length !== request.records.length) fail();
  const labels = contract.input.labels;
  for (let i = 0; i < result.records.length; i++) {
    const row = result.records[i];
    if (!sameKeys(row, ['content_id', 'predicted_trend', 'probability_down', 'probabilities', 'model_version', 'warning']) || row.content_id !== request.records[i].content_id || row.model_version !== model.model_version || !labels.includes(row.predicted_trend) || row.warning !== 'Human review required; do not automate content changes.') fail();
    if (!scores) { if (row.probabilities !== null || row.probability_down !== null) fail(); }
    else if (!sameKeys(row.probabilities, labels) || labels.some(k => typeof row.probabilities[k] !== 'number' || !Number.isFinite(row.probabilities[k]) || row.probabilities[k] < 0 || row.probabilities[k] > 1) || row.probability_down !== row.probabilities.down || Math.abs(labels.reduce((sum, k) => sum + row.probabilities[k], 0) - 1) > 0.000001) fail();
  }
}
