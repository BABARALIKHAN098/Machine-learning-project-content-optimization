export class APIError extends Error {
  constructor(message, status = 0, code = '', requestId = '', retry = 0) {
    super(message); Object.assign(this, {status, code, requestId, retry});
  }
}
const guidance = {
  unauthorized: 'Enter a valid API token and reconnect.',
  request_body_timeout: 'The upload timed out. Check the service before an explicit retry.',
  request_too_large: 'Reduce the batch to fit the active request limit.',
  unsupported_media_type: 'The API requires uncompressed UTF-8 JSON.',
  prediction_capacity_exceeded: 'The model is busy, possibly in another tab. Wait before retrying.',
  prediction_failed: 'Prediction failed. Reconnect to check readiness; the local service may need restarting.',
  response_limit_exceeded: 'The result exceeded the response limit. The service may need restarting; use a smaller batch or omit scores.',
  service_not_ready: 'The research service is not ready. Check the local service, then reconnect.',
  invalid_host: 'Open this workspace at the configured local API address.',
  invalid_origin: 'Open this workspace on the same local origin as the API.',
  invalid_json: 'The API rejected the JSON. Check syntax, duplicate keys, and encoding.',
  invalid_request: 'The request does not match the model contract. Validate its required fields, types, and limits.',
};
export async function apiRequest(path, {token = '', body, signal, maxBytes} = {}) {
  let response;
  try {
    response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', cache: 'no-store', credentials: 'omit', redirect: 'error', signal, headers: {...(token ? {Authorization: `Bearer ${token}`} : {}), ...(body === undefined ? {} : {'Content-Type': 'application/json'})}, body});
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new APIError(body === undefined ? 'Cannot reach the local API. Check that it is running, then reconnect.' : 'Connection interrupted. The prediction may still be running; reconnect before an explicit retry.');
  }
  const requestId = response.headers.get('X-Request-ID') || '';
  if (!response.headers.get('content-type')?.includes('application/json')) throw new APIError('The server returned a non-JSON response.', response.status, '', requestId);
  let value;
  try {
    if (maxBytes && response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8', {fatal: true});
      let size = 0, text = '';
      try {
        while (true) {
          const {done, value: chunk} = await reader.read();
          if (done) break;
          size += chunk.length;
          if (size > maxBytes) { await reader.cancel(); throw new Error('Response limit'); }
          text += decoder.decode(chunk, {stream: true});
        }
        text += decoder.decode();
        value = JSON.parse(text);
      } finally { reader.releaseLock(); }
    } else value = await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new APIError('The server response could not be read safely. Reconnect before retrying.', response.status, '', requestId);
  }
  if (!response.ok) {
    const code = typeof value.error?.code === 'string' ? value.error.code : '';
    const retryHeader = response.headers.get('Retry-After');
    const retry = retryHeader && /^\d+$/.test(retryHeader) ? Number(retryHeader) * 1000 : Math.max(0, Date.parse(retryHeader || '') - Date.now()) || 1000;
    throw new APIError(guidance[code] || `The request failed (HTTP ${response.status}). Check the local API.`, response.status, code, requestId, retry);
  }
  if (response.headers.get('X-API-Contract-Version') !== '1.0') throw new APIError('Unsupported API contract version.', response.status, '', requestId);
  return value;
}
