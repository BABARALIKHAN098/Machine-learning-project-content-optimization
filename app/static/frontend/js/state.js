export function createSession() {
  return {
    generation: 0, controller: new AbortController(), token: '', contract: null, model: null,
    ready: false, busy: false, retryUntil: 0,
    invalidate() { this.controller.abort(); this.controller = new AbortController(); return ++this.generation; },
    clear() { this.invalidate(); this.token = ''; this.contract = null; this.model = null; this.ready = false; this.busy = false; this.retryUntil = 0; },
  };
}
