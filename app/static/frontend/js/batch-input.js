import {parseStrictJSON} from './strict-json.js';
import {bytes} from './validation.js';
import {element} from './schema-form.js';

export class BatchInput {
  constructor() {
    this.text = document.getElementById('batch-text');
    this.file = document.getElementById('batch-file');
    this.raw = ''; this.preview = null; this.page = 0; this.importID = 0;
    document.getElementById('preview-prev').addEventListener('click', () => { this.page--; this.renderPreview(); });
    document.getElementById('preview-next').addEventListener('click', () => { this.page++; this.renderPreview(); });
    this.text.addEventListener('input', () => { this.importID++; this.raw = ''; this.file.value = ''; this.preview = null; document.getElementById('batch-preview').hidden = true; document.getElementById('file-status').textContent = 'Pasted JSON · file cleared'; });
  }
  async loadFile(limit, alive) {
    const file = this.file.files[0];
    const id = ++this.importID;
    this.raw = ''; this.text.value = ''; this.preview = null;
    document.getElementById('batch-preview').hidden = true;
    if (!file) return;
    if (file.size > limit) throw new Error(`The file exceeds the ${limit.toLocaleString()} byte request limit.`);
    let raw;
    try { raw = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(await file.arrayBuffer()); }
    catch { throw new Error('The file must contain valid UTF-8 JSON.'); }
    if (!alive() || id !== this.importID) throw new DOMException('Cancelled', 'AbortError');
    const parsed = parseStrictJSON(raw);
    this.raw = raw;
    if (raw.length < 100000) { this.text.value = raw; this.raw = ''; }
    document.getElementById('file-status').textContent = `JSON loaded · ${(file.size / 1024).toFixed(1)} KiB${this.raw ? ' · large file kept out of the editor' : ''}`;
    return parsed;
  }
  read(limit) {
    const raw = this.raw || this.text.value;
    if (bytes(raw) > limit) throw new Error(`The JSON exceeds the ${limit.toLocaleString()} byte input limit.`);
    return parseStrictJSON(raw);
  }
  set(document) { this.clear(); this.text.value = JSON.stringify(document, null, 2); }
  showPreview(document) { this.preview = document; this.page = 0; this.renderPreview(); }
  renderPreview() {
    const rows = this.preview?.records;
    if (!rows?.length) return;
    this.page = Math.max(0, Math.min(this.page, Math.ceil(rows.length / 50) - 1));
    document.getElementById('batch-preview').hidden = false;
    const keys = Object.keys(rows[0]);
    const header = element('tr');
    for (const key of keys) { const th = element('th', key); th.scope = 'col'; header.append(th); }
    document.getElementById('preview-head').replaceChildren(header);
    const fragment = document.createDocumentFragment();
    for (const record of rows.slice(this.page * 50, (this.page + 1) * 50)) {
      const row = element('tr');
      for (const key of keys) { const cell = element('td', record[key] === null ? 'Missing' : String(record[key]), 'id-cell'); cell.title = cell.textContent; row.append(cell); }
      fragment.append(row);
    }
    document.getElementById('preview-body').replaceChildren(fragment);
    document.getElementById('preview-count').textContent = `${rows.length.toLocaleString()} records`;
    document.getElementById('preview-page').textContent = `Page ${this.page + 1} of ${Math.ceil(rows.length / 50)}`;
    document.getElementById('preview-prev').disabled = this.page === 0;
    document.getElementById('preview-next').disabled = (this.page + 1) * 50 >= rows.length;
  }
  clear() {
    this.importID++; this.raw = ''; this.preview = null; this.page = 0; this.text.value = ''; this.file.value = '';
    document.getElementById('batch-preview').hidden = true;
    document.getElementById('preview-head').replaceChildren(); document.getElementById('preview-body').replaceChildren();
    document.getElementById('file-status').textContent = 'No file selected · JSON only';
  }
}
