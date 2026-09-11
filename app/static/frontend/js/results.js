import {element} from './schema-form.js';
import {downloadResults} from './download.js';
const $ = id => document.getElementById(id);
const symbols = {down: '↓', stable: '→', up: '↑', new: '✧', flat: '−'};
const label = value => value[0].toUpperCase() + value.slice(1);
export function summarize(records, labels) {
  const counts = Object.fromEntries(labels.map(key => [key, 0]));
  for (const row of records) counts[row.predicted_trend]++;
  return counts;
}
export function filterRecords(records, query, trend) {
  const needle = query.toLowerCase();
  return records.map((record, index) => ({record, index})).filter(({record}) => (!trend || record.predicted_trend === trend) && record.content_id.toLowerCase().includes(needle));
}
export class Results {
  constructor() {
    this.result = null; this.labels = []; this.page = 0;
    for (const id of ['result-search', 'result-filter', 'page-size']) $(id).addEventListener('input', () => { this.page = 0; this.renderRows(); });
    $('clear-filters').addEventListener('click', () => { $('result-search').value = ''; $('result-filter').value = ''; this.page = 0; this.renderRows(); });
    $('result-prev').addEventListener('click', () => { this.page--; this.renderRows(); });
    $('result-next').addEventListener('click', () => { this.page++; this.renderRows(); });
    $('download').addEventListener('click', () => { if (this.result) downloadResults(this.result); });
  }
  set(result, labels) {
    this.result = result; this.labels = [...labels]; this.page = 0;
    $('results-content').hidden = false; $('results-empty').hidden = true; $('download').disabled = false;
    $('result-search').value = ''; $('result-filter').replaceChildren(new Option('All trends', ''));
    for (const key of labels) $('result-filter').append(new Option(label(key), key));
    const counts = {total: result.row_count, ...summarize(result.records, labels)};
    $('result-counts').replaceChildren(...Object.entries(counts).map(([key, count]) => {
      const card = element('div', undefined, 'count-card'); card.append(element('span', key === 'total' ? 'Total records' : `${symbols[key]} ${label(key)}`), element('strong', count.toLocaleString())); return card;
    }));
    $('result-context').textContent = `${result.row_count.toLocaleString()} records · ${result.model_version} · ${result.include_probabilities ? 'Uncalibrated scores included' : 'Scores not requested'} · Download includes the complete run.`;
    this.renderRows(); $('results-title').focus();
  }
  stale() { if (this.result) $('result-context').textContent = `Previous run · ${this.result.row_count.toLocaleString()} records · ${this.result.model_version}. Inputs or connection have changed. Download still includes that complete run.`; }
  renderRows() {
    if (!this.result) return;
    const filtered = filterRecords(this.result.records, $('result-search').value, $('result-filter').value);
    const size = Number($('page-size').value);
    const pages = Math.max(1, Math.ceil(filtered.length / size));
    this.page = Math.max(0, Math.min(this.page, pages - 1));
    const fragment = document.createDocumentFragment();
    for (const {record, index} of filtered.slice(this.page * size, (this.page + 1) * size)) {
      const tr = element('tr');
      const id = element('td', record.content_id, 'id-cell'); id.title = record.content_id;
      const trend = element('td'); trend.append(element('span', `${symbols[record.predicted_trend]} ${label(record.predicted_trend)}`, `badge ${record.predicted_trend}`));
      const action = element('td'); const button = element('button', 'Details', 'text-button'); button.type = 'button'; button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-label', `Details for row ${index + 1}`); action.append(button);
      tr.append(element('td', String(index + 1)), id, trend, element('td', record.probability_down === null ? 'Not requested' : `${(record.probability_down * 100).toFixed(1)}%`), element('td', record.model_version), action);
      let detail = null;
      button.addEventListener('click', () => {
        if (detail) { detail.remove(); detail = null; button.setAttribute('aria-expanded', 'false'); button.textContent = 'Details'; return; }
        detail = element('tr'); detail.id = `details-${index}`;
        const cell = element('td', undefined, 'details-cell'); cell.colSpan = 6;
        cell.append(element('p', record.content_id), element('p', record.warning));
        if (record.probabilities) {
          cell.append(element('p', 'Uncalibrated, non-causal model scores.'));
          const scores = element('div', undefined, 'score-list');
          for (const key of this.labels) scores.append(element('span', `${label(key)}: ${(record.probabilities[key] * 100).toFixed(1)}%`));
          cell.append(scores);
        } else cell.append(element('p', 'Model scores were not requested.'));
        detail.append(cell); tr.after(detail); button.setAttribute('aria-expanded', 'true'); button.setAttribute('aria-controls', detail.id); button.textContent = 'Close';
      });
      fragment.append(tr);
    }
    $('result-rows').replaceChildren(fragment);
    $('no-matches').hidden = filtered.length !== 0;
    $('result-page').textContent = `${filtered.length.toLocaleString()} matching records · Page ${this.page + 1} of ${pages}`;
    $('result-prev').disabled = this.page === 0; $('result-next').disabled = this.page + 1 === pages;
  }
  clear() {
    this.result = null; this.labels = []; this.page = 0;
    $('results-content').hidden = true; $('results-empty').hidden = false; $('download').disabled = true;
    $('result-search').value = ''; $('result-filter').replaceChildren(new Option('All trends', '')); $('page-size').value = '50';
    $('result-counts').replaceChildren(); $('result-rows').replaceChildren(); $('result-page').textContent = '';
    $('result-context').textContent = 'Your predictions will appear here after a successful run.';
  }
}
