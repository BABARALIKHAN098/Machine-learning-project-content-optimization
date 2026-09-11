export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
const pretty = key => key.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
const groups = [
  ['Content identity', ['content_id']],
  ['Search metrics', ['search_volume', 'competition', 'cpc']],
  ['Content size', ['word_count', 'char_count']],
  ['Traffic · previous 30 days', ['impressions_prev_30d', 'clicks_prev_30d', 'sessions_prev_30d']],
  ['Age & freshness', ['content_age_days', 'age_tier_order', 'days_since_last_update']],
  ['Content categories', ['competition_level', 'content_type', 'main_intent', 'model_used', 'age_tier', 'freshness_tier', 'word_count_tier', 'char_count_tier']],
];
export class SchemaForm {
  constructor(root) { this.root = root; this.fields = new Map(); }
  render(schema) {
    this.fields.clear(); this.root.replaceChildren();
    const all = ['content_id', ...schema.features];
    const known = new Set(groups.flatMap(([, keys]) => keys));
    const displayGroups = [...groups, ['Additional fields', all.filter(k => !known.has(k))]];
    let groupIndex = 0;
    for (const [title, keys] of displayGroups) {
      const included = keys.filter(k => all.includes(k));
      if (!included.length) continue;
      const group = element('fieldset', undefined, 'feature-group');
      const legend = element('legend');
      legend.append(element('span', String(++groupIndex).padStart(2, '0'), 'group-index'), document.createTextNode(title));
      const grid = element('div', undefined, 'field-grid');
      for (const key of included) {
        const wrapper = element('div', undefined, 'field');
        const heading = element('div', undefined, 'field-heading');
        const input = element('input');
        const index = this.fields.size;
        input.id = `feature-${index}`;
        input.autocomplete = 'off';
        input.spellcheck = false;
        const numeric = schema.numeric_columns.includes(key);
        input.type = numeric ? 'number' : 'text';
        if (numeric) { input.step = 'any'; if (schema.non_negative_columns.includes(key)) input.min = '0'; }
        const label = element('label', key === 'content_id' ? 'Content ID' : pretty(key));
        label.htmlFor = input.id;
        heading.append(label);
        let missing = null;
        if (key !== 'content_id') {
          const missingLabel = element('label', undefined, 'missing-control');
          missing = element('input'); missing.type = 'checkbox'; missing.checked = true;
          missing.setAttribute('aria-label', `${pretty(key)} missing`);
          missing.addEventListener('change', () => { input.disabled = missing.checked; });
          missingLabel.append(missing, document.createTextNode('Missing')); heading.append(missingLabel);
          input.disabled = true;
        } else input.placeholder = 'e.g. content-001';
        const hint = element('small', key, 'field-key'); hint.id = `hint-${index}`;
        input.setAttribute('aria-describedby', hint.id);
        wrapper.append(heading, input, hint); grid.append(wrapper);
        this.fields.set(key, {input, missing, numeric});
      }
      group.append(legend, grid); this.root.append(group);
    }
  }
  read() {
    return Object.fromEntries([...this.fields].map(([key, {input, missing, numeric}]) => [key, missing?.checked ? null : numeric ? input.value === '' ? NaN : input.valueAsNumber : input.value]));
  }
  fill(record) {
    for (const [key, {input, missing}] of this.fields) {
      const value = record[key];
      if (missing) { missing.checked = value === null; input.disabled = missing.checked; }
      input.value = value === null ? '' : String(value);
    }
  }
  clearErrors() { for (const {input} of this.fields.values()) input.removeAttribute('aria-invalid'); }
  mark(field) { const input = this.fields.get(field)?.input; if (input) input.setAttribute('aria-invalid', 'true'); return input?.id; }
}
