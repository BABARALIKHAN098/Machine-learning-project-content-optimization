const token = document.getElementById('token');
const payload = document.getElementById('payload');
const result = document.getElementById('result');
const buttons = [...document.querySelectorAll('button')];
async function request(path, body) {
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST',
    cache: 'no-store', headers: {Authorization: 'Bearer ' + token.value, 'Content-Type': 'application/json'}, body});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || 'Request failed.');
  return value;
}
async function action(work) {
  buttons.forEach(b => b.disabled = true);
  try { await work(); } catch (error) { result.textContent = error.message; }
  finally { buttons.forEach(b => b.disabled = false); }
}
document.getElementById('load').onclick = () => action(async () => {
  const schema = await request('/v1/schema');
  const model = await request('/v1/model');
  payload.value = JSON.stringify(schema.example, null, 2);
  document.getElementById('model').textContent = model.package_id + '\nResearch only · Maximum rows: ' + schema.limits.maximum_batch_rows;
  result.textContent = 'Invented example loaded. Ready to predict.';
});
document.getElementById('send').onclick = () => action(async () => {
  result.textContent = 'Predicting…';
  result.textContent = JSON.stringify(await request('/v1/predictions', payload.value), null, 2);
});
document.getElementById('clear').onclick = () => {
  token.value = ''; payload.value = ''; result.textContent = 'Private fields cleared.';
  document.getElementById('model').textContent = '';
};
