export function downloadResults(result) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(result)], {type: 'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = 'content-trend-results.json';
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
