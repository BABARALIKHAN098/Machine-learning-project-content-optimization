// Integration runner: installed Playwright + Chrome, local frozen packages, invented inputs only.
// Set PLAYWRIGHT_MODULE to an existing module path when Playwright is not on Node's module path.
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
import {randomBytes} from 'node:crypto';
import {mkdir, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const python = process.env.FRONTEND_TEST_PYTHON || resolve('.venv/Scripts/python.exe');
const output = resolve('.frontend-work');
await mkdir(output, {recursive: true});
const families = [
  ['random_forest', '17d3d181eea90223de0f5ea59edd750d495b1efd32f0a8cafaae45c10d4279e6'],
  ['logistic_regression', '9bd1ae7043ea49339bbc21f8311e9b1e296043b83db8f90c25e220a0741e658a'],
];
const uiOnly = process.argv.includes('--ui-only');
const evidence = {checks: [], browsers: [], models: [], screenshots: []};
const note = message => { evidence.checks.push(message); console.log(`PASS ${message}`); };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let browser, server;
async function stopServer() {
  if (!server || server.exitCode !== null) return;
  const exited = new Promise(resolve => server.once('exit', resolve));
  server.kill(); await Promise.race([exited, delay(5000)]);
}
try {
  browser = await chromium.launch({channel: 'chrome', headless: true});
  evidence.browsers.push(await browser.version());
  for (const [family, pin] of (uiOnly ? families.slice(0, 1) : families)) {
    const port = 8127;
    const origin = `http://127.0.0.1:${port}`;
    const token = randomBytes(32).toString('hex');
    let serverError = '';
    server = spawn(python, ['scripts/serve_api.py', '--purpose', 'research', '--package-dir', `artifacts/packages/spec08-final-reference-${family}`, '--expected-manifest-sha256', pin, '--port', String(port)], {cwd: process.cwd(), env: {...process.env, CONTENT_TREND_API_TOKEN: token}, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe']});
    server.stderr.on('data', data => { serverError = (serverError + data.toString()).slice(-5000); });
    server.stdout.on('data', () => {});
    let ready = false;
    for (let i = 0; i < 120; i++) {
      if (server.exitCode !== null) throw new Error(`API startup failed: ${serverError}`);
      try { const response = await fetch(`${origin}/ready`); if (response.ok) { ready = true; break; } } catch { /* Wait for startup. */ }
      await delay(500);
    }
    assert.ok(ready, `Local API did not become ready: ${serverError}`);
    const context = await browser.newContext({viewport: {width: 1440, height: 1050}, acceptDownloads: true});
    const page = await context.newPage();
    const pageErrors = []; const requests = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && !/Failed to load resource:.*status of (401|429|503)/.test(message.text())) pageErrors.push(message.text());
    });
    page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/v1/predictions')) requests.push(JSON.parse(request.postData())); });
    await page.goto(origin); await page.waitForLoadState('networkidle');
    assert.equal(await page.title(), 'Content Trend \u00b7 Research workspace');
    assert.ok(await page.locator('#input-controls').evaluate(node => node.disabled));
    await page.screenshot({path: `${output}/desktop-empty.png`, fullPage: true});
    evidence.screenshots.push('desktop-empty.png');
    note(`${family}: page and local assets load, inputs gated before connection`);

    await page.locator('#token').fill('invalid-synthetic-token-00000000000');
    await page.locator('#connect').click();
    await page.locator('#error-summary').waitFor({state: 'visible'});
    assert.match(await page.locator('#error-message').innerText(), /valid API token/);
    assert.equal(await page.locator('#token').inputValue(), '');
    await page.locator('#token').fill(token); await page.locator('#connect').click();
    await page.waitForFunction(() => document.querySelector('#connection-label').textContent === 'Connected');
    assert.equal(await page.locator('#schema-fields .field').count(), 20);
    assert.equal(await page.locator('#token').inputValue(), '');
    note(`${family}: invalid token rejected, valid token loads all 19 features`);

    if (uiOnly) {
      await page.getByLabel('Content ID', {exact: true}).fill(' trailing ');
      await page.locator('#validate').click(); await page.locator('#error-summary').waitFor({state: 'visible'});
      assert.ok(await page.locator('#error-summary').evaluate(node => node === document.activeElement));
      assert.equal(await page.getByLabel('Content ID', {exact: true}).getAttribute('aria-invalid'), 'true');
      await page.locator('#error-list a').first().click();
      assert.ok(await page.getByLabel('Content ID', {exact: true}).evaluate(node => node === document.activeElement));
      await page.getByLabel('Content ID', {exact: true}).fill('keyboard-synthetic');
      await page.getByRole('checkbox', {name: 'Search volume missing', exact: true}).uncheck();
      await page.getByLabel('Search volume', {exact: true}).fill('-1');
      await page.locator('#validate').click(); await page.locator('#error-summary').waitFor({state: 'visible'});
      assert.equal(await page.getByLabel('Search volume', {exact: true}).getAttribute('aria-invalid'), 'true');
      await page.getByLabel('Search volume', {exact: true}).fill('0');
      await page.locator('#validate').click(); await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Inputs are valid'));
      await page.locator('#predict').click(); await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Completed'), null, {timeout: 60000});
      assert.equal(requests.at(-1).records[0].search_volume, 0);
      assert.equal(requests.at(-1).records[0].competition, null);
      note('Field error links move focus; explicit Missing and numeric zero serialize correctly');
      await page.locator('input[name=input-mode][value=single]').focus(); await page.keyboard.press('ArrowRight');
      assert.ok(await page.locator('#batch-panel').isVisible());
      await page.keyboard.press('ArrowLeft'); assert.ok(await page.locator('#single-panel').isVisible());
      await page.emulateMedia({reducedMotion: 'reduce'});
      assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior), 'auto');
      await page.setViewportSize({width: 1280, height: 1000});
      await page.evaluate(() => { document.documentElement.style.zoom = '2'; });
      assert.ok(await page.evaluate(() => document.documentElement.getBoundingClientRect().width <= innerWidth + 1));
      await page.screenshot({path: `${output}/workspace-zoom-200.png`, fullPage: true});
      await page.evaluate(() => { document.documentElement.style.zoom = ''; });
      // Measure contrast from the actual design colors for normal text and primary buttons.
      const contrast = await page.evaluate(() => {
        const rgb = value => value.match(/\d+/g).slice(0, 3).map(Number).map(v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; });
        const luminance = value => rgb(value).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
        const ratio = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
        const button = getComputedStyle(document.querySelector('#predict'));
        return {button: ratio(button.color, button.backgroundColor), helper: ratio(getComputedStyle(document.querySelector('.helper')).color, 'rgb(255,255,255)')};
      });
      assert.ok(contrast.button >= 4.5 && contrast.helper >= 4.5);
      note('Native input-mode keyboard navigation, reduced motion, 200% zoom, and text contrast verified');
      assert.deepEqual(pageErrors, []);
      await context.close(); await stopServer(); server = null;
      continue;
    }

    await page.locator('#load-example').click(); await page.locator('#validate').click();
    await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Inputs are valid'));
    const submittedBefore = requests.length;
    await page.locator('#predict').click();
    await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Completed'), null, {timeout: 60000});
    assert.equal(requests.length, submittedBefore + 1);
    assert.equal(requests.at(-1).include_probabilities, false);
    assert.match(await page.locator('#result-rows').innerText(), /Not requested/);
    note(`${family}: single-content label-only prediction completes through real model`);

    await page.locator('#include-scores').check(); await page.locator('#predict').click();
    await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Completed'), null, {timeout: 60000});
    await page.getByRole('button', {name: 'Details for row 1', exact: true}).click();
    assert.equal(await page.locator('.score-list span').count(), 5);
    assert.match(await page.locator('.details-cell').innerText(), /non-causal/);
    const downloadPromise = page.waitForEvent('download'); await page.locator('#download').click();
    const download = await downloadPromise;
    const stream = await download.createReadStream(); const chunks = []; for await (const chunk of stream) chunks.push(chunk);
    const exported = JSON.parse(Buffer.concat(chunks).toString());
    assert.equal(exported.include_probabilities, true); assert.equal(exported.package_manifest_sha256, pin);
    assert.equal(exported.records[0].content_id, requests.at(-1).records[0].content_id);
    note(`${family}: five scores, provenance, and complete JSON download verified`);

    if (family === 'random_forest') {
      // Error and stale-response states are controlled at the browser boundary.
      await page.route('**/v1/predictions', route => route.fulfill({status: 429, contentType: 'application/json', headers: {'Retry-After': '1', 'X-API-Contract-Version': '1.0'}, body: JSON.stringify({error: {code: 'prediction_capacity_exceeded', request_id: 'synthetic-busy'}})}));
      await page.locator('#predict').click(); await page.locator('#error-summary').waitFor({state: 'visible'});
      assert.match(await page.locator('#error-message').innerText(), /busy/);
      assert.ok(await page.locator('#predict').isDisabled());
      await page.waitForFunction(() => !document.querySelector('#predict').disabled);
      await page.unroute('**/v1/predictions'); note('429 preserves inputs and delays manual retry');
      await page.route('**/v1/predictions', route => route.fulfill({status: 503, contentType: 'application/json', headers: {'X-API-Contract-Version': '1.0'}, body: JSON.stringify({error: {code: 'service_not_ready', request_id: 'synthetic-not-ready'}})}));
      await page.locator('#predict').click(); await page.waitForFunction(() => document.querySelector('#connection-label').textContent === 'Disconnected');
      assert.ok(await page.locator('#input-controls').evaluate(node => node.disabled));
      await page.unroute('**/v1/predictions');
      await page.locator('#token').fill(token); await page.locator('#connect').click(); await page.waitForFunction(() => document.querySelector('#connection-label').textContent === 'Connected');
      note('503 disables inference; explicit reconnection restores the session');
    }

    await page.locator('input[name=input-mode][value=batch]').check();
    await page.locator('#load-example').click();
    await page.locator('#batch-text').fill('{"purpose":"research","purpose":"research","records":[]}');
    await page.locator('#validate').click(); await page.locator('#error-summary').waitFor({state: 'visible'});
    assert.match(await page.locator('#error-message').innerText(), /strict JSON/);
    await page.locator('#load-example').click(); await page.locator('#validate').click();
    await page.locator('#batch-preview').waitFor({state: 'visible'});
    assert.equal(await page.locator('#preview-body tr').count(), 2);
    note(`${family}: batch syntax errors and two-record preview verified`);

    const schemaResponse = await fetch(`${origin}/v1/schema`, {headers: {Authorization: `Bearer ${token}`}});
    const schema = await schemaResponse.json();
    const synthetic = {purpose: 'research', include_probabilities: true, records: Array.from({length: 30000}, (_, i) => Object.fromEntries([['content_id', i === 0 ? '<img src=x onerror=alert(1)>' : `synthetic-${String(i).padStart(6, '0')}`], ...schema.input.numeric_columns.map(k => [k, i % 101]), ...schema.input.categorical_columns.map(k => [k, '__SYNTHETIC_UNKNOWN__'])]))};
    await page.locator('#batch-file').setInputFiles({name: 'invented-batch.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(synthetic))});
    await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('JSON file loaded'));
    assert.equal(await page.locator('#batch-text').inputValue(), '');
    const start = Date.now();
    await page.locator('#predict').click();
    await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Completed 30,000'), null, {timeout: 120000});
    evidence.models.push({family, largeBatchRows: 30000, probabilities: true, browserWorkflowMs: Date.now() - start});
    assert.equal(await page.locator('#result-rows > tr').count(), 50);
    assert.equal(await page.locator('#preview-body > tr').count(), 50);
    assert.equal(await page.locator('#result-rows img').count(), 0);
    await page.locator('#result-search').fill('synthetic-000100'); assert.equal(await page.locator('#result-rows > tr').count(), 1);
    await page.locator('#result-search').fill('does-not-exist'); assert.ok(await page.locator('#no-matches').isVisible());
    await page.locator('#clear-filters').click();
    await page.locator('#result-next').click(); assert.match(await page.locator('#result-page').innerText(), /Page 2/);
    note(`${family}: 30,000 real-model score results, bounded DOM, search, pagination, and inert ID text verified`);
    assert.deepEqual(await page.evaluate(() => ({local: localStorage.length, session: sessionStorage.length, cookies: document.cookie})), {local: 0, session: 0, cookies: ''});

    if (family === 'random_forest') {
      await page.locator('input[name=input-mode][value=single]').check();
      for (const width of [1440, 768, 390, 320]) {
        await page.setViewportSize({width, height: 1000});
        await page.evaluate(() => window.scrollTo(0, 0));
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Page overflow at ${width}px`);
        await page.screenshot({path: `${output}/workspace-${width}.png`, fullPage: true}); evidence.screenshots.push(`workspace-${width}.png`);
      }
      note('Desktop, tablet, mobile and 320px layouts have no page-wide overflow');
      await page.emulateMedia({reducedMotion: 'reduce'});
      await page.setViewportSize({width: 1440, height: 1000});
      await page.locator('#clear-session').click();
      assert.equal(await page.locator('#result-rows > tr').count(), 0);
      assert.equal(await page.locator('#batch-text').inputValue(), '');
      assert.equal(await page.locator('#token').inputValue(), '');
      await page.locator('#token').fill(token); await page.locator('#connect').click(); await page.waitForFunction(() => document.querySelector('#connection-label').textContent === 'Connected');
      await page.locator('#load-example').click();
      await page.route('**/v1/predictions', async route => { await delay(700); try { await route.fulfill({status: 200, contentType: 'application/json', headers: {'X-API-Contract-Version': '1.0'}, body: JSON.stringify(exported)}); } catch { /* Deliberately aborted request. */ } });
      await page.locator('#predict').click(); await page.waitForFunction(() => document.querySelector('#live-status').textContent.startsWith('Analyzing'));
      await page.locator('#clear-session').click(); await delay(1000);
      assert.ok(await page.locator('#results-content').isHidden()); assert.equal(await page.locator('#connection-label').innerText(), 'Disconnected');
      note('Clear session removes data and ignores a delayed prediction response');
    }
    assert.deepEqual(pageErrors, []);
    note(`${family}: no browser JavaScript errors or persistent session data`);
    await context.close(); await stopServer(); server = null;
  }
} finally {
  await browser?.close(); await stopServer();
  await writeFile(`${output}/${uiOnly ? 'accessibility' : 'verification'}.json`, JSON.stringify(evidence, null, 2));
}

