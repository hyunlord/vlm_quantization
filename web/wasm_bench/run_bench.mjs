// Drives bench.html in real headless Chromium (Playwright Chrome-for-Testing).
// Usage: node run_bench.mjs [--headed]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MIME = { '.html': 'text/html', '.wasm': 'application/wasm', '.js': 'text/javascript', '.mjs': 'text/javascript' };

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/bench.html';
  const fp = path.join(__dirname, p);
  if (!fp.startsWith(__dirname) || !fs.existsSync(fp)) { res.writeHead(404); res.end('not found'); return; }
  res.writeHead(200, { 'content-type': MIME[path.extname(fp)] || 'application/octet-stream' });
  fs.createReadStream(fp).pipe(res);
});

await new Promise(r => server.listen(0, r));
const port = server.address().port;
const url = `http://localhost:${port}/bench.html?auto`;

const headed = process.argv.includes('--headed');
const browser = await chromium.launch({ headless: !headed });
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error') console.error('[page error]', m.text()); });

console.error('navigating', url);
await page.goto(url);
await page.waitForFunction('window.__RESULTS__ !== undefined', null, { timeout: 600000 });
const payload = await page.evaluate('window.__RESULTS__');

await browser.close();
server.close();

if (payload.error) { console.error('BENCH ERROR:', payload.error); process.exit(1); }

// version info
const ver = browser.version ? '' : '';
payload.env.browser = 'Playwright Chromium (Chrome for Testing)';

// write raw json
fs.writeFileSync(path.join(__dirname, 'bench_results.json'), JSON.stringify(payload, null, 2));

// write CSV to paper/wasm_latency.csv
const repoRoot = path.resolve(__dirname, '..', '..');
const csvPath = path.join(repoRoot, 'paper', 'wasm_latency.csv');
const lines = ['gallery,method,impl,bytes_per_item,p50_ms,p95_ms,mean_ms,n_queries'];
for (const r of payload.results) {
  lines.push([r.gallery, r.method, r.impl, r.bytes_per_item, r.p50_ms, r.p95_ms, r.mean_ms, r.n_queries].join(','));
}
fs.writeFileSync(csvPath, lines.join('\n') + '\n');

console.log('\n=== ENV ===');
console.log(JSON.stringify(payload.env, null, 2));
console.log('\n=== CORRECTNESS ===');
console.log(JSON.stringify(payload.checks, null, 2));
console.log('\n=== RESULTS (p50/p95 ms) ===');
console.table(payload.results.map(r => ({
  gallery: r.gallery, method: r.method, impl: r.impl, B: r.bytes_per_item, p50: r.p50_ms, p95: r.p95_ms,
})));
console.log('\nCSV ->', csvPath);
console.log('JSON ->', path.join(__dirname, 'bench_results.json'));
