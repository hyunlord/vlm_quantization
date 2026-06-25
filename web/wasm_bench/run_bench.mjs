// Drives bench.html in real headless Chromium (Playwright Chrome-for-Testing).
// Sweeps CPU-throttle rates via CDP to test verdict robustness on slower /
// mobile-class compute. Throttling slows the SAME real V8+WASM engine — it is
// NOT an emulator and NOT a different browser; it is honestly labelled as
// "emulated slower CPU", not a real mobile device.
// Usage: node run_bench.mjs [--headed] [--rates=1,4,6]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MIME = { '.html': 'text/html', '.wasm': 'application/wasm', '.js': 'text/javascript', '.mjs': 'text/javascript' };

const ratesArg = process.argv.find(a => a.startsWith('--rates='));
const RATES = ratesArg ? ratesArg.split('=')[1].split(',').map(Number) : [1, 4, 6];
const headed = process.argv.includes('--headed');

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
const url = `http://localhost:${port}/bench.html`;

const browser = await chromium.launch({ headless: !headed });
const all = [];          // flattened result rows with cpu_throttle
let env = null, checks = null;

for (const rate of RATES) {
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error') console.error('[page error]', m.text()); });
  const client = await page.context().newCDPSession(page);
  if (rate > 1) await client.send('Emulation.setCPUThrottlingRate', { rate });
  console.error(`navigating ${url} (cpu_throttle=${rate}x)`);
  await page.goto(url);
  await page.evaluate('window.runBench()');
  await page.waitForFunction('window.__RESULTS__ !== undefined', null, { timeout: 900000 });
  const payload = await page.evaluate('window.__RESULTS__');
  await page.close();
  if (payload.error) { console.error('BENCH ERROR:', payload.error); await browser.close(); server.close(); process.exit(1); }
  if (!env) { env = payload.env; env.browser = 'Playwright Chromium (Chrome for Testing)'; checks = payload.checks; }
  for (const r of payload.results) all.push({ ...r, cpu_throttle: rate });
}
await browser.close();
server.close();

// write raw json
fs.writeFileSync(path.join(__dirname, 'bench_results.json'), JSON.stringify({ env, checks, results: all }, null, 2));

// write CSV to paper/wasm_latency.csv
const repoRoot = path.resolve(__dirname, '..', '..');
const csvPath = path.join(repoRoot, 'paper', 'wasm_latency.csv');
const lines = ['gallery,method,impl,bytes_per_item,cpu_throttle,p50_ms,p95_ms,mean_ms,n_queries'];
for (const r of all) {
  lines.push([r.gallery, r.method, r.impl, r.bytes_per_item, r.cpu_throttle, r.p50_ms, r.p95_ms, r.mean_ms, r.n_queries].join(','));
}
fs.writeFileSync(csvPath, lines.join('\n') + '\n');

console.log('\n=== ENV ===');
console.log(JSON.stringify(env, null, 2));
console.log('\n=== CORRECTNESS ===');
console.log(JSON.stringify(checks, null, 2));
console.log('\n=== RESULTS (p50 ms, by cpu_throttle) ===');
console.table(all.map(r => ({
  cpu: r.cpu_throttle + 'x', gallery: r.gallery, method: r.method, impl: r.impl, B: r.bytes_per_item, p50: r.p50_ms, p95: r.p95_ms,
})));
console.log('\nCSV ->', csvPath);
console.log('JSON ->', path.join(__dirname, 'bench_results.json'));
