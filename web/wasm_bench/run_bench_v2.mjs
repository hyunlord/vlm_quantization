// Drives bench_v2.html in real headless Chromium. Sweeps optional CPU-throttle
// (mobile-class compute proxy on the same real V8+WASM engine — labelled, not a
// real device). Usage: node run_bench_v2.mjs [--headed] [--rates=1,4]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MIME = { '.html':'text/html', '.wasm':'application/wasm', '.js':'text/javascript', '.mjs':'text/javascript' };
const ratesArg = process.argv.find(a => a.startsWith('--rates='));
const RATES = ratesArg ? ratesArg.split('=')[1].split(',').map(Number) : [1];
const headed = process.argv.includes('--headed');

const server = http.createServer((req,res)=>{
  let p=decodeURIComponent(req.url.split('?')[0]); if(p==='/') p='/bench_v2.html';
  const fp=path.join(__dirname,p);
  if(!fp.startsWith(__dirname)||!fs.existsSync(fp)){ res.writeHead(404); res.end('nf'); return; }
  res.writeHead(200,{'content-type':MIME[path.extname(fp)]||'application/octet-stream'});
  fs.createReadStream(fp).pipe(res);
});
await new Promise(r=>server.listen(0,r));
const port=server.address().port;
const url=`http://localhost:${port}/bench_v2.html`;

const browser=await chromium.launch({ headless:!headed });
const all=[]; let env=null, checks=null;
for(const rate of RATES){
  const page=await browser.newPage();
  page.on('console',m=>{ if(m.type()==='error') console.error('[page error]',m.text()); });
  const client=await page.context().newCDPSession(page);
  if(rate>1) await client.send('Emulation.setCPUThrottlingRate',{ rate });
  console.error(`navigating ${url} (cpu_throttle=${rate}x)`);
  await page.goto(url);
  await page.evaluate('window.runBench()');
  await page.waitForFunction('window.__RESULTS__ !== undefined', null, { timeout: 1800000 });
  const payload=await page.evaluate('window.__RESULTS__');
  await page.close();
  if(payload.error){ console.error('BENCH ERROR:',payload.error); await browser.close(); server.close(); process.exit(1); }
  if(!env){ env=payload.env; env.browser='Playwright Chromium (Chrome for Testing)'; checks=payload.checks; }
  for(const r of payload.results) all.push({ ...r, cpu_throttle:rate });
}
await browser.close(); server.close();

fs.writeFileSync(path.join(__dirname,'bench_results_v2.json'), JSON.stringify({ env, checks, results:all }, null, 2));
const repoRoot=path.resolve(__dirname,'..','..');
const csvPath=path.join(repoRoot,'paper','wasm_latency_v2.csv');
const cols=['experiment','gallery','method','impl','phase','bytes_per_item','cpu_throttle','p50_ms','p95_ms','ns_per_item','ns_per_byte','inner_reps','n_queries'];
const lines=[cols.join(',')];
for(const r of all) lines.push(cols.map(c=>r[c]).join(','));
fs.writeFileSync(csvPath, lines.join('\n')+'\n');

console.log('\n=== ENV ==='); console.log(JSON.stringify(env,null,2));
console.log('\n=== CORRECTNESS ==='); console.log(JSON.stringify(checks,null,2));
// scan-only main grid: binary vs int8 ns/item
console.log('\n=== MAIN scan-only ns/item (wasm_simd) binary vs int8 ===');
const rows=all.filter(r=>r.experiment==='main'&&r.impl==='wasm_simd'&&r.phase==='scan'&&(r.method==='binary'||r.method==='int8')&&r.cpu_throttle===RATES[0]);
const gs=[...new Set(rows.map(r=>r.gallery))].sort((a,b)=>a-b);
console.table(gs.map(g=>{ const b=rows.find(r=>r.gallery===g&&r.method==='binary'),i=rows.find(r=>r.gallery===g&&r.method==='int8');
  return { gallery:g, binary_ns:b?.ns_per_item, int8_ns:i?.ns_per_item, 'bin/int8':b&&i?(b.ns_per_item/i.ns_per_item).toFixed(2):'' }; }));
console.log('\nCSV ->',csvPath);
