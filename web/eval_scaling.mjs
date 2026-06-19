/* D (JS half): single-query Hamming top-10 latency using the ACTUAL search.js hammingTopK,
 * at N given as argv[1] (comma list). Warm + median of 30. Prints {N: median_ms} JSON.
 * Driven by web/eval_scaling.py (sets SEARCH=/tmp/search_paper.mjs). Latency is
 * content-independent (full scan), so a cheap pseudo-random fill is representative.
 */
const { hammingTopK } = await import(process.env.SEARCH || "/tmp/search_paper.mjs");

const NS = (process.argv[2] || "50000").split(",").map(Number);
const CB = 128, REPEAT = 30;
const out = {};
for (const N of NS) {
  const db = new Uint8Array(N * CB);
  for (let i = 0; i < db.length; i++) db[i] = (i * 1103515245 + 12345) & 255; // varied dists
  const q = db.slice(0, CB);
  hammingTopK(db, N, CB, q, 10); // warm
  const ts = [];
  for (let r = 0; r < REPEAT; r++) {
    const t0 = performance.now();
    hammingTopK(db, N, CB, q, 10);
    ts.push(performance.now() - t0);
  }
  ts.sort((a, b) => a - b);
  out[N] = +ts[Math.floor(ts.length / 2)].toFixed(3);
}
console.log(JSON.stringify(out));
