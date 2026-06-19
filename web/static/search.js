/* Pure client-side Hamming search core — NO DOM, NO network.
 *
 * Shared by app.js (browser) and verify_js.mjs (Node parity test), and trivially
 * movable into a Web Worker. Mirrors web/verify_parity.py byte-for-byte:
 *   - 256-entry popcount LUT
 *   - dist = sum_k LUT[a[k] ^ b[k]]  over the 128-byte rows
 *   - Top-K with ties broken by ASCENDING index (so results match faiss IndexBinaryFlat
 *     deterministically).
 */

// 256-entry popcount lookup table (popcount of each byte value).
export const LUT = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < 256; i++) t[i] = (i & 1) + t[i >> 1];
  return t;
})();

// index: Uint8Array of n*cb bytes; q: Uint8Array(cb). Returns [{idx,dist}] sorted by
// (dist asc, idx asc), length <= K. Bounded max-heap of capacity K (O(n log K)).
export function hammingTopK(index, n, cb, q, K) {
  const hd = new Uint16Array(K);   // heap distances (root = current worst kept)
  const hi = new Int32Array(K);    // heap indices
  let hs = 0;                      // heap size
  // a "worse" than b iff larger dist, or equal dist & larger index
  const worse = (aD, aI, bD, bI) => aD > bD || (aD === bD && aI > bI);

  for (let i = 0; i < n; i++) {
    const off = i * cb;
    let d = 0;
    for (let k = 0; k < cb; k++) d += LUT[index[off + k] ^ q[k]];

    if (hs < K) {                                  // grow heap
      let c = hs++;
      hd[c] = d; hi[c] = i;
      while (c > 0) {                              // sift up (max-heap)
        const p = (c - 1) >> 1;
        if (worse(hd[c], hi[c], hd[p], hi[p])) {
          const td = hd[c]; hd[c] = hd[p]; hd[p] = td;
          const ti = hi[c]; hi[c] = hi[p]; hi[p] = ti;
          c = p;
        } else break;
      }
    } else if (d < hd[0] || (d === hd[0] && i < hi[0])) {  // better than worst -> replace root
      hd[0] = d; hi[0] = i;
      let c = 0;                                   // sift down
      for (;;) {
        const l = 2 * c + 1, r = l + 1; let m = c;
        if (l < hs && worse(hd[l], hi[l], hd[m], hi[m])) m = l;
        if (r < hs && worse(hd[r], hi[r], hd[m], hi[m])) m = r;
        if (m === c) break;
        const td = hd[c]; hd[c] = hd[m]; hd[m] = td;
        const ti = hi[c]; hi[c] = hi[m]; hi[m] = ti;
        c = m;
      }
    }
  }
  const out = [];
  for (let i = 0; i < hs; i++) out.push({ idx: hi[i], dist: hd[i] });
  out.sort((a, b) => a.dist - b.dist || a.idx - b.idx);
  return out;
}

// Pack a length-D (D multiple of 8) array of ±1/continuous values into D/8 bytes.
// bit = (value > 0); big-endian within each byte (first value -> MSB). Byte-identical to
// web/common.py:pack_bits (np.packbits(code > 0, bitorder='big')). Used by the offline
// query path (browser packs its own code) and the Node parity test.
export function packBits(values) {
  const nb = values.length >> 3;
  const out = new Uint8Array(nb);
  for (let b = 0; b < nb; b++) {
    let byte = 0;
    const off = b << 3;
    for (let k = 0; k < 8; k++) {
      if (values[off + k] > 0) byte |= 1 << (7 - k);
    }
    out[b] = byte;
  }
  return out;
}
