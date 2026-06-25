// WASM SIMD kernels for browser retrieval-latency benchmark (v2).
// Built with AssemblyScript (`asc --enable simd`).
//
// v2 changes vs v1:
//   - byte-length is a PARAMETER (`nbytes`) so we can sweep 16/32/64/128/256 B
//     per item for the roofline / latency-vs-bytes slope analysis.
//   - binary reduction uses a single SIMD horizontal add (no per-lane i32.and),
//     removing the v1 redundant-mask handicap on the binary side.
//
// Each kernel scans a gallery and writes a per-item score to sOff:
//   binary : Hamming distance (lower = closer), i32.   nbytes bits-per-item / 8.
//   int8   : dot product (higher = closer), i32.        nbytes int8 dims.
//   fp32   : dot product (higher = closer), f32.        nbytes/4 f32 dims.

export const PAGE_SIZE: i32 = 65536;

// BINARY — Hamming via SIMD popcount. nbytes must be a multiple of 16.
export function binary_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32, nbytes: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = i16x8.splat(0);
    for (let b = 0; b < nbytes; b += 16) {
      let g = v128.load(base + b);
      let q = v128.load(qOff + b);
      let pc = i8x16.popcnt(v128.xor(g, q));
      acc = i16x8.add(acc, i16x8.extadd_pairwise_i8x16_u(pc));
    }
    // horizontal sum of 8 u16 lanes via widening pairwise adds (no per-lane mask)
    let w = i32x4.extadd_pairwise_i16x8_u(acc);           // 4 i32 lanes
    let dist: i32 =
      i32x4.extract_lane(w, 0) + i32x4.extract_lane(w, 1) +
      i32x4.extract_lane(w, 2) + i32x4.extract_lane(w, 3);
    store<i32>(sOff + (i << 2), dist);
  }
}

// INT8 — dot via SIMD (widen i8->i16, i32x4.dot_i16x8_s). nbytes = dims, mult of 16.
export function int8_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32, nbytes: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = i32x4.splat(0);
    for (let b = 0; b < nbytes; b += 16) {
      let g = v128.load(base + b);
      let q = v128.load(qOff + b);
      acc = i32x4.add(acc, i32x4.dot_i16x8_s(i16x8.extend_low_i8x16_s(g), i16x8.extend_low_i8x16_s(q)));
      acc = i32x4.add(acc, i32x4.dot_i16x8_s(i16x8.extend_high_i8x16_s(g), i16x8.extend_high_i8x16_s(q)));
    }
    let dot: i32 =
      i32x4.extract_lane(acc, 0) + i32x4.extract_lane(acc, 1) +
      i32x4.extract_lane(acc, 2) + i32x4.extract_lane(acc, 3);
    store<i32>(sOff + (i << 2), dot);
  }
}

// FP32 — dot via SIMD f32x4. nbytes = 4*dims, mult of 16.
export function fp32_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32, nbytes: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = f32x4.splat(0);
    for (let b = 0; b < nbytes; b += 16) {
      let g = v128.load(base + b);
      let q = v128.load(qOff + b);
      acc = f32x4.add(acc, f32x4.mul(g, q));
    }
    let dot: f32 =
      f32x4.extract_lane(acc, 0) + f32x4.extract_lane(acc, 1) +
      f32x4.extract_lane(acc, 2) + f32x4.extract_lane(acc, 3);
    store<f32>(sOff + (i << 2), dot);
  }
}
