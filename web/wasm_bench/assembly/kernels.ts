// WASM SIMD kernels for browser retrieval-latency benchmark.
// Built with AssemblyScript (`asc --enable simd`).
//
// Memory layout (all kernels operate on linear WASM memory):
//   - One query vector at byte offset `qOff`.
//   - A gallery of `n` items, each `stride` bytes, starting at `gOff`.
//   - An output scores buffer of `n` * 4 bytes (i32 or f32) at `sOff`.
//
// Equal-byte representations (128 B per item) compared here:
//   binary  : 1024 bits = 128 B = 8 x v128.  Score = Hamming distance (lower = closer).
//   int8    : 128 dims int8 = 128 B = 8 x v128. Score = dot product i32 (higher = closer).
//   fp32    : 64 dims f32 = 256 B = 16 x v128. Score = dot product f32 (higher = closer). Reference.
//
// All kernels fill the scores buffer; top-k selection is done identically in JS
// across every method, so it cannot bias the binary-vs-int8 comparison.

// Expose a flat memory we can write into from JS.
export const PAGE_SIZE: i32 = 65536;

// ---------------------------------------------------------------------------
// BINARY — Hamming distance via SIMD popcount (i8x16.popcnt).
// 128 B/item -> 8 v128 loads. Per load: xor, popcnt (per-byte 0..8),
// extadd_pairwise to i16 lanes, accumulate. Final horizontal sum -> distance.
// ---------------------------------------------------------------------------
export function binary_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = i16x8.splat(0);
    // 128 bytes = 8 * 16
    for (let b = 0; b < 128; b += 16) {
      let g = v128.load(base + b);
      let q = v128.load(qOff + b);
      let x = v128.xor(g, q);
      let pc = i8x16.popcnt(x);                       // per-byte counts 0..8
      let pairs = i16x8.extadd_pairwise_i8x16_u(pc);  // 8 lanes, each 0..16
      acc = i16x8.add(acc, pairs);
    }
    // horizontal sum of 8 i16 lanes (max total 1024, fits i32)
    let dist: i32 =
      <i32>i16x8.extract_lane_u(acc, 0) + <i32>i16x8.extract_lane_u(acc, 1) +
      <i32>i16x8.extract_lane_u(acc, 2) + <i32>i16x8.extract_lane_u(acc, 3) +
      <i32>i16x8.extract_lane_u(acc, 4) + <i32>i16x8.extract_lane_u(acc, 5) +
      <i32>i16x8.extract_lane_u(acc, 6) + <i32>i16x8.extract_lane_u(acc, 7);
    store<i32>(sOff + (i << 2), dist);
  }
}

// ---------------------------------------------------------------------------
// INT8 — dot product via SIMD (widen i8->i16, i32x4.dot_i16x8_s).
// 128 B/item -> 8 v128 loads of 16 int8 each. Accumulate i32x4.
// ---------------------------------------------------------------------------
export function int8_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = i32x4.splat(0);
    for (let b = 0; b < 128; b += 16) {
      let g = v128.load(base + b);
      let q = v128.load(qOff + b);
      // widen low 8 int8 -> i16, high 8 int8 -> i16
      let gl = i16x8.extend_low_i8x16_s(g);
      let gh = i16x8.extend_high_i8x16_s(g);
      let ql = i16x8.extend_low_i8x16_s(q);
      let qh = i16x8.extend_high_i8x16_s(q);
      acc = i32x4.add(acc, i32x4.dot_i16x8_s(gl, ql));
      acc = i32x4.add(acc, i32x4.dot_i16x8_s(gh, qh));
    }
    let dot: i32 =
      i32x4.extract_lane(acc, 0) + i32x4.extract_lane(acc, 1) +
      i32x4.extract_lane(acc, 2) + i32x4.extract_lane(acc, 3);
    store<i32>(sOff + (i << 2), dot);
  }
}

// ---------------------------------------------------------------------------
// FP32 — dot product via SIMD (f32x4 fma). 64 dims = 256 B = 16 v128 loads.
// Reference row (256 B, not equal-byte to the 128 B methods).
// ---------------------------------------------------------------------------
export function fp32_simd(qOff: i32, gOff: i32, sOff: i32, n: i32, stride: i32): void {
  for (let i = 0; i < n; i++) {
    let base = gOff + i * stride;
    let acc = f32x4.splat(0);
    for (let b = 0; b < 256; b += 16) {
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
