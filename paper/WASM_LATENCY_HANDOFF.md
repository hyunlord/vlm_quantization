> ⚠️ **SUPERSEDED IN PART by `WASM_LATENCY_V2_HANDOFF.md` (2026-06-25).** A follow-up that
> isolates the kernel (no top-k), tests cache-resident galleries, and sweeps byte-length found
> this doc's "binary ≈ int8 tie, memory-bandwidth bound" conclusion was **wrong**: binary
> popcount IS faster (~8–12 % @128 B, up to 30 % @256 B, 2.7× scalar-JS), the regime is
> load/compute-bound (cache-resident), not DRAM-bound, and v1 here under-measured binary due
> to a redundant-mask reduction inefficiency + top-k dilution. The *practical* conclusion still
> holds (the 128 B win is ~tens of µs and int8 wins accuracy at equal bytes), but the mechanism
> claims below are corrected by v2. Read v2 for the accurate story.

# WASM Browser Search-Latency — Final 1-bit Life-or-Death Verdict

**Branch:** `paper-wasm-latency`
**Date:** 2026-06-25
**Question:** In a *real browser (WASM)*, is binary popcount search **decisively faster** than int8-dot at the same byte budget and same accuracy? This is the only axis on which pure 1-bit could survive (equal-byte accuracy-per-byte already lost: int8 80.4 ≥ binary 79.9 R@10 @128 B).

## TL;DR Verdict: 🔴 **RED — binary has NO fair latency advantage.**

Under the mandated fairness rule (*both methods at the same optimization level, no tuning only popcount*):

| Comparison level | binary vs int8 @100K | Winner |
|---|---|---|
| **WASM SIMD (both optimized)** | 0.475 ms vs **0.467 ms** | **TIE — int8 ~2% faster** |
| JS scalar (both naive) | **3.53 ms** vs 8.2 ms | binary 2.3× faster |

The binary 2.3× win exists **only in the naive-JS regime** — i.e., only if you refuse to write the ~30 lines of WASM-SIMD int8 dot (which this experiment shows is trivial and already done). The moment int8 is optimized to the same level as binary, the advantage **vanishes**: both become memory-bandwidth-bound at 128 B/item and run identically (int8 marginally faster). The decisive "2×+ at 50K" bar is **not** met at equal optimization.

**→ Drop the "1-bit = latency 우위" pillar. Reframe around `int8 head-z 128 B browser-native`** (equal compression, ≥ accuracy, equal SIMD latency). See honest niche below for the one surviving exception.

---

## Measurement (real browser, no extrapolation)

- **Engine:** Playwright **Chromium / Chrome-for-Testing, HeadlessChrome 149.0.7827.55** — a real browser V8+WASM engine, *not* torch/numpy/desktop-FAISS and not an emulator.
- **Device:** Apple Silicon Mac, macOS 25.4, 14 logical cores, 32 GB. CPU-throttle sweep (1× / 4× / 6× via CDP `Emulation.setCPUThrottlingRate`) emulates slower / mobile-class **compute** on the same real engine — honestly labelled as throttled CPU, **not** a real mobile device. For a true cross-device run, open `web/wasm_bench/bench.html` on real desktop/mobile Chrome ("Run" button, same code path).
- **WASM SIMD supported:** `true` (verified via `WebAssembly.validate` of an `i8x16.popcnt` module).
- **Galleries built for real:** 5 000 / 50 000 / 100 000 items (actual buffers scanned, no extrapolation from a small index).
- **Isolated region:** *gallery scan + top-10 only*, after warm-up. Model load / encoding excluded. Top-k is the **identical** JS routine for every method, so it cannot bias binary-vs-int8.
- **Timing:** `performance.now()` is coarsened to ~0.1 ms in headless Chrome, so each timed sample batches a **calibrated number of queries (~8 ms/batch)** and divides — per-query latency is resolved to ~10 µs. p50/p95 over 60 batches (effective n = 60 × inner-reps queries, e.g. 1 200–19 740 per cell).
- **Correctness self-check (in-page):** WASM-SIMD output == JS-scalar reference. binary maxAbsDiff **0**, int8 maxAbsDiff **0**, fp32 maxAbsDiff 9.5e-7 → all PASS. The SIMD kernels compute the right answer.

### Equal-byte / equal-accuracy setup (128 B each, R@10 parity from `equalbyte_pareto.csv`)
| method | repr | bytes | R@10 EN |
|---|---|---|---|
| binary | 1024-bit | 128 | 79.88 |
| int8 head-z | 128-d int8 | 128 | 80.36 |
| fp32 (reference) | 64-d f32 | 256 | 80.54 |

### Fairness: both implementations exist at both optimization levels
- **WASM SIMD** (`web/wasm_bench/assembly/kernels.ts`, AssemblyScript `--enable simd`): binary uses `i8x16.popcnt` + `extadd_pairwise`; int8 uses `i16x8.extend_*` + `i32x4.dot_i16x8_s`; fp32 uses `f32x4`. **Both** binary and int8 are SIMD-optimized — neither is handicapped.
- **JS scalar** (the realistic PWA path): binary = `Uint32Array` XOR + SWAR popcount; int8 = `Int8Array` MAC loop. **Both** naive.

## Full results (p50/p95 ms per query)

| gallery | method | impl | B | p50 | p95 |
|---|---|---|---|---|---|
| 5 000 | binary | wasm_simd | 128 | 0.0243 | 0.0252 |
| 5 000 | int8 | wasm_simd | 128 | 0.0242 | 0.0255 |
| 5 000 | binary | js_scalar | 128 | 0.177 | 0.181 |
| 5 000 | int8 | js_scalar | 128 | 0.410 | 0.419 |
| 50 000 | binary | wasm_simd | 128 | 0.235 | 0.238 |
| 50 000 | int8 | wasm_simd | 128 | 0.233 | 0.240 |
| 50 000 | binary | js_scalar | 128 | 1.76 | 1.80 |
| 50 000 | int8 | js_scalar | 128 | 4.05 | 4.20 |
| 100 000 | binary | wasm_simd | 128 | 0.475 | 0.495 |
| 100 000 | int8 | wasm_simd | 128 | 0.467 | 0.490 |
| 100 000 | binary | js_scalar | 128 | 3.53 | 3.67 |
| 100 000 | int8 | js_scalar | 128 | 8.20 | 8.30 |
| 100 000 | fp32 | wasm_simd | 256 | 0.664 | 0.679 |
| 100 000 | fp32 | js_scalar | 256 | 3.83 | 4.00 |

(Table shows cpu_throttle=1×.) Full table incl. fp32 + 4×/6× throttle: `paper/wasm_latency.csv` (now has a `cpu_throttle` column). Raw JSON (env + checks): `web/wasm_bench/bench_results.json`.

## Robustness to slower / mobile-class compute (CPU-throttle sweep)

binary vs int8 **at equal WASM-SIMD optimization**, p50 ms @100K, across throttle:

| cpu_throttle | binary | int8 | ratio (int8/bin) |
|---|---|---|---|
| 1× | 0.467 | 0.471 | 1.01 (tie) |
| 4× | 1.94 | 1.92 | 0.99 (tie) |
| 6× | 2.90 | 2.87 | 0.99 (tie) |

The SIMD tie is **invariant to CPU speed** — throttle scales all methods near-linearly (4×→~4.1×, 6×→~6.2× latency), so the memory-bound equal-bytes conclusion holds on mobile-class compute. The naive-JS binary/int8 ratio also stays ~2.3× at every throttle (6× @100K: 22.1 vs 50.5 ms). On a slow device the naive int8 path becomes genuinely sluggish (50 ms), but **int8 SIMD (2.87 ms) still beats binary-naive (22 ms) by 7–8×** — so the correct slow-device choice is SIMD-int8, not naive-binary. Binary's edge does not reappear on weaker hardware.

## Answering the ★ judgment questions

**1. Is binary popcount decisively (2×+) faster than int8-dot?**
- **At equal SIMD optimization: NO.** Tie — int8 is 0–2 % *faster* (5K 1.00×, 50K 0.99×, 100K 0.98×).
- At equal naive JS: yes, 2.3× — but this regime leaves int8 un-optimized, which the fairness rule forbids as the headline comparison.

**2. Does the gap widen with scale (5K→100K)?**
- **No.** Ratios are flat across scale: SIMD ≈ 1.0× at every size; naive ≈ 2.3× at every size. Latency is a pure linear scan for all methods, so the binary/int8 relationship is a **constant factor set by impl level, not by gallery size**. Binary does not become more advantageous at scale.

**3. Does WASM SIMD popcount beat int8 when both use SIMD?**
- **No** — they tie. Root cause confirmed by the fp32 control: at SIMD, fp32 (256 B) is **1.40× slower** than binary (128 B) — latency tracks **bytes moved, not op-count**. The scan is **memory-bandwidth-bound**, so two methods at the same 128 B/item cost the same regardless of whether the per-item op is popcount or a dot product. Popcount's compute cheapness is hidden behind memory latency and buys nothing.

## Why binary loses every angle at 128 B
- **Accuracy:** int8 80.36 ≥ binary 79.88.
- **Compression:** both are 128 B → **both** give 36× over fp32-1152d (4608 B). 36× is *not* unique to binary.
- **Latency (fair/optimized):** tie; int8 marginally faster.
- Net: at equal bytes, int8 head-z dominates or ties binary on **all three** axes.

## The one honest niche where binary still wins 🟡
A **pure-JS PWA with no WASM/build toolchain** (ships only `.js`, hand-rolled bitwise search — the original repo demo path) gets a **real 2.3× latency reduction** from binary popcount *and* 36× compression vs float. If "no WASM, JS-only, large client gallery" is a hard deployment constraint, 1-bit remains defensible there. Absolute numbers, though, are tiny on both sides (100K: 3.5 ms vs 8.2 ms — both comfortably interactive), so even this niche is a convenience, not a necessity.

## Recommendation
Retire the **latency** survival thesis for pure 1-bit. The paper's deploy story should be **`int8 head-z, 128 B, browser-native (WASM-SIMD dot)`**: equal compression, equal-or-better accuracy, equal-or-faster search. Mention binary only as: (a) the matched-accuracy equal-byte baseline, and (b) the JS-only-PWA fallback that trades ~0.5 pt R@10 for a 2.3× scan speedup *when SIMD is unavailable*. Matryoshka remains the real lever (see `equalbyte-pareto-verdict`).

## Reproduce
```bash
cd web/wasm_bench
npm install                 # assemblyscript + playwright
npm run asbuild             # build/kernels.simd.wasm (verify i8x16.popcnt / i32x4.dot_i16x8_s in .wat)
node run_bench.mjs          # headless Chromium, throttle sweep 1x/4x/6x -> paper/wasm_latency.csv
node run_bench.mjs --rates=1 --headed   # single un-throttled run, watch it
# manual cross-device: serve web/wasm_bench over http, open bench.html in desktop/mobile Chrome, click "Run"
```

## Artifacts
- `paper/wasm_latency.csv` — gallery × method × impl × {p50,p95,mean,n}
- `web/wasm_bench/bench.html` — the in-browser harness (correctness check + protocol)
- `web/wasm_bench/assembly/kernels.ts` + `build/kernels.simd.{wasm,wat}` — SIMD kernels (readable .wat)
- `web/wasm_bench/run_bench.mjs` — Playwright driver
- `web/wasm_bench/bench_results.json` — raw env + correctness + results
