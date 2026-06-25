# WASM Search-Latency v2 — Chasing the popcount suspicion (v1 CORRECTED)

**Branch:** `paper-wasm-latency` (now on origin — see §0)
**Date:** 2026-06-25
**Trigger:** User's suspicion — *"XOR+popcount is cheaper compute than int8 MAC, so how can it NOT be faster?"* v1 reported a tie at equal SIMD and called it "memory-bandwidth bound." This pass tests that head-on with cache-resident galleries, kernel-only timing, a byte-size sweep (roofline), and an assembly audit.

## TL;DR: 🟡 PARTIAL — v1 was wrong on this point. **Binary popcount IS faster than int8 dot.**

The suspicion was correct. With a properly-optimized binary kernel and the top-k cost stripped out, **binary popcount beats int8 dot at every gallery size and every CPU speed** — by ~8–12 % at 128 B, **growing to ~30 % at 256 B and 2.7× in plain JS**. v1's "tie / memory-bound" conclusion was an artifact of (a) a redundant-mask inefficiency in v1's binary reduction and (b) lumping the shared O(N) top-k cost into the kernel time.

**But** the win at the equal-byte 128 B operating point is **~0.2 ns/item ≈ 20–40 µs over a 100 K gallery — imperceptible** — and int8 still wins accuracy at 128 B (R@10 80.36 vs 79.88). So binary's latency edge is **real but small and dimension-dependent**, not the 2× the op-count implies, and not enough on its own to resurrect the 1-bit thesis. The honest headline moves from *"binary has no latency advantage"* → *"binary has a small, real latency advantage that doesn't matter much at 128 B but grows for longer codes / non-SIMD targets."*

---

## §0 — Branch is now on origin (was missing; clone-verify unblocked)
```
$ git ls-remote origin paper-wasm-latency
78d43aa…  refs/heads/paper-wasm-latency   (before v2)
<this commit SHA after v2 push — see end>
```
v1 artifacts (`paper/wasm_latency.csv`, `web/wasm_bench/`) plus all v2 files are pushed.

## §2 — Assembly audit: both kernels are optimal SIMD, fairly matched
From `web/wasm_bench/build/kernels.simd.wat` (v2, `asc --enable simd`), per 16-byte chunk:

| | per chunk | per item @128B (8 chunks) | gallery loads @128B |
|---|---|---|---|
| **binary** | `v128.xor` → `i8x16.popcnt` → `i16x8.extadd_pairwise` → `add` = **4 vec-ops** | 32 vec-ops + 16 v128 loads | 8 (+8 query) |
| **int8** | 2×`i16x8.extend_*` → `i32x4.dot_i16x8_s` → `add`, ×hi/lo = **8 vec-ops** | 64 vec-ops + 16 v128 loads | 8 (+8 query) |

- Both use the canonical best SIMD op (`i8x16.popcnt`, `i32x4.dot_i16x8_s`). Neither is hand-favored.
- **Loads are identical (16 v128/item @128 B).** int8 has exactly **2× the vector-ALU**. This is the whole story.
- **v1 bug found & fixed:** v1's binary reduction emitted a redundant `i32.const 65535 / i32.and` after each `extract_lane_u` (8× per item) and used an 8-lane scalar reduction. v2 replaces it with one `i32x4.extadd_pairwise_i16x8_u` + 4 extracts. This is why v1 under-measured binary.

## §1 — The decisive tests

### (A) Cache-resident sweep — is it DRAM-bound? **No.** And binary wins everywhere.
WASM-SIMD, **scan-only** (no top-k), ns per gallery item:

| gallery | bytes in cache | binary ns | int8 ns | bin/int8 |
|---|---|---|---|---|
| 256 | 32 KB (L1) | 2.73 | 2.94 | **0.93** |
| 1 000 | 128 KB | 2.82 | 3.14 | **0.90** |
| 5 000 | 640 KB | 2.78 | 2.94 | **0.95** |
| 50 000 | 6.4 MB | 2.70 | 2.91 | **0.93** |
| 100 000 | 12.8 MB | 2.74 | 2.90 | **0.95** |

**ns/item is flat from 256 → 100 000.** If this were DRAM-bandwidth-bound, per-item cost would rise sharply once the gallery spills L1/L2 — it doesn't, because Apple Silicon's large L2/SLC (~16 MB) holds even 12.8 MB. So v1's *"memory-bandwidth bottleneck"* was imprecise: at these sizes it is **load-throughput / compute bound, cache-resident**, and **binary is ~5–12 % faster at every size** (including the L1-resident 256 case the suspicion targeted).

### (B) Byte-size sweep (roofline) — the smoking gun: binary's edge grows with code length
WASM-SIMD scan-only, n = 50 000:

| bytes/item | binary ns | int8 ns | bin/int8 | binary ns/byte | int8 ns/byte |
|---|---|---|---|---|---|
| 16 (128-bit) | 0.70 | 0.69 | 0.99 (tie) | 0.0436 | 0.0433 |
| 32 | 0.96 | 0.98 | 0.97 | 0.0300 | 0.0305 |
| 64 | 1.58 | 1.67 | 0.95 | 0.0247 | 0.0260 |
| 128 | 2.74 | 2.98 | **0.92** | 0.0214 | 0.0233 |
| 256 (2048-bit) | 5.07 | 7.18 | **0.71** | 0.0198 | 0.0280 |

- **binary's advantage scales with vector length: tie @16 B → 8 % @128 B → 29 % @256 B.**
- **ns/byte tells the roofline:** binary flattens toward a **~0.020 ns/byte floor** (load/bandwidth-limited); int8 bottoms at ~0.023 @128 B then **rises to 0.028 @256 B** — it crosses into **compute-bound** because of its 2× arithmetic intensity. Binary stays load-bound.

### Why only ~10 % at 128 B, not the 2× the op-count implies (the core of the suspicion)
At 128 B both kernels issue **16 identical v128 loads/item**; loads are the shared near-bottleneck, and a superscalar core overlaps int8's extra ALU with those loads. So binary's halved ALU is mostly *hidden* and surfaces only ~10 %. The full 2× appears **only when ALU actually binds** — long codes (256 B → 29 %) or no SIMD at all (scalar JS → 2.7×, below). popcount *is* cheaper and *is* faster; the magnitude is set by where you sit on the roofline.

### (C) top-k dilution (why v1 looked like a tie)
WASM-SIMD @128 B, n = 100 000: binary scan 2.74 → +topk 4.50 ns/item; int8 scan 2.90 → +topk 4.83. The shared O(N) top-k adds ~1.8 ns/item to both. v1 timed kernel+topk *and* had the handicapped binary reduction, collapsing the gap to a "tie." Isolating the kernel restores binary's edge.

### (D) Plain JS (no SIMD) — binary 2.7× faster
scan-only ns/item @128 B: binary ~37, int8 ~98 (bin/int8 ≈ 0.37), flat across sizes. Without SIMD, int8's 128 scalar MACs lose badly to 32-word XOR+SWAR-popcount. Matches v1's ~2.3–2.7×.

### (mobile proxy) CPU-throttle 4× — edge is invariant to compute speed
binary/int8 @128 B scan: 1× → 0.90–0.95, 4× → 0.91–0.92. Throttle scales all methods ~4.2× uniformly; the modest binary edge persists, neither widening nor closing. (CDP `setCPUThrottlingRate`, same real engine — labelled proxy, **not** a real phone.)

## Practical significance (the honest big-picture line)
At the deployable 128 B equal-byte point, binary's SIMD win is ~0.2 ns/item → **~0.02 ms (scan) / ~0.04 ms (with top-k) over a 100 K gallery**. Imperceptible to a user. Meanwhile int8 head-z holds **R@10 80.36 vs binary 79.88** at the same 128 B. So: binary trades ~0.5 pt accuracy for ~tens-of-µs latency. **The latency edge is real and now correctly measured, but too small at 128 B to justify the accuracy loss when int8 exists at equal bytes.** Binary's edge becomes *materially* large only for (i) very long codes (≥256 B / 2048-bit, ~30 %) or (ii) SIMD-less JS deployments (2.7×).

## Verdict
- **v1 CORRECTION (honest):** "binary popcount ≈ int8, no advantage, memory-bandwidth bound" was **wrong**. Binary popcount **is** faster (cache-resident at all tested sizes; ~8–12 % @128 B, up to 30 % @256 B, 2.7× scalar-JS), and the regime is load/compute-bound, not DRAM-bound, on this hardware.
- **Thesis impact:** still does **not** by itself resurrect pure 1-bit — at 128 B the win is microseconds and int8 wins accuracy. But the deploy narrative should now read: *1-bit gives a small, real, dimension-scaling search-latency edge (large for long codes / non-SIMD targets), at a ~0.5 pt accuracy cost vs int8 at equal bytes.* Not "no edge."
- **Scale caveat:** on a device with a small cache (≤8 MB) or a 1 M+ gallery that spills to DRAM, the equal 128 B/item traffic would dominate and the gap would shrink back toward the byte-bound tie. Verified regime: ≤100 K on Apple-Silicon-class cache.

## Not done (honest scope)
- **WebGPU**: not measured (needs a separate compute-shader harness; the CPU/WASM question is settled without it). Flagged for a future pass if parallel-popcount on GPU is of interest.
- **Real mobile device**: used a labelled 4× CPU-throttle proxy, not a physical phone.

## Reproduce
```bash
cd web/wasm_bench
npm install
npm run asbuild                       # build/kernels.simd.{wasm,wat}
node run_bench_v2.mjs --rates=1       # -> paper/wasm_latency_v2.csv (+ bench_results_v2.json)
node run_bench_v2.mjs --rates=1,4     # add mobile-proxy throttle
# manual: serve dir, open bench_v2.html in desktop/mobile Chrome, click Run
```

## Artifacts
- `paper/wasm_latency_v2.csv` — experiment × gallery{256,1K,5K,50K,100K} × method × impl × phase{scan,topk} × bytes × cpu_throttle × {p50,p95,ns/item,ns/byte}
- `web/wasm_bench/bench_v2.html`, `run_bench_v2.mjs`, `assembly/kernels.ts`, `build/kernels.simd.{wasm,wat}`, `bench_results_v2.json`
- Correctness self-check (in-page): binary/int8 WASM == JS-scalar, maxAbsDiff 0.
