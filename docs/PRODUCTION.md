# Productionization Guide — DGX Spark → Image-Retrieval Service

This document covers how to train/test on **DGX Spark** and how to take the
cross-modal hashing model to a **production image-retrieval service**. The product
promise is two things, and the tooling here is built to prove both:

1. **Quality** — hash codes preserve retrieval results close to full float search.
2. **Speed/size** — Hamming search over compact bitstrings is far smaller and faster
   than float-embedding ("normal") image search.

> A 64-bit code is **8 bytes/item** vs **~4,608 bytes** for an fp32 1152-d embedding
> (~**576× smaller**). Hamming distance is integer XOR+popcount, not float dot-product.

---

## 0. What was built for the service

| Component | File | Purpose |
|---|---|---|
| Binary index core | `src/serve/binary_index.py` | `HammingIndex` (numpy popcount / FAISS), `FloatIndex` baseline, `MatryoshkaIndex` (coarse→fine), `ServingIndex` loader |
| Index builder | `scripts/build_serving_index.py` | Encode a corpus (or convert a dashboard `.pt`) → packed `.npz` |
| Benchmark | `scripts/bench_search.py` | Measures Recall@k, latency, QPS, index size: **hash vs float** |
| Retrieval API | `src/serve/api.py` | FastAPI: text→image / image→image search with `took_ms` |
| Tests | `tests/test_binary_index.py`, `tests/test_api_import.py` | CPU correctness of packing/Hamming/Matryoshka + API smoke |

Dashboard index builder `scripts/build_index.py` was also repaired (it referenced a
removed "shared bottleneck" architecture and would have crashed).

---

## 1. DGX Spark — train & test

DGX Spark = GB10 (Grace-Blackwell, **aarch64**, **CUDA 13**, **128 GB unified memory**).
The repo already targets it: `configs/dgx_spark.yaml`, unified-memory handling in
`src/utils/gpu_config.py`, and the `cu130` wheel index in `pyproject.toml`.

```bash
# 0) environment (aarch64 auto-selects cu130 torch)
uv sync                       # or: pip install -e ".[dev,serve]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1) regression gate — CPU tests (fast, no GPU/backbone)
pytest tests/ -q

# 2) speed/size proof on CPU right now (no trained model needed)
python scripts/bench_search.py --synthetic --n 100000 --queries 1000 --bit 64

# 3) train + live dashboard
python -m monitor.server &
python train.py --config configs/dgx_spark.yaml

# 4) quality eval — per-bit mAP / P@K / R@K, both directions
python eval.py --checkpoint checkpoints/<run>/best-*.ckpt --config configs/dgx_spark.yaml
```

**The gate that matters (Phase 0):** during validation the model logs
`val/map_i2t` (hash) next to `val/backbone_map_i2t` (frozen-backbone cosine). **Hash mAP
must reach or beat the backbone baseline** — otherwise hashing is losing information and
there is no point building the serving stack yet. Also pick the **operating bit length**
here (shorter = smaller/faster index, lower recall).

Thermal note: `dgx_spark.yaml` already lowers batch/workers "to reduce heat"; watch
temps on long runs.

---

## 2. Build the serving index

```bash
# Fast path: convert the dashboard index you already built (no GPU needed)
python scripts/build_serving_index.py \
    --from-pt checkpoints/<run>/index_test.pt \
    --bits 16,64 --save-emb --out indexes/serving.npz

# Or encode a fresh image corpus (JSONL of {"image_path": ...})
python scripts/build_serving_index.py \
    --checkpoint checkpoints/<run>/best.ckpt \
    --jsonl data/corpus.jsonl --data-root data/coco \
    --bits 16,64 --save-emb --out indexes/serving.npz
```

`--save-emb` stores fp32 baseline embeddings so the benchmark can compare against
"normal" float search. Drop it in production to keep the index tiny.

---

## 3. Prove the speed/quality story

```bash
python scripts/bench_search.py --index indexes/serving.npz --bit 64 --k 10
```

Outputs a table — index MB, ms/query, QPS, Recall@k — for **float cosine (exact)** vs
**hash Hamming** (+FAISS binary if `faiss-cpu` is installed), plus the compression ratio.
This is the artifact to show stakeholders: *"same top-k results, Nx smaller, Nx faster."*
Numbers are measured on the host; FAISS `IndexBinaryFlat` uses hardware popcount and is
typically far faster than the numpy fallback.

**Coarse-to-fine** (`MatryoshkaIndex`): filter a large candidate set with the 16-bit
prefix, rerank only those with 64/128-bit — long-code accuracy at short-code speed. This
is the lever for scaling to large corpora.

---

## 4. Serve it

```bash
export RETRIEVAL_CHECKPOINT=checkpoints/<run>/best.ckpt
export RETRIEVAL_INDEX=indexes/serving.npz
# export RETRIEVAL_FAISS=1     # if faiss-cpu installed
uvicorn src.serve.api:app --host 0.0.0.0 --port 8100

curl -s localhost:8100/search/text \
  -H 'content-type: application/json' \
  -d '{"query":"a dog running on the beach","k":5,"bit":64}'
```

Every response carries `took_ms.{encode,search}` so latency is visible per request.

> **macOS dev caveat:** `faiss-cpu` and `torch` can deadlock in one process on macOS
> (duplicate libomp). The index loads faiss lazily and the test suite skips the faiss
> unit test when torch is already imported, so this only affects running the faiss
> backend *and* torch together locally. On Linux / DGX they coexist fine — keep
> `RETRIEVAL_FAISS=1` there.

---

## 5. Phased roadmap to production

| Phase | Goal | Exit criterion |
|---|---|---|
| **0 · Validate** | Does hashing beat the float baseline? Pick bit length. | `val/map_i2t ≥ val/backbone_map_i2t`; bench shows acceptable Recall@k |
| **1 · Index pipeline** | Reproducible corpus → packed index + ANN | Versioned `.npz`; build is restartable; FAISS binary wired |
| **2 · Online serving** | Query encoder + search behind an API | p99 latency + Recall SLO met under load; encode batched on GPU |
| **3 · Productionize** | Containerize, autoscale, observe | Eval gate in CI blocks regressions; index refresh + drift monitoring |

### Cross-cutting decisions / risks
- **Model ↔ index are version-locked.** Re-encoding the corpus is required whenever the
  encoder changes (code space shifts). Plan blue/green index swaps.
- **Reproduce inference exactly.** Serving must use eval-mode BatchNorm running stats and
  the same SigLIP2 Gemma tokenizer/processor (note the `SiglipProcessor` fallback in the
  data module) — otherwise query codes won't match the index.
- **DGX Spark is a great build/inference box, not an HA cluster.** For real traffic,
  deploy the encoder to cloud GPU/CPU and the index to a dedicated search service.
- **Don't over-build before Phase 0.** If the model doesn't beat the baseline, the rest is
  wasted; the synthetic benchmark and the eval harness exist to make that call cheaply.

### Concrete next steps (recommended order)
1. Run Phase 0 on DGX; confirm the baseline gate and choose the bit length.
2. Export the encoder to ONNX/TensorRT (aarch64 / x86) for fast, dependency-light serving.
3. Add an offline eval harness with real labels (`instances_json`) → category mAP, and a
   Korean/multilingual retrieval benchmark (KO R@K).
4. Wire FAISS `IndexBinaryHNSW`/`IndexBinaryMultiHash` for >10⁶ corpora; add index refresh.
