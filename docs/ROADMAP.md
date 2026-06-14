# Roadmap & Product Vision

Where this model goes next: (A) how to advance retrieval quality, and (B) a concrete
side-project it enables — **on-device natural-language photo search**.

_Last updated: 2026-06-14._

---

## Part A — Model advancement roadmap

Current baseline: frozen SigLIP2 backbone + nested binary hash heads, fixed
hyperparameters, swept over code length (`bit_list` 128/256/512/1024) and data
(augmentation, extra datasets). Already decent — e.g. 1024-bit R@10 ≈ 0.75 vs the
float ceiling ≈ 0.81 — so the work below is about closing that last gap and raising
the ceiling itself.

### Tier 1 — Hyperparameter tuning (cheap, do first)

The backbone is frozen, so we tune on **cached embeddings** with no backbone forward —
hundreds of Optuna trials are realistic. `optuna_search.py` already exists.

Search space: `hash_lr`, `weight_decay`, `temperature` (InfoNCE), the loss weights
(`ortho` / `quantization` / `balance` / `consistency` / `lcs`), `hidden_dim`,
`dropout`, `focal_gamma`, warmup/schedule. Objective: val mAP / R@10 at a chosen
`bit` (e.g. 64 or 256), or a multi-bit average.

### Tier 2 — Method advancement (what HP tuning can't reach)

- **Unfreeze / LoRA the backbone.** Currently `freeze_backbone: true`. Letting the
  encoder adapt to the hashing objective (full unfreeze, or low-rank/LoRA, or just the
  last N blocks) is the most likely *large* jump — at higher compute cost.
- **Stronger backbone.** SigLIP2-giant / MetaCLIP2 (see `BACKBONE_CANDIDATES.md`)
  raise the float ceiling itself. Drop-in because the backbone is frozen.
- **Float → hash distillation.** Treat the float-cosine ranking as a teacher and
  distill its ordering into the binary student (rank/list-wise loss), not just InfoNCE.
- **Hard-negative mining** + improved quantization loss (reduce the bf16/binary gap).
- **More + better data** (in progress): DOCCI (dense), Flickr30K (diverse); next
  ShareGPT4V (GPT-4V dense captions at COCO scale).

### Tier 3 — Evaluation hardening

- Add **nocaps** (out-of-domain) and **XM3600** (multilingual) benchmarks so quality
  claims generalize beyond COCO. Keep validating the raw backbone vs published numbers
  (`scripts/zeroshot_benchmark.py`).

**Order:** lock the current baseline (running) → Tier 1 HP tuning on the best option →
then decide Tier 2 based on the gap that remains.

---

## Part B — Side project: on-device natural-language photo search

> "우리집 강아지 태기가 졸려하는 사진" → instantly surfaces the right photos from your
> own library, offline.

### What already exists (honest landscape)

This is **not** an empty space — be clear-eyed about it:

- **Apple Photos** (iOS 18+) has on-device natural-language search ("dog on a beach").
- **Google Photos** has strong semantic search (cloud-based).
- **immich** (open-source, self-hosted) already does CLIP-based smart search;
  PhotoPrism has some too.

So the *concept* is solved by big tech and one OSS project. The opening is in **how**,
not whether.

### Why binary cross-modal hashing is a real differentiator

The angle is **efficiency**, backed by concrete numbers, not hype:

- **Index size.** A 1024-bit code is **128 bytes/image**. A 256-bit code is **32
  bytes/image**.
  - 100k photos @ 1024-bit ≈ **12.8 MB**; @ 256-bit ≈ **3.2 MB**.
  - 1M photos @ 1024-bit ≈ **128 MB**.
  - vs float SigLIP2 (1152-dim fp32 = 4,608 B/img): 100k ≈ 461 MB, 1M ≈ 4.6 GB —
    roughly **36× larger** than the 1024-bit index (18× vs fp16).
- **Search.** Hamming distance = XOR + popcount (pure bitwise, SIMD-friendly). At
  100k–1M codes of 128 bytes, brute-force scan is single-digit-to-tens of ms on a
  phone CPU — **no ANN index required** at this scale. (We measured hash-vs-float in
  `scripts/bench_search.py`.)
- **Footprint.** The hash heads are ~MB. Indexing runs the backbone once per photo at
  import time; **query time only encodes the text** (small) + a Hamming scan.

Net: this makes a **fully offline, private, tiny** search index that runs on weak
hardware (an old phone, a Raspberry Pi) over a large library — the part the walled
gardens don't offer openly.

### Edges worth leaning on

1. **Offline + private + open-source.** No cloud upload (unlike Google Photos), not
   locked to an ecosystem (unlike Apple), self-hostable.
2. **Multilingual / Korean.** SigLIP2 is multilingual — Korean queries like
   "졸려하는 강아지" work natively. Apple/Google Korean NL search is comparatively weak.
   This is a concrete, demoable edge.
3. **Scales cheap.** The 128-bytes/photo + XOR story is the visceral "wow" (search a
   million photos instantly on a laptop).

### Honest tradeoffs (don't oversell)

- **Quality cost.** Binary hashing *loses* accuracy vs float (our R@10 1024-bit ≈ 0.75
  vs float ≈ 0.81). For a consumer "find the right photo" app this matters. Design
  options: use float for top-quality on small libraries and switch to hashing for
  scale, or present hashing as the "millions-of-photos, on-device" tier.
- **"태기" = a specific named pet** is *personalization*, not zero-shot retrieval.
  Zero-shot handles attributes/scenes ("sleepy dog") well; recognizing *your* dog
  specifically needs an enrollment step (few-shot prototype, or a pet/face cluster).
  Worth building as a follow-on feature — and a genuine differentiator if done.

### MVP tiers

1. **Web demo (fastest "wow", builds on what we have).** Point it at a folder →
   index with SigLIP2 → a search box → results + a live "128 B/photo, scanned 100k in
   X ms" stat panel. We already have `src/serve/binary_index.py` + a retrieval API +
   the monitor UI to extend.
2. **Self-hosted service.** A search microservice over a photo library (immich-style
   plugin / standalone).
3. **On-device mobile (the real vision).** iOS/Android, on-device indexing + offline
   NL search. Hardest part is shipping the encoder on-device (CoreML/TFLite export, or
   a smaller SigLIP variant). Index + search are trivially light once codes exist.

**Recommended start:** the **web demo** — it directly showcases the efficiency story,
reuses our serving code, and is the quickest path to something impressive. Mobile is
the eventual product; the demo de-risks and sells it first.
