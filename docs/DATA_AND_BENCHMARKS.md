# Data & Benchmark Validation

History and plan for the datasets used to train/evaluate the cross-modal hashing
model, plus how we validate that the frozen **SigLIP2** backbone reproduces the
published retrieval numbers.

_Last updated: 2026-06-14._

---

## 1. Why this exists

The model hashes SigLIP2 image/text embeddings into binary codes. Every downstream
quality number therefore rests on two assumptions:

1. **Our embedding extraction is correct** — i.e. the raw SigLIP2 backbone, as we
   call it, reproduces the retrieval numbers Google published. If it doesn't, the
   hashing results are built on a broken foundation.
2. **Train and eval data do not overlap** — see [§5 Data integrity](#5-data-integrity).

This document records the validation, the datasets we use, and how to reproduce both.

---

## 2. Benchmark validation — does SigLIP2 reproduce the paper?

**Model:** `google/siglip2-so400m-patch14-384`
**Published (arXiv 2502.14786, Table 1 — paper reports R@1 only):**

| Benchmark | T→I R@1 | I→T R@1 |
|---|---|---|
| COCO-5K (Karpathy test) | **55.8** | **71.7** |
| Flickr30K-1K | **85.7** | **94.9** |
| ImageNet-1k zero-shot top-1 | **84.1%** | — |

Protocol: zero-shot, 5 captions/image (COCO = 25,010 captions, Flickr = 5,000),
L2-normalized cosine, **MAP-head `pooler_output`** as the embedding.

### Measured (our pipeline)

**fp32 + official `AutoProcessor` (bicubic)** — `scripts/zeroshot_benchmark.py`:

| Benchmark | Dir | R@1 | R@5 | R@10 | Published R@1 | Δ |
|---|---|---|---|---|---|---|
| COCO-5K | T→I | 52.1 | 75.3 | 83.1 | 55.8 | −3.7 |
| COCO-5K | I→T | 67.3 | 87.9 | 92.9 | 71.7 | −4.4 |
| Flickr30K-1K | T→I | 79.0 | 94.2 | 97.0 | 85.7 | −6.7 |
| Flickr30K-1K | I→T | 92.9 | 99.3 | 99.9 | 94.9 | −2.0 |

For reference, the **bf16 cached-embedding** path (albumentations bilinear resize) gives
COCO T→I/I→T = 52.1 / 68.0 and Flickr 79.1 / 92.7 — **almost identical to fp32**.

**Conclusion:** the pipeline is correct — pooling (MAP-head `pooler_output`), L2-norm,
and the 5-caption protocol all match, reproducing **~93% of the published R@1** stably.
**But it is NOT a 1:1 reproduction** — a consistent ~3.7-pt gap remains.

> ⚠️ **Correction (verified 2026-06-14):** an earlier hypothesis blamed the gap on
> **bf16 + bilinear resize**. That was **wrong** — fp32 + the official bicubic processor
> gives essentially the same numbers as the bf16/bilinear cache (Δ < 0.8 pt). dtype and
> resize are **not** the cause. The residual gap is most likely a subtle **evaluation
> protocol / preprocessing difference** vs. Google's `big_vision` reference eval (e.g.
> exact text handling or image preprocessing details). Closing it 1:1 would require
> diffing against the original `big_vision` SigLIP2 eval code — a separate task.

The hashing results below are measured **relative to this validated float baseline**, so
the float-vs-hash comparisons are sound regardless of the absolute gap to the paper.

### Implementation notes / gotchas

- Use `backbone.vision_model(...).pooler_output` and `backbone.text_model(...).pooler_output`.
  **Do NOT use `model.get_image_features()`** — in the current `transformers` version it
  returns a `BaseModelOutputWithPooling` object, not a tensor. The repo's `_pool()` in
  `src/models/cross_modal_hash.py` is the canonical path.
- SigLIP has no separate projection head; the attention-pool output *is* the embedding.
- Text must be tokenized with `padding="max_length", max_length=64` (SigLIP training setting).

### How to run

```bash
# COCO-5K straight from the Karpathy json (5-caption protocol)
python scripts/zeroshot_benchmark.py \
  --karpathy data/coco/dataset_coco.json --split test --data-root data/coco \
  --dtype fp32 --published T2I:55.8,I2T:71.7 --out bench_coco.json

# Flickr30K-1K from the exported jsonl
python scripts/zeroshot_benchmark.py \
  --jsonl data/flickr30k/flickr30k_test.jsonl --data-root data/flickr30k \
  --dtype fp32 --published T2I:85.7,I2T:94.9 --out bench_flickr.json
```

`scripts/zeroshot_benchmark.py` works for COCO (`--karpathy`) or any
`{image_path, captions}` jsonl (`--jsonl`), and prints a MATCH/CLOSE/OFF verdict per
direction against the `--published` targets.

---

## 3. Datasets

### In use

| Dataset | Split sizes | Role | Notes |
|---|---|---|---|
| **COCO Captions (Karpathy)** | train+restval 113,287 / val 5,000 / test 5,000 | train + eval | `data/coco/dataset_coco.json`; images span `train2014/` + `val2014/` (use the `filepath` field) |

### Added 2026-06-14 (higher-quality + more diverse)

| Dataset | Split sizes | Role | Caption quality | License | On-disk |
|---|---|---|---|---|---|
| **Flickr30K** | train 29,000 / val 1,014 / **test 1,000** | eval benchmark (+ train aug) | 5 human captions/image | research | 4.4 GB |
| **DOCCI** | train 9,647 / test 5,000 | train aug | **~136-word dense human captions** | CC BY 4.0 | 15 GB |

- **Flickr30K** is the standard *second* retrieval benchmark — every cross-modal
  hashing paper reports COCO **and** Flickr30K R@1/5/10.
- **DOCCI** brings long, precise descriptions that push the hash heads to encode
  finer-grained visual detail than COCO's short captions.

### Fetch method — `datasets 5.0.0` gotcha

`datasets >= 4` **dropped loading-script support**, so `load_dataset("nlphuji/flickr30k")`,
`load_dataset("google/docci")`, and `HuggingFaceM4/NoCaps` all fail with
`RuntimeError: Dataset scripts are no longer supported`. **Do not downgrade `datasets`
on a shared venv** while training runs. Fetch **script-free** instead:

- **Flickr30K** — pull raw repo files and assemble locally:
  `hf_hub_download("nlphuji/flickr30k", "flickr30k-images.zip" / "flickr_annotations_30k.csv", repo_type="dataset")`.
  The CSV columns are `raw` (a list-as-string of 5 captions), `split`, `filename`, `img_id`.
- **DOCCI** — authoritative GCS tarball (no HF):
  `https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines` (11 MB) +
  `docci_images.tar.gz` (7.6 GB). Description lines: `{example_id, split, image_file, description}`.

Both are exported to `data/<name>/images/*` + `data/<name>/<name>_<split>.jsonl` by
`scripts/fetch_extra_datasets.py`:

```bash
python scripts/fetch_extra_datasets.py --data-root data --datasets flickr30k docci
```

---

## 4. Data usage plan — how extra data enters training

The repo already supports mixing extra datasets via the `data.extra_datasets` config
key, which builds `GenericImageTextDataset` instances and `ConcatDataset`s them onto the
COCO base. Each entry needs a `jsonl_path` ({`image_path`, `caption`|`captions`}) and a
`data_root` that `image_path` is relative to.

`configs/experiment_extra_data.yaml` wires the new English data in (frozen backbone, so
only the small hash heads train — adding data is cheap):

```yaml
data:
  data_root: "data/coco"
  karpathy_json: "data/coco/dataset_coco.json"
  extra_datasets:
    - jsonl_path: "data/docci/docci_train.jsonl"
      data_root: "data/docci/images"        # DOCCI image_file is a BARE filename
    - jsonl_path: "data/flickr30k/flickr30k_train.jsonl"
      data_root: "data/flickr30k"           # flickr image_path is "images/<file>"
```

> ⚠ **DOCCI path gotcha:** DOCCI's `image_file` is a bare filename (`train_00000.jpg`)
> while images extract to `data/docci/images/`, so `data_root` must be
> `data/docci/images`. Flickr30K's `image_path` already includes `images/`, so its
> `data_root` is `data/flickr30k`. Both were verified end-to-end through
> `GenericImageTextDataset` (correct tensors, all augmentation views).

**Eval** still uses COCO Karpathy test (5K). Validate the raw backbone separately with
`scripts/zeroshot_benchmark.py` (§2).

---

## 5. Data integrity

- **COCO-Korean leakage:** `data/coco_ko/coco_ko.jsonl` pairs **all** COCO images,
  including the 5,000 Karpathy test + 5,000 val images → training on it leaks the eval
  set. Use the filtered `data/coco_ko/coco_ko_train.jsonl` (113,287 lines, leakage = 0).
  Past hash-quality numbers measured against the leaked file were upward-biased; the
  float/speed/bit-ordering findings were unaffected.
- Flickr30K/DOCCI are disjoint image sets from COCO, so no cross-leakage with the COCO
  eval split.

---

## 6. Candidate datasets (future)

From a survey of standard / high-quality / diverse image-text sources. Eval = small,
standard, for measuring retrieval; Train = larger, to improve the model.

| Dataset | Imgs | Caption | Role | Access | License |
|---|---|---|---|---|---|
| nocaps | 15.1K | 11 human/img, novel objects | eval (out-of-domain) | `HuggingFaceM4/NoCaps` (script — needs direct Open Images fetch) | CC BY 4.0 |
| XM3600 | 3.6K | 36 languages | eval (multilingual) | GCS images.tgz + captions.zip (320 MB) | CC BY 4.0 |
| ShareGPT4V | ~100K | GPT-4V dense (~200–400 w) | train aug | `Lin-Chen/ShareGPT4V` (images = COCO train2017) | CC-BY-NC |
| Localized Narratives | ~849K | spoken, grounded | train aug | `HuggingFaceM4/LocalizedNarratives` (annotations only) | CC BY 4.0 |
| CC3M | ~2.9M | web alt-text (noisy) | train aug | `pixparse/cc3m-wds` (281 GB) | permissive |
| Recap-DataComp-1B | ~941M | LLaVA recaptions | train aug (stream a scored subset) | `UCSC-VLAA/Recap-DataComp-1B` | CC BY 4.0 |

Recommended next adds: **nocaps** + **XM3600** as eval benchmarks (small, standard,
multilingual); **ShareGPT4V** for dense-caption training at COCO scale.

---

## 7. Hashing results — full 113K + augmentation (2026-06-14)

Trained 4 nested-code options (frozen SigLIP2, 40 epochs each, on the full 113K
train+restval with 3-view augmentation). Evaluated on COCO Karpathy test 5K.
Speed/size measured on a 50K corpus, 1K queries (numpy popcount vs float cosine).

**opt-1024 (codes [8…1024]) — quality + size + speed by code length:**

| Code | I2T R@10 | T2I R@10 | mAP | vs float | Index size | Search QPS | ms |
|---|---|---|---|---|---|---|---|
| 64-bit | 0.641 | 0.633 | 0.371 | 79% | 0.38 MB | 25,355 | 0.04 |
| 128-bit | 0.729 | 0.718 | 0.448 | 90% | 0.76 MB | 27,667 | 0.04 |
| 256-bit | 0.759 | 0.764 | 0.491 | 94% | 1.53 MB | 13,346 | 0.07 |
| 512-bit | 0.787 | 0.787 | 0.517 | 97% | 3.05 MB | 6,357 | 0.16 |
| **1024-bit** | **0.798** | **0.800** | 0.531 | **98%** | 6.10 MB | 3,388 | 0.30 |
| float | 0.811 | 0.816 | 0.589 | 100% | 219.73 MB | 618 | 1.62 |

**Headline:** 1024-bit binary keeps **98% of float R@10** at **1/36 the index size** and
**5.5× the search speed**. 256-bit is the balanced sweet spot (94% quality, 1.53 MB).

Nested structure works: short-code quality is stable across options (64-bit ≈ 0.63–0.64
in every option), so adding longer codes does not hurt shorter ones. Full-data + aug
improved over the earlier 40K-clean run (1024-bit R@10 0.752 → 0.798).

> Note on speed: per-code ms is not perfectly monotonic (measurement noise on a small
> 1K-query numpy-popcount benchmark). The robust takeaway is the float-vs-hash gap
> (all hash options ≪ float in size and ≫ float in speed), not bit-to-bit ms deltas.

Visualized in `claudedocs/retrieval_compare.html` (per-option tabs, quality/speed/size
matrix, the §2 benchmark-validation panel, and real text→image search galleries).

---

## 8. Reproduction quick-reference

```bash
# 1. Fetch extra datasets (script-free)
python scripts/fetch_extra_datasets.py --data-root data --datasets flickr30k docci

# 2. Validate the raw backbone vs published numbers
python scripts/zeroshot_benchmark.py --karpathy data/coco/dataset_coco.json \
  --split test --data-root data/coco --dtype fp32 --published T2I:55.8,I2T:71.7
python scripts/zeroshot_benchmark.py --jsonl data/flickr30k/flickr30k_test.jsonl \
  --data-root data/flickr30k --dtype fp32 --published T2I:85.7,I2T:94.9

# 3. Train with the extra high-quality/diverse data
python train.py --config configs/experiment_extra_data.yaml
```
