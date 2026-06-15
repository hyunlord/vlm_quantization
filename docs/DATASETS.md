# Datasets — inventory & plan

Two buckets, because they serve different purposes:

- **Image-only** → search **demo corpus** (text query → image). Captions not needed.
  Want large + diverse + nice-looking + permissive license.
- **Image-text pairs** → **training** (additional / larger-model). Want paired captions.

> ⚠️ **Embedding is the real bottleneck.** The encoder runs at ~3–12 img/s on the GB10
> (sm_121 fallback), so **local embedding is practical only up to ~150K images**
> (25K ≈ 2 h, 100K ≈ 9 h, 1M ≈ tens of hours → needs a faster GPU). Datasets can be
> *downloaded* ahead of time cheaply; *embedding* them is what's gated. See
> [[hp-tuning-results]] / [[dgx-spark-access]] for the box.

## Folder layout (on DGX `data/`)

```
data/
  coco/        20G  image-text  COCO Karpathy (train 113K / val 5K / test 5K) — PRIMARY train+eval
  flickr30k/  4.4G  image-text  Flickr30K (train 29K / test 1K) — eval benchmark + train aug
  docci/       15G  image-text  DOCCI 9.6K — dense human captions, train aug
  coco_ko/    208M  image-text  Korean COCO — use coco_ko_train.jsonl ONLY (leakage; see [[coco-ko-data-leakage]])
  image_only/       image-only  demo corpus sources (NEW)
    unsplash25k/    Unsplash Lite 25K (jamescalam/unsplash-25k-photos)
  image_text/       image-text  extra training/eval (NEW)
    nocaps/         nocaps 15K (HuggingFaceM4/NoCaps) — eval/val
```
(Existing coco/flickr30k/docci are image-text but kept at their current paths to avoid
breaking code/cache references; classified here in docs instead of moving them.)

## Bucket A — image-only (demo corpus)

| Dataset | # imgs | Download | License | GB10 embed |
|---|---|---|---|---|
| **Unsplash Lite** `jamescalam/unsplash-25k-photos` | 25K | parquet (images embedded) — `load_dataset(...).save_to_disk` | Unsplash License | ~2 h ✅ |
| Open Images V7 val | 42K | `aws s3 --no-sign-request sync s3://open-images-dataset/validation` | CC BY 2.0 (per-image) | ~4 h |
| Open Images V7 train subset | N | FiftyOne `max_samples=N` or CVDF downloader | CC BY 2.0 | 100K = overnight |
| Unsplash Full | 6.5M | request form, TSV of CDN URLs | Unsplash (non-commercial) | impractical here |

## Bucket B — image-text pairs (training)

| Dataset | # pairs | Download | License | Notes |
|---|---|---|---|---|
| COCO (have) | 113K | `data/coco` | CC BY 4.0 | primary train+eval |
| Flickr30K (have) | 31K | `data/flickr30k` | research | benchmark + aug |
| DOCCI (have) | 15K | `data/docci` | CC BY 4.0 | dense captions |
| **nocaps** `HuggingFaceM4/NoCaps` | 15K | parquet (works on datasets≥4) | CC BY 4.0 | eval/val |
| CC3M `pixparse/cc3m-wds` | 2.9M | WebDataset tars / img2dataset | CC (Google) | stream a 100–200K subset for GB10 |
| Recap-DataComp-1B `UCSC-VLAA/Recap-DataComp-1B` | 941M | parquet (URL-only) + img2dataset | CC BY 4.0 | GPT4V recaptions; filter `re_gpt4v_score≥3` |
| COYO-700M `kakaobrain/coyo-700m` | 747M | parquet (URL-only) + img2dataset | CC BY 4.0 | scored; filter by CLIP/aesthetic |
| LAION-2B-en-aesthetic | ~100–600M | parquet (URL-only) | research | aesthetic subset |

> `datasets ≥ 4` dropped loading scripts. nocaps now works (auto-parquet). Flickr30K/DOCCI
> were fetched script-free (see [[siglip2-benchmark-validation]] / `scripts/fetch_extra_datasets.py`).
> CC3M/Recap/COYO/DataComp are parquet (URL-only) → use `img2dataset` to fetch images.

## Starter set (being downloaded)

```bash
# image-only demo corpus
load_dataset("jamescalam/unsplash-25k-photos")  -> data/image_only/unsplash25k
# image-text eval
load_dataset("HuggingFaceM4/NoCaps")            -> data/image_text/nocaps
```

## Demo corpus plan

- **Now (already embedded):** COCO **train 113K** (`emb_aug.pt` clean view) → real 113K
  unique corpus, no tiling. Far more honest than the prior 5K×200 tiled 1M.
- **Add (after embedding ~2 h):** Unsplash 25K for visual variety.
- **Big (needs faster GPU):** Open Images / CC3M subset for a true large corpus.

## Embedding-time reality check

| Images | @3 img/s | @12 img/s | Verdict |
|---|---|---|---|
| 25K | 2.3 h | 35 min | starter ✅ |
| 113K (have embeds) | — | — | instant (cached) |
| 100K (new) | 9 h | 2.3 h | overnight |
| 1M+ | days | ~1 day | DGX/faster GPU only |
