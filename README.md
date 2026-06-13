# VLM Quantization — Cross-Modal Deep Hashing with SigLIP2

Dual-encoder cross-modal deep hashing. Images and text are projected into
**prefix-nested (Matryoshka) binary codes** at multiple bit lengths
`[8, 16, 32, 48, 64, 128]`, enabling coarse-to-fine retrieval via Hamming
distance (XOR + popcount). The backbone is
[SigLIP2 So400m](https://huggingface.co/google/siglip2-so400m-patch14-384),
with Korean / multilingual retrieval as a target use case
(COCO Korean, AIHub #71454, CC3M-Ko).

The project ships a FastAPI + SQLite + Next.js real-time training dashboard
and an Optuna hyperparameter search, and runs on three environments:
Colab (A100), DGX Spark (GB10 / ARM), and local CPU dev.

## Architecture

```
            image ─► SigLIP2 vision_model ─┐
                                           ├─► NestedHashLayer ─► {continuous, binary} per bit
            text  ─► SigLIP2 text_model  ──┘     (prefix slice → per-bit BatchNorm → L2norm
                                                  → tanh / SignSTE)
                                                          │
                                                          ▼
                                                  CombinedHashLoss
```

`CombinedHashLoss` aggregates, per bit length:

- **InfoNCE** cross-modal contrastive (optionally *focal*, see below)
- **OrthoHash** cross-modal orthogonality
- **EAQL** equilibrium-aware quantization (EMA-weighted, ramped)
- **BitBalance** balance + decorrelation
- **Consistency** (augmented-image alignment)

and globally:

- **LCS** long-to-short self-distillation across bit lengths
- **Supervised** pairwise loss (optional, requires COCO category labels)

### Optional features (default-off, opt-in via config)

| Feature | Config key | Default | Effect |
|---|---|---|---|
| Focal InfoNCE | `loss.focal_gamma` | `0.0` | `> 0` down-weights easy positives by `(1 - p)**gamma`; `0` is exactly plain InfoNCE |
| Multi-caption | `data.num_captions` | `1` | `2` adds a second distinct caption as an extra contrastive positive |
| Text dropout | `data.text_dropout_prob` | `0.0` | `> 0` randomly drops tokens for text augmentation |
| Supervised | `loss.supervised_weight` | `0.0` | `> 0` enables pairwise label loss (needs `data.instances_json`) |

With all defaults, training is numerically identical to the base model.

## Setup

```bash
pip install -e ".[dev]"        # runtime + pytest/ruff
# or, runtime only:
pip install -e .
```

Download the Karpathy split (`dataset_coco.json`) from
[cs.stanford.edu/people/karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
into `data/coco/`, alongside the COCO `train2014/` and `val2014/` images.

## Training

```bash
python train.py --config configs/default.yaml          # local / generic
python train.py --config configs/colab.yaml            # Colab A100 (auto batch size)
python train.py --config configs/dgx_spark.yaml        # DGX Spark (GB10 / ARM)
python train.py --config configs/colab_multilingual.yaml
```

Batch size, gradient accumulation, and worker count auto-configure to GPU VRAM
when `training.batch_size: auto`.

## Evaluation

```bash
python eval.py --checkpoint path/to/best.ckpt --config configs/default.yaml
```

Reports, per bit length, bit entropy and cross-modal retrieval metrics
(**mAP**, **P@k**, **R@k**) for both I2T and T2I directions. During training,
validation additionally logs a frozen-backbone cosine-similarity mAP baseline
for an honest comparison against the learned hash codes.

## Monitoring dashboard

```bash
# backend (FastAPI + SQLite)
python -m monitor.server
# frontend (Next.js)
cd monitor/frontend && npm install && npm run dev
```

Enable `monitor.enabled: true` in the config to stream live training/eval
metrics, per-bit activation rates, and augmentation-robustness samples.

## Hyperparameter search (Optuna)

```bash
python optuna_search.py --config configs/colab.yaml --n-trials 50 \
    --search-epochs 5 --subset-ratio 0.1
```

Searches the parameters the model actually accepts (hidden dim, loss weights,
temperature, learning rates, `focal_gamma`, `text_dropout_prob`), prunes weak
trials with a median pruner, and exports the best config as YAML.

## Tests

```bash
pytest tests/ -q       # 31 CPU-only tests
ruff check src/ tests/ eval.py train.py optuna_search.py monitor/callback.py
```

Tests cover Hamming distance, retrieval metrics (mAP / P@k / R@k / bit entropy)
including a brute-force equivalence check for the vectorized implementations,
loss components, the SignSTE straight-through estimator, the EAQL eval-mode
buffer guard, the NestedHashLayer prefix property, focal-InfoNCE / aux-caption
wiring, and a backbone-free integration smoke test (NestedHashLayer →
CombinedHashLoss → backward → optimizer step). They run on CPU and do not
download the backbone. CI runs both on every push and pull request.

Training is seeded (`seed: 42`, overridable per config) via
`pl.seed_everything(..., workers=True)` for reproducible runs.
