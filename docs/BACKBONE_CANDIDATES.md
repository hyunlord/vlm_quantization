# Backbone Candidates — replacing / upgrading SigLIP2 So400m

Survey (as of **June 2026**) of dual-encoder image-text models that could replace the
current backbone (`google/siglip2-so400m-patch14-384`) for the cross-modal hashing
pipeline, with emphasis on **larger/newer** models and **Korean/multilingual** quality.

## Why swapping is cheap here
The backbone is used **frozen** (`freeze_backbone: true`). Swapping it only requires
**retraining the small per-modality hash heads** — not the whole model — so a backbone
A/B is fast. `embed_dim` is **auto-detected** from the backbone config, so most
SigLIP/CLIP-family models are near drop-in.

## Compatibility constraint
The code calls the HF **dual-tower API**: `backbone.vision_model(pixel_values=…)` and
`backbone.text_model(input_ids=…, attention_mask=…)`, reading `pooler_output`
(`src/models/cross_modal_hash.py`). So:
- **SigLIP2 / CLIP-family (incl. MetaCLIP 2)** → drop-in or near drop-in.
- **Jina CLIP / EVA-CLIP** → custom processor/`get_*_features` → moderate code change.
- **LLM/VLM-based embedders (Jina v4, Kakao Kanana-v)** → single-tower, instruction-based → major rework.

## Candidates

| Model | HF id | ~Size / dim | Multilingual (KO) | Compat | Notes |
|---|---|---|---|---|---|
| **SigLIP2 So400m** (current) | `google/siglip2-so400m-patch14-384` | 0.9B / ~1152, 384px | yes (trained ML) | ✅ baseline | Still rated **strongest open image-text model (Jun 2026)**. Good, hard to beat. |
| **SigLIP2 giant** | `google/siglip2-giant-opt-patch16-384` | ~1.8B / ~1536, 384px | yes | ✅ true drop-in | Larger ceiling, but community reports it **does not clearly improve retrieval** over so400m — measure before committing. |
| **MetaCLIP 2 (worldwide)** | `facebook/metaclip-2-worldwide-huge-378` | ~1B / ViT-H-14, 378px | **300+ langs** (strong KO) | ✅ near drop-in (set `image_size: 378`) | NeurIPS 2025. Beats mSigLIP on multilingual zero-shot. **Best drop-in for Korean.** |
| **Jina CLIP v2** | `jinaai/jina-clip-v2` | 0.9B / 1024 (MRL) | **89 langs** (incl. KO) | ⚠️ moderate (custom proc / `trust_remote_code`) | Retrieval-optimized, SOTA Flickr30k, Matryoshka-native dims. Strong KO; needs encode/processor changes. |
| **Jina Embeddings v4** | `jinaai/jina-embeddings-v4` | 3B (Qwen2.5-VL) | multilingual | 🔴 major rework | Universal multimodal; single VLM tower, instruction-based. Heavy. |
| **Kakao Kanana-v / 1.5-o** | `kakaocorp/*` | 2–10B (LMM) | **Korean-optimized** | 🔴 major rework | Korean-first multimodal, but LMM API. Korean-multimodal embedding (`Kanana-v-embedding`) weights availability unclear. |

## Recommendation
1. **Don't assume bigger/newer wins.** SigLIP2 So400m is still SOTA-class open for
   image-text (Jun 2026), and SigLIP2-giant reportedly doesn't help retrieval. Treat
   any swap as a hypothesis to **measure**, not an upgrade by default.
2. **For the Korean target → try `MetaCLIP 2 worldwide-huge-378` first.** It's a near
   drop-in (CLIP API), explicitly trained on 300+ languages, and the most likely to beat
   SigLIP2 on Korean retrieval. Set `image_size: 378`.
3. **If willing to write some code → `jina-clip-v2`** is the strongest Korean/retrieval
   specialist; budget time for the custom processor + `get_image_features`/`get_text_features`.
4. **Raising the quality ceiling beyond any backbone's frozen embedding → unfreeze /
   fine-tune** the chosen backbone (`freeze_backbone: false`) — bigger lever than more bits.

## How to A/B (cheap, frozen backbone)
For each candidate: set `model.backbone` (+ `image_size`) in `configs/experiment_highbit.yaml`,
retrain the hash heads, then compare with the same harness:
```bash
python train.py --config configs/experiment_highbit.yaml         # frozen backbone → fast
python scripts/bench_search.py --index <built index> --bit 64    # size/speed/recall
# + the cross-modal eval (float vs per-bit mAP/R@K) used for the comparison table
```
Decision metric: **does the new backbone's frozen-cosine baseline (mAP) beat SigLIP2's?**
If the float ceiling doesn't move, the hash codes won't either.

## Sources
- SigLIP 2 (HF blog): https://huggingface.co/blog/siglip2
- SigLIP2 giant: https://huggingface.co/google/siglip2-giant-opt-patch16-384
- "SigLIP-2 strongest open image-text as of Jun 2026": https://www.spheron.network/blog/multimodal-embedding-models-gpu-cloud-siglip2-jinaclip-cohere/
- MetaCLIP 2 (paper/HF): https://huggingface.co/papers/2507.22062 · https://huggingface.co/facebook/metaclip-2-worldwide-huge-378
- Jina CLIP v2: https://jina.ai/models/jina-clip-v2/ · https://arxiv.org/html/2412.08802v1
- Embedding models 2026 roundups: https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models · https://milvus.io/blog/choose-embedding-model-rag-2026.md
- Kakao Kanana (multimodal, Korean): https://tech.kakao.com/posts/801 · https://huggingface.co/kakaocorp
