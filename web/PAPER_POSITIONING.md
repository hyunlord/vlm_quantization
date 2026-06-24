# PAPER_POSITIONING — prior-work contrast (AAAI main-track positioning)

Branch `paper-aaai-rigor`, section 0. Honest citation + contrast against the two most relevant recent single-loss
hashing works. **We claim no novelty on BN / simplification / lightweight heads** (pre-empted by OrthoHash and
HashCoder); we cite them and note our findings are *consistent* with theirs. Our novelty is the **cross-modal +
multilingual + on-device extension and the failure modes / deployment constraints that only appear there.**

## Sources
- **CroVCA / HashCoder** — "Cross-View Code Alignment", arXiv:2510.27584 (2025).
- **OrthoHash** — "One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective",
  NeurIPS 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/cbcb58ac2e496207586df2854b17995f-Abstract.html) ·
  [code](https://github.com/kamwoh/orthohash).

## Contrast table

| axis | OrthoHash (NeurIPS'21) | HashCoder / CroVCA (2025) | **Ours** |
|---|---|---|---|
| **modality** | unimodal image hashing | **unimodal cross-view** (foundation-model embeddings) | **cross-modal image↔text** |
| **multilingual** | no | no | **yes — 36-lang (XM3600), KO-focused** |
| **on-device / deploy** | no | no (server retrieval) | **yes — browser fp32-only, phone latency, client Hamming** |
| **eval metric** | category-mAP + instance retrieval (CIFAR/ImageNet) | retrieval mAP (COCO unsup, ImageNet100 sup) | **instance R@{1,5,10} + mAP@bit (COCO 5K), XM3600** |
| **loss** | single cosine-to-orthogonal-binary-target (CE form) | single BCE-align + coding-rate (anti-collapse) | InfoNCE + quant + margin + Matryoshka nesting (but **composition shown inert — §loss_composition v1/v2**) |
| **head** | CNN + BN code-balance layer | lightweight MLP probe on frozen emb (or LoRA) | NestedHashLayer (Matryoshka) probe on frozen SigLIP2 |
| **bits** | fixed (16–64 typical) | 16-bit highlighted | **nested 16→1024 (one model, prefix-sliced)** |
| **code balance** | **BatchNorm** | coding-rate regularizer | per-bit BatchNorm + L2 (same lever as OrthoHash) |

## Reading (for related-work + intro)
1. **BN/single-loss/lightweight-head are prior art.** OrthoHash already uses **BatchNorm for code balance** with a
   single objective; HashCoder already uses a single-loss MLP probe on frozen embeddings. Our v1/v2 loss-composition
   negatives (composition & weight-tuning inert; BN dominates the code geometry) are therefore **consistent with, not
   novel over, these works** — we cite them as the explanation and do **not** claim BN-dominance as a contribution.
2. **The open lane is cross-modal + multilingual + on-device.** HashCoder is explicitly unimodal cross-view; OrthoHash
   is unimodal image hashing. Neither evaluates image↔text retrieval, non-English text, or browser/phone deployment.
   Our contribution is to **extend single-loss frozen-feature hashing into that lane and characterize the failure
   modes that only arise there**:
   - on-device binary deployment science (precision/bit-flip sensitivity, runtime constraints) — Pillar 1;
   - cross-modal binary-parity failure of distillation (cosine high, R@K collapses) — Pillar 2;
   - multilingual binary robustness across 36 languages / non-Latin scripts — Pillar 3.
3. **apples-to-apples baseline (planned, §1):** a CroVCA-style head (BCE-align + coding-rate) trained on the *same*
   frozen SigLIP2 features in our pipeline, compared head-to-head with our NestedHashLayer at matched bits — so the
   "ours vs single-loss prior art" comparison is fair on our own cross-modal/multilingual eval (BN-dominance predicts
   near-parity on COCO instance; the value is what our analysis surfaces when extended to cross-modal + multilingual).

## Honesty notes
- No claim that BN-balance or single-loss simplicity is ours. Cited to OrthoHash/HashCoder.
- "HashCoder leaves cross-modal as future work" — the public abstract does **not** contain that exact sentence
  (it frames the method as cross-*view* on foundation embeddings); we will phrase as "HashCoder/CroVCA target the
  unimodal cross-view setting; cross-modal image–text hashing is outside their scope" rather than quoting a
  future-work claim we cannot verify. (Flagged for the writing pass — verify against the full PDF before final.)
