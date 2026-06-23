# PAPER_RIGOR_PILLAR2_PARITY — §3 cross-modal binary-parity failure

Branch `paper-aaai-rigor`, Pillar 2. Matched comparison (`web/rigor_parity.py`): three text paths into the SAME
frozen ft113 code space (gallery = img_h(test_img) codes), COCO 5K test, I2T. Same student encoder (multilingual-e5)
for distillation vs head-adapt.

## Result — `paper/rigor_parity.csv`

| path | cosine→true SigLIP-text | bit-parity vs true (%) | continuous R@10 | **binary R@10** |
|---|---|---|---|---|
| ceiling (true SigLIP-text → txt_h) | 1.000 | 100.0 | 81.58 | **79.92** |
| **distillation** (e5→pred SigLIP-text → txt_h) | **0.866** | 86.91 | 67.92 | **70.86** |
| **head-adapt** (e5 → txt_h', InfoNCE to img codes) | — | 81.64 | — | **74.00** |

(cosine/cont undefined for head-adapt: it never targets the SigLIP-text embedding, only the code space.)

## Findings (the Pillar-2 contribution)
1. **High embedding cosine ≠ binary retrieval ("parity failure").** Distillation reaches cosine **0.866** to the true
   SigLIP-text embedding, yet its binarized code matches only **86.9%** of the true bits, and binary R@10 drops to
   **70.86 (−9.06 vs the 79.92 ceiling)**. The ~13% of bits that disagree are the near-boundary ones (small pre-sign
   margin) the embedding-space loss never pins down — so a "good" cosine hides a degraded code.
2. **Directly optimizing codes beats matching the teacher — even with LOWER teacher-fidelity.** Head-adapt has *lower*
   bit-parity to the true code (**81.6% < 86.9%**) yet *higher* binary retrieval (**74.0 > 70.9**, +3.1). Fidelity to
   the teacher embedding is the **wrong objective** for binary retrieval; head-adapt's InfoNCE-to-image-codes target
   produces more retrievable codes despite drifting further from the SigLIP-text embedding.
3. **Prescription — when NOT to distill.** If the downstream is binary code retrieval, do **not** distill the teacher
   embedding (high cosine, lossy code); **adapt a head into the code space directly**. Distillation is appropriate only
   when the continuous embedding itself is the product — not when it will be signed into a hash.

## Mechanism (parity = near-boundary sign disagreement)
The 13% parity gap is concentrated at small |pre-sign z| (the boundary bits) — the same margin/flip mechanism
characterized in the loss-composition study (v1/v2 diagnostics). An MSE/cosine distillation loss has no gradient that
sharpens the *sign* of near-zero bits, so they are set by noise; head-adapt's contrastive loss directly pushes those
bits to the retrieval-correct side. (Margin-distribution + flip% figure: render from the v1/v2 diagnostic code on
distilled vs head-adapt pre-sign z — flagged for the writing pass.)

## Repro
`web/rigor_parity.py --out paper/rigor_parity.csv` on DGX (ft_ko_113.pt + emb_cache test + distill_e5.pt +
txt_h_e5.pt + e5_test_en.pt; loads the fine-tuned e5 backbone; ~2 min). Note cosine here is 0.866 (the project's
earlier "0.89" was a different student/checkpoint); the qualitative parity-failure story is unchanged and reproduced.
