# PAPER_LOSS_COMPOSITION — loss-composition gate (text head-adapt) · NEGATIVE result

Branch `paper-loss-composition`. Tests the AAAI main claim — *"in binary-code learning the **composition** of loss
terms (weighted-sum vs curriculum vs unified) dominates weight-tuning for accuracy AND robustness"* — on the
**text head-adapt front first (the gate)**. Per the work order, image + distillation proceed only if the text gate
passes. **It does not.** This is the anticipated honest negative, reported via the sign-margin mechanism.

## Setup (all head-only, encoders/SigLIP2 codes/index frozen)
- Anchor = frozen `ft113 img_h(SigLIP2 image emb)` 1024-bit nested codes (the exact head behind `index.bin`).
- Train `txt_h' = NestedHashLayer(e5-384 → hidden → 1024)` on 226,574 EN+KO COCO pairs (cached e5 + SigLIP2
  embeddings), 8 epochs, hp_results recipe. Eval COCO 5K test EN/KO, faiss binary R@{1,5,10}, Matryoshka bits
  {64,128,256,512,1024}. Driver `web/loss_composition.py` (`--sweep`), output `paper/loss_composition.csv` (115 rows).
- Three compositions on the same CombinedHashLoss (math verified):
  - **(W)** `align(InfoNCE on tanh(z/T)→tanh(anchor)) + wq·EAQL + wm·relu(m−z·b) + nesting(LCS)` — weight grid wq×wm = 3×3.
  - **(C)** curriculum: T annealed 4→1 (cosine), wq/wm/wu ramped over 60% of steps; T0∈{2,4}.
  - **(U)** unified: margin+quant fused into one squared sign-boundary term `wu·relu(m−z·b)²`; grid (wu,m_rel)=6 pts.
  - Best setting of each re-run at seeds 0,1 for noise.
- **Gate epochs = 8** (relative comparison; all comps identical conditions). A winning composition would be re-run at
  25 epochs for final paper numbers — not needed here since there is no winner.

## Result — R@10 EN by bit (seed-42 grid, min..max across each comp's settings)

| bit | W (min..max, **sensitivity range**) | C (min..max) | U (min..max) |
|----:|---|---|---|
| 64   | 53.28..53.36 (**0.08**) | 53.24..53.28 | 53.28..53.30 |
| 128  | 62.10..62.30 (**0.20**) | 62.18..62.20 | 62.06..62.08 |
| 256  | 68.00..68.20 (**0.20**) | 68.12        | 68.08..68.10 |
| 512  | 70.34..70.42 (**0.08**) | 70.36..70.38 | 70.32        |
| 1024 | 71.20..71.34 (**0.14**) | 71.36..71.42 | 71.24..71.26 |

Seed noise (best setting, seeds 42/0/1, R@10 EN @1024): W spread **0.36**, C **0.22**, U **0.40**.
Diagnostics @1024 (identical across all comps/weights): margin_mean **0.0250**, cos_pair **0.6512**, flip **2.17%**.

## Verdict: GATE FAILS (decisively, both criteria)
1. **C/U do not beat W** — all three tie within ~0.1–0.2 pt at every bit, i.e. **inside seed noise** (0.22–0.40 pt).
2. **No robustness gap to close** — W's weight-sensitivity range is **0.08–0.20 pt, *smaller* than seed noise**.
   W is already weight-insensitive, so "(C)/(U) markedly lower sensitivity" is not demonstrable — there is no
   sensitivity to lower.

## Mechanism (why composition is inert here — the honest deliverable)
The margin/quant/curriculum terms have **no effect on the binary code**: `margin_mean`, `cos_pair`, and `flip%` are
constant to 3–4 digits across all compositions and all weights. Cause is **structural to head-adapt**: `txt_h'` fits a
**frozen, already-binary anchor**, so (a) the per-bit **BatchNorm + L2-norm** fix the pre-sign scale/geometry and (b)
**InfoNCE alignment to the clean anchor codes does all the work** — there is no quantization tug-of-war left for a
margin/quant/curriculum term to resolve. The composition can only matter where the **binary structure is being
learned**, not inherited. This confirms and extends the earlier "loss ablation near-inert at 1024-bit ∵ head
BatchNorm" finding to the full bit-sweep and to the explicit W/C/U comparison.

## Recommendation (design rethink — protects the month's schedule)
- **Do NOT extend to the image / distillation fronts as specified.** They reuse the same frozen-anchor head-adapt
  setup and will be inert for the same structural reason.
- The claim is only testable where codes are **learned from scratch**: re-run W/C/U inside the **original SigLIP2 hash
  training** (the encoder+head jointly learn the 1024-bit space, no frozen anchor), where quant/margin actually shape
  the code geometry. That is the correct testbed for "composition dominates tuning" and is a different (heavier) cost
  profile than the head-only gate.
- Until that redesign, the negative is itself a contribution: *in adapt-to-fixed-code (deployment-time) hashing, loss
  composition and weight tuning are both inert — the anchor's geometry dominates.*

## Repro
`web/loss_composition.py --sweep --epochs 8 --out paper/loss_composition.csv` on DGX (`.venv/bin/python`, GPU; ~62 min,
incremental durable CSV). Grid emits C+U before the W grid so a partial run still yields the gate signal. Frozen inputs:
`ft_ko_113.pt` anchor, cached e5/SigLIP2 train embeddings, hp_results recipe.
