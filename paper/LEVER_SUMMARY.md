# Deep-Lever Multi-Gate — One-Page Summary (1차군: A, B)

**Setup**: head-only on cached SigLIP2 embeddings, mode=coco, EPOCHS=15 (baseline `bn-infonce@25`=80.88@1024 reproduces deployed 80.3 → harness faithful). Metric = COCO 5K instance EN T2I R@10, ΔR@10 vs `bn-infonce` baseline, bit-sweep {64,128,256,512,1024}, 2 seeds (noise ≤0.3pt). GREEN = +1.0pt over ≥3 bits beyond noise.

| Lever | best ΔR@10 (bit) | across-bit pattern | gate | one-line reason |
|---|---|---|---|---|
| **A** negsep loss | **+0.69** (1024) | −6.4/−2.6/−1.0/+0.2/+0.7 | **RED** | pure hmargin widens separation (mechanism ✓) but only re-trades top-1↔recall along bit budget; InfoNCE already negsep-optimal for R@10 |
| **A** infonce+hmargin/hardneg | +0.02 (256) | flat/negative | **RED** | InfoNCE subsumes the margin/hardneg term — inert |
| **B** BN-free (none) | +0.09 (64) | ≈0, −0.5 @1024 | **RED** | removing per-bit BN neither helps nor breaks (trains stably); BN not essential but no benefit |
| **B** LN / affine / rotation | +0.41 / −0.13 / −0.67 | ≈0; rotation worst | **RED** | any reasonable normalizer ≈ baseline; rotation slightly worse + breaks nesting |
| **B** composition under BN-free | spread 0.32 (inert) | — | **RED** | aux stays inert without BN → inertness is InfoNCE+L2 dominance, not BN |

**Both 1차 levers RED on the primary R@10 gate.** Honest, robust (2-seed; even vs stricter @25 baseline). Per the brief, RED *strengthens the analysis-paper narrative* — no 5-month loss.

### What the REDs establish (analysis-paper material)
1. **InfoNCE is negative-separation-optimal for R@10.** A binary Hamming-margin loss *does* widen the realized margin (−12→−3.4 bits) yet R@10 doesn't rise — it only sharpens top-1 at high bits (R@1 +4.2, KO R@10 +1.2 @1024) while killing low-bit recall (−5.7 @64). A capacity-dependent precision/recall re-trade, not a method.
2. **The per-bit BatchNorm is not the bottleneck and not essential.** LN/none/affine/rotation all land within ~0.5pt and train stably; composition stays inert *with or without* BN. The code geometry is fixed by InfoNCE alignment + L2, which any normalizer preserves. (Sharpens the prior loss-composition story, which had attributed inertness to BN.)

### Deployability (all 1차 variants)
Training-loss / head-norm only → **encoder & head unchanged at inference, zero deploy cost, ort-web/on-device clean.** The hmargin R@1/high-bit sharpening is a *free deployable side-effect* (a paragraph, not a method).

### Recommendation → deep lever to pursue
With both shallow-and-arch (1차) levers RED, the remaining untested deep lever is **C — backbone LoRA** (the brief's 2차군; first lever that actually moves the encoder, escaping the frozen+head-only regime that bounded everything so far). Bar is higher (+1.5pt) and it trades deployability (encoder grows). **Awaiting your call**: pursue **C (LoRA)**, or treat the RED/RED as the analysis-paper result (WACV/AAAI) and stop. D (task redesign, e.g. dynamic bit-allocation) is an identity change — discuss separately.

### Branches / SHAs
- `paper-lever-negsep` (Lever A): `web/PAPER_LEVER_A.md`, `paper/lever_A.csv`
- `paper-lever-arch` (Lever B): `web/PAPER_LEVER_B.md`, `paper/lever_B.csv`
- raw: `paper/lever_all.csv`, `paper/lever_report.txt`; harness `scripts/lever_sweep.py` (+`lever_run.sh`/`lever_run2.sh`/`lever_report.py`)
