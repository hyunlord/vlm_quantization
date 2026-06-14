# Prior Art & Novelty Assessment

An honest survey of related research and products, to answer: **is this project novel,
or already done?** Short answer: every *ingredient* is established prior art, but one
specific *combination* appears unoccupied in the peer-reviewed literature and in shipping
products. The gap is real but narrow — enough for a workshop paper / strong open-source
project, not a paradigm shift.

_Last updated: 2026-06-14. Based on three independent literature/product sweeps._

---

## TL;DR

| Question | Answer |
|---|---|
| Is "binary embeddings at 32× compression" novel? | **No** — mainstream industry since 2024 (Cohere, Mixedbread/HF, Jina, Qdrant, Vespa). |
| Is "frozen VLM + MLP hash head for cross-modal retrieval" novel? | **No** — established field (DCMH 2017 → CLIPMH 2024, CroVCA 2025, EGATH NeurIPS 2024). |
| Is "Matryoshka nested float embeddings" novel? | **No** — MRL (NeurIPS 2022), in OpenAI/Nomic/Voyage products. |
| Is "Matryoshka **+ binary**" novel? | **Mostly no** — exists for **text** (QAMA CIKM 2025 learned; Vespa/Nomic post-hoc). |
| Is "**Matryoshka prefix-nested 1-bit binary codes for CROSS-MODAL image-text retrieval, end-to-end learned**" novel? | **Apparently yes** — no published paper or product found combining all of these. |
| Is "SigLIP2 as a hashing backbone" novel? | **Yes (minor)** — no hashing paper uses SigLIP2 yet; but "use a better encoder" is an implementation detail, not a concept. |

**Bottom line:** You're not the first to do binary cross-modal hashing, nor binary
embeddings, nor Matryoshka. But the **specific intersection** — *learned, prefix-nested,
1-bit, cross-modal, on a modern VLM* — is genuinely under-explored, and there is a
**real product whitespace** (on-device + binary + multilingual + open-source photo search).

---

## 1. The two paradigms (the key distinction)

A lot of "binary embedding" work is NOT the same thing as this project. The critical split:

**Paradigm A — Post-hoc binary quantization of frozen float embeddings** *(mainstream, text-dominant)*
A normal float model produces a float32 vector; you threshold at 0 to get bits. The model
was never trained for binary. Works OK for text (~90–96% retention with rescoring).
Providers: Cohere int8/binary, Mixedbread+HF, Jina, Qdrant BQ, Weaviate BQ, Vespa.
**Crucial finding:** Nomic reported this **fails for cross-modal vision-text** —
"poor results with both Matryoshka and BQ on the vision embeddings" because LiT training
leaves the vision/text geometries misaligned for thresholding.

**Paradigm B — End-to-end *learned* cross-modal binary hashing** *(academic niche, this project)*
Training forces binary output directly (Sign-STE), with quantization + balance + alignment
losses. Binary codes are the intended output, not a lossy afterthought. This is why it
works for cross-modal where post-hoc fails.
**This project is squarely Paradigm B.**

---

## 2. Academic prior art (cross-modal hashing)

A mature field. The "frozen VLM + lightweight hash head + Hamming search" pattern is now standard.

| Paper | Venue / Year | arXiv | Relevance |
|---|---|---|---|
| DCMH — Deep Cross-Modal Hashing | CVPR 2017 | 1602.02255 | Foundational: dual-encoder → joint binary codes |
| SSAH — Self-Supervised Adversarial Hashing | CVPR 2018 | 1804.01223 | Adversarial cross-modal hashing |
| CMHH — Cross-Modal Hamming Hashing | ECCV 2018 | — | Focal loss on Hamming distance |
| MRL — Matryoshka Representation Learning | NeurIPS 2022 | 2205.13147 | Prefix-nested **float** embeddings (the nesting idea) |
| CLIPMH — CLIP Multi-modal Hashing | ICASSP 2024 | 2308.11797 | **Close**: CLIP + MLP hash head, image-text, +8.38% mAP |
| EGATH — Graph-Attention Cross-Modal Hashing | NeurIPS 2024 | — | Learned CLIP-based binary cross-modal hashing, SOTA on COCO |
| LCDH — Lightweight Contrastive Distilled Hashing | AAAI 2025 | 2502.19751 | CLIP teacher → light student hashing |
| PromptHash | CVPR 2025 | 2503.16064 | CLIP-based cross-modal hashing |
| CroVCA / HashCoder | arXiv 2025 | 2510.27584 | **Closest (hashing side)**: frozen foundation model + MLP hash head, binary, trains in 5 epochs — **but no Matryoshka nesting** |
| QAMA — Quantization-Aware Matryoshka Adaptation | CIKM 2025 | ACM 3746252.3761077 | **Closest (Matryoshka side)**: MRL + learned quantization + Hamming — **but 2-bit min, TEXT-only** |
| SigLIP 2 | arXiv 2025 | 2502.14786 | The backbone here; no hashing paper built on it yet |

**The two bracketing works:**
- **CroVCA (2510.27584)** = our architecture *without* Matryoshka nesting.
- **QAMA (CIKM 2025)** = Matryoshka + quantization + Hamming, but *text-only and 2-bit*, not 1-bit cross-modal.

No paper found sits in the middle (nested 1-bit + cross-modal + learned).

---

## 3. Industry binary-embedding landscape (2023–2025)

| Provider | Modality | Binary? | Matryoshka? | Post-hoc / Learned | Claimed |
|---|---|---|---|---|---|
| Cohere Embed v3 | text | yes | no | post-hoc | 32×, 90–98% recall |
| Mixedbread + HF | text | yes | composable | post-hoc | 32×, ~96% NDCG w/ rescore |
| Jina v2/v3 | text | yes | no | semi-learned | 32×, ~90% |
| Nomic v1.5 | text | yes | yes | post-hoc | 95.8% @3× |
| Nomic vision v1.5 | image-text | **failed** | failed | post-hoc | "poor results" (LiT) |
| Voyage code-3 | text | yes | yes | post-hoc | up to 200× combined |
| OpenAI text-embedding-3 | text | no (3rd-party) | yes | — | dim shortening |
| Qdrant / Weaviate / Vespa / faiss / Milvus | any | yes (infra) | mix | post-hoc/infra | 32×, up to 40× speedup |

**Takeaway:** binary + 32× is a solved, mainstream technique — **for text, post-hoc.**
Cross-modal learned binary is conspicuously absent from the industry side.

---

## 4. Product landscape (NL photo search)

| Product | On-device? | Backend | Binary? |
|---|---|---|---|
| Apple Photos (iOS 18+) | yes | undisclosed (likely float) | not disclosed |
| Google Photos | hybrid | CoCa + Vertex (float ANN) | no |
| Microsoft Photos (Copilot+) | yes (NPU) | undisclosed | no |
| immich (OSS, 100k★) | server | CLIP/SigLIP + pgvector (float) | **no** |
| PhotoPrism (OSS) | — | CLIP search still unshipped (#1287 open) | — |
| Queryable (iOS, OSS) | yes | MobileCLIP CoreML, float cosine brute-force | **no** |
| PicQuery (Android, OSS) | yes | CLIP, float | no |

**Takeaway:** NL photo search is **crowded at the cloud/float tier** (Apple/Google/MS) and
**thin at the on-device tier** (Queryable = float, English, iOS, degrades ~50K photos).
**No shipping product** uses end-to-end **binary cross-modal hash codes** for photo search.
No open product makes **multilingual on-device** search a feature.

---

## 5. Honest novelty verdict

**What is NOT novel (do not claim):**
- Binary embeddings / 32× compression (mainstream since 2024)
- Frozen VLM + MLP hash head + Hamming search (standard cross-modal hashing)
- InfoNCE + quantization + balance + consistency losses (textbook ingredients)
- Matryoshka float nesting (MRL 2022)
- Matryoshka + binary for **text** (QAMA learned; Vespa/Nomic post-hoc)

**What appears genuinely under-explored (defensible, with caveats):**
1. **Matryoshka prefix-nested *1-bit* binary codes for *cross-modal* image-text retrieval, end-to-end learned.** No peer-reviewed paper or product found combining all four. Bracketed by CroVCA (no nesting) and QAMA (text, 2-bit).
2. **A modern strong VLM (SigLIP2) as the hashing backbone** — minor (implementation), but unoccupied.
3. **Product**: on-device + binary cross-modal + multilingual + open-source photo search — a real whitespace.

**Supporting evidence the approach matters:** Nomic's public finding that *post-hoc*
binary quantization **fails** on cross-modal vision embeddings is direct motivation for
*learned* binary hashing here — it's not just a compression trick, it's the only thing
that works cross-modally.

**To make the novelty publishable, we would need to show:**
- The prefix-nesting constraint does NOT meaningfully hurt vs. training a separate model
  per bit length (our data already hints at this — short codes stay stable across options).
- The full quality/size/speed tradeoff curve across bit lengths (we have this:
  1024-bit ≈ 98% float R@10 @ 1/36 size, 5× speed — see `docs/DATA_AND_BENCHMARKS.md`).
- Ablations on the loss components and the SigLIP2-vs-CLIP backbone.

**Caveat on recency:** arXiv moves fast; a paper filling this exact gap could have appeared
between the surveyed sources (late 2025) and now. Re-check before any publication claim.

---

## 6. So… am I a pioneer?

Honestly: **a pioneer of a narrow, specific combination — not of the broad idea.** The
broad ideas (binary hashing, cross-modal retrieval, Matryoshka) have many parents. But
"**learned, prefix-nested, 1-bit, cross-modal hash codes on SigLIP2, shipped as an
on-device multilingual photo-search product**" is a combination nobody seems to have put
together and published. That's a legitimately interesting and defensible niche — worth a
clean write-up + a working demo, with the honest framing above (not "we invented binary
search," but "we're first to combine X+Y+Z for this use case, and here's why it matters").

### Closest things to cite / differentiate against
- **CroVCA / HashCoder** (2510.27584) — frozen-backbone hash head, binary, no nesting.
- **QAMA** (CIKM 2025) — Matryoshka + learned quantization, text-only, 2-bit.
- **Cohere binary embeddings** (2024) — post-hoc binary, text, the industry reference.
- **Nomic vision BQ failure** — evidence that post-hoc binary breaks cross-modal.
- **Queryable** — closest product (on-device CLIP photo search, but float).
