# FORMULATION — what the current pipeline computes, exactly

Mathematical specification + ML analysis of the deployed 1-bit cross-modal hashing pipeline, and a
per-stage list of *changeable points* (the basis for a paradigm-change "A" bet). Notation matches the
code (`src/models/nested_hash_layer.py`, `src/losses/*.py`). Approximations are flagged **[approx]**.

## 1. Forward pass
Backbone (frozen) $f$ maps input $x$ (image or text) to embedding $e=f(x)\in\mathbb{R}^{d}$
($d=1152$ for SigLIP2-So400m; $d=384$ for the e5 head-adapt text path).

Trunk (shared MLP), producing a length-$M$ pre-code ($M=1024$):
$$r \;=\; W_2\,\big(\mathrm{Drop}(\mathrm{GELU}(\mathrm{LN}(W_1 e + b_1)))\big) + b_2 \;\in\;\mathbb{R}^{M},\qquad W_1\!\in\!\mathbb{R}^{384\times d},\;W_2\!\in\!\mathbb{R}^{1024\times384}.$$
(final layer init: Xavier, gain $0.1$; bias $0$.)

Matryoshka nesting: for each code length $k\in\mathcal{K}=\{8,16,32,64,128,256,512,1024\}$ the code is a
**prefix** $s_k = r_{1:k}$, passed through a **per-length BatchNorm** then **L2**:
$$\tilde z_k=\mathrm{BN}_k(s_k)=\gamma_k\odot\frac{s_k-\hat\mu_k}{\sqrt{\hat\sigma_k^2+\epsilon}}+\beta_k,\qquad
z_k=\frac{\tilde z_k}{\lVert\tilde z_k\rVert_2}\in\mathbb{S}^{k-1}.$$
Outputs per length: continuous $c_k=\tanh(z_k)$ and binary $b_k=\operatorname{sign}(z_k)\in\{-1,+1\}^k$.
Backward uses a straight-through estimator (STE): $\frac{\partial b_k}{\partial z_k}\!:=\!\mathbb{1}[\,|z_k|\le 1\,]$.

> **[exact]** Because $z_k$ is L2-normalized, every component satisfies $|z_{k,i}|\le 1$, so the STE clip is
> **never** active — the gradient passes through $\operatorname{sign}$ as the identity on $z_k$. The model
> trains as if $b_k\equiv z_k$ on the backward pass; the only nonlinearity the gradient "sees" is the L2 sphere.
> Also, for $k=1024$, $\lVert z_k\rVert=1$ over $1024$ dims ⇒ typical $|z_{k,i}|\approx 1/\sqrt{1024}\approx0.031$,
> so $c_k=\tanh(z_k)\approx z_k$ (tanh is ~linear here). **Continuous code ≈ pre-sign $z$ ≈ scaled raw direction.**

Deployed head `ft113` always evaluates at $k=1024$ via $\mathrm{BN}_{1024}$ (the canonical anchor).

## 2. Objective
Training is on the **continuous** $c_k$ (never the discrete $b_k$ — STE bridges). Per length $k$, with paired
batch $(c_k^{I},c_k^{T})$ and temperature $\tau$ ($\approx0.083$):

**InfoNCE (the load-bearing term):** with $u=\mathrm{norm}(c_k^I),\,v=\mathrm{norm}(c_k^T)$, logits $\ell=uv^\top/\tau$,
$$\mathcal L_{\text{NCE}}=\tfrac12\big[\mathrm{CE}(\ell,\mathrm{diag})+\mathrm{CE}(\ell^\top,\mathrm{diag})\big].$$
Auxiliaries (all on $c_k$, summed over $k$ with small weights):
- **EAQL** (quant): $\sum_d w_d\,(c_{d}-\operatorname{sign}(c_d))^2$, $w_d=\mathrm{ema}_d/\sum\mathrm{ema}$.
- **OrthoHash**: $\mathrm{mean}_i(1-\mathrm{sim}_{ii})^2+\mathrm{mean}_{i\ne j}\mathrm{sim}_{ij}^2$, $\mathrm{sim}=\mathrm{norm}(c^I)\,\mathrm{norm}(c^T)^\top$.
- **BitBalance**: $\lVert\overline{c}\rVert^2+\lVert c^\top c/B-I\rVert_F^2$ (balance + decorrelation).
- **Consistency**: $\lVert c_k^{I}-c_k^{I,\text{aug}}\rVert^2$. **LCS**: $\sum_k \mathrm{MSE}(\mathrm{norm}(c_kc_k^\top),\,\mathrm{norm}(c_{k+1}c_{k+1}^\top).\mathrm{detach}())$.
- **margin** (an off-by-default option): $\mathrm{relu}(m-|z|)$-type — operates on $|z|$.

### 2.1 Why the auxiliaries are (near-)inert — derivation
$\mathrm{BN}_k$ pins the batch statistics of $\tilde z_k$ ($\mathbb E_B[\tilde z_{k,d}]\!\to\!\beta_{k,d}$, $\mathrm{Var}_B\!\to\!\gamma_{k,d}^2$) on **every** forward, and L2 pins $\lVert z_k\rVert=1$. So the *distribution* of $z_k$ (hence $c_k$) is fixed by $(\gamma_k,\beta_k)$ and the sphere, not by the upstream trunk gradient. Consequently:
- **Balance** $\lVert\overline c\rVert^2$: $\mathrm{BN}$ already centers each bit ($\beta_{k}\!\approx\!0$ at init) ⇒ $\overline c\!\approx\!0$; the balance gradient only nudges $\beta_k$ — **redundant with BN's centering**.
- **EAQL/margin** target $|c|\!\to\!1$ / large $|z|$, but L2 forces $\lVert z\rVert=1$ across $1024$ dims (can't enlarge all bits) and BN fixes the variance ⇒ the term can only *redistribute* magnitude on the sphere — a weak, **absorbed** gradient. (Empirically $\mathrm{mean}|z|\!=\!0.0249$ **identical** across ceiling/distill/head-adapt/image and across loss configs.)
- **Decorrelation / OrthoHash off-diagonal**: bits are already near-decorrelated ($\overline{|\rho_{ij}|}\!\approx\!0.08$) and near-max-entropy under BN+L2+InfoNCE ⇒ near-zero residual gradient.
- **Net** (matches the loss-ablation: every aux within seed noise at $k=1024$): **InfoNCE + BN carry the load; the regularizers are largely subsumed by BN+L2.** **[exact reasoning; magnitudes are empirical]**

## 3. Retrieval and the sign information loss
Search is exact Hamming over $\{-1,+1\}^K$:
$$H(b_q,b_g)=\#\{i:b_{q,i}\ne b_{g,i}\}=\tfrac12\big(K-\langle b_q,b_g\rangle\big).$$
Ranking by $H$ uses **only the signs** of $z$; the magnitudes $|z_{k,i}|$ (the model's per-bit confidence) are
discarded at query time. The continuous score $\langle z_q,z_g\rangle=K-2\!\sum_i$ (soft mismatch) would weight bits
by confidence; binarization replaces each $z_i$ by $\operatorname{sign}(z_i)$, a per-bit information reduction
$z_i\!\mapsto\!\mathbb{1}[z_i>0]$. **[exact]** (See `viz/sign_information_loss.html` for the empirical rank-flip
continuous-vs-Hamming, and `viz/forward_stages.html` for where the distribution collapses.)

## 4. ML-perspective analysis
- **Implicit assumptions.** (i) *Sign-only sufficiency*: relevance is recoverable from bit signs, magnitudes are noise. (ii) *(near-)Isotropic, decorrelated code*: BN+L2+balance push toward a roughly uniform sphere with independent bits (max coding entropy). (iii) *Bit independence at search*: Hamming treats bits as i.i.d.; correlated bits double-count.
- **Information.** Continuous $z_k\in\mathbb S^{k-1}$ → $b_k$: each bit caps at $1$ bit; $1024$ bits ≤ $1024$ bits of capacity, but correlated/unbalanced bits realize less. Empirically codes are near-balanced/full-rank (≈max realized), so the binarization loss is mostly the discarded **magnitude/ordering within a bit**, not lost bits.
- **Gradient flow.** Frozen $f$ ⇒ no backbone gradient (head-only). STE makes $\partial b/\partial z=\mathrm{id}$ on the sphere; BN re-normalizes each step (absorbs scale grads); the trunk learns a **rotation/projection into a fixed-geometry sphere**, fit by InfoNCE. There is no gradient path that changes "how many bits" or the sphere's radius.
- **Representational capacity.** $b_{1024}=\operatorname{sign}(z(e))$ with a 2-layer trunk over a *frozen* $e$: the realizable code family is $\operatorname{sign}$ of a (rotated, normalized) **2-layer MLP image** of the frozen embedding — i.e. **linear-separable partitions of the SigLIP manifold into $2^{1024}$ orthants**, not arbitrary functions of $x$. Ceiling is set by what $f$ already linearly exposes (this bounded everything: gates 1–8).

## 5. Changeable points (basis for an "A" bet)
| stage | current | alternative | what it would change |
|---|---|---|---|
| slice (Matryoshka) | strict prefix + per-$k$ BN | learned per-$k$ projection / nested-dropout / no nesting | decouple short-code quality from prefix; risk losing one-index-fits-all |
| BN$_k$ | per-bit BatchNorm (batch stats) | LayerNorm / none+scale / **learned rotation (ITQ)** | gate 6 (RED): all ≈ baseline → BN incidental, not the lever |
| L2 | unit sphere | learned radius / whitening / no-norm | frees magnitude the quant/margin terms want; may destabilize (gate 6 instability watch) |
| **sign / code alphabet** | $\operatorname{sign}$, 1 bit/dim, STE | **multi-bit / residual-PQ / soft-then-quantize / learned codebook (VQ)** | *changes the alphabet* — the one stage that discards magnitude; candidate for a real paradigm shift (none of gates 1–8 touched this) |
| loss | InfoNCE + aux (aux inert) | replace InfoNCE primary (gate 5 RED for add-ons); supervised/category; **rank-preservation (CroVCA, gate 8 RED as predictor)** | InfoNCE already R@10-optimal for the sign code; only a *different output structure* (above) is likely to move it |
| Hamming search | i.i.d. bit XOR | weighted/learned-metric Hamming; asymmetric (query continuous, gallery binary); **coarse-to-fine (gate 7 GREEN, system)** | keep accuracy at lower bit-ops; asymmetric search could recover some discarded magnitude at query time |
| backbone $f$ | frozen | **LoRA / partial unfreeze (Lever C, untested)** | the only path that changes what $e$ exposes — beyond the head's reach; deployability cost (encoder grows) |

**Reading (neutral, for joint discussion):** gates 1–8 varied {loss terms, normalizer, predictors, dynamic bits}
and never beat InfoNCE+sign on R@10. The **two untouched stages** are (a) the **code alphabet / sign step**
(magnitude is provably discarded here) and (b) the **frozen backbone**. An "A" bet that changes the code-generation
paradigm should target (a) — a different output structure than per-dim sign — since every in-place tweak to the
current sign code has been gated RED.
