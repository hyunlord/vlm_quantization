"""Deep-lever gate harness — head-only training on cached SigLIP2 embeddings.

Reproduces train_1024.py mode=coco baseline EXACTLY when HEAD_NORM=bn LOSS=infonce.
Adds two orthogonal knobs that prior work flagged as the genuinely-untested levers:

  Lever B (head architecture): HEAD_NORM swaps the per-bit BatchNorm
    bn         : F.normalize(BN(sliced))          [== current head / baseline]
    ln         : F.normalize(LayerNorm(sliced))   [norm w/o batch stats]
    none       : F.normalize(sliced)              [pure L2, no per-bit norm]
    none_scale : F.normalize(affine(sliced))      [learnable per-dim scale+bias, no batch stats]
    rotation   : F.normalize(slice(raw @ R))      [learnable rotation (ITQ-style) + ortho penalty, no BN]

  Lever A (negative-separation training LOSS — prior work only used these as
  PREDICTORS, never as a training objective):
    infonce          : full CombinedHashLoss recipe only   [baseline]
    hardneg          : Hamming-space hard-negative-weighted contrastive (Robinson'21) ONLY
    hmargin          : margin-in-Hamming triplet (maximize correct-vs-nearest-wrong gap) ONLY
    infonce_hardneg  : full recipe + NEGSEP_W * hardneg
    infonce_hmargin  : full recipe + NEGSEP_W * hmargin

Eval: COCO 5K instance retrieval, EN (T2I + I2T) + KO (T2I), R@{1,5,10}, every bit in BITS.
Also logs the realized test-set hmargin (mean nearest-wrong-Hamming minus paired Hamming,
in bits) at max bit — the mechanism check for Lever A.

Env: HEAD_NORM(bn) LOSS(infonce) NEGSEP_W(0.5) HMARGIN_M(0.1) ORTHO_R_W(0.01)
     AUX_SCALE(1.0) SEED(0) EPOCHS(25) BITS(64,128,256,512,1024) TAG(label) CSV(path)
"""
from __future__ import annotations
import csv, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization")
sys.path.insert(0, REPO)
from src.losses.combined import CombinedHashLoss
from src.models.hash_layer import SignSTE

HEAD_NORM = os.environ.get("HEAD_NORM", "bn")
LOSS = os.environ.get("LOSS", "infonce")
NEGSEP_W = float(os.environ.get("NEGSEP_W", "0.5"))
HMARGIN_M = float(os.environ.get("HMARGIN_M", "0.1"))   # fraction-of-bits margin
ORTHO_R_W = float(os.environ.get("ORTHO_R_W", "0.01"))  # rotation orthogonality penalty
AUX_SCALE = float(os.environ.get("AUX_SCALE", "1.0"))   # scales ortho/quant/balance/cons/lcs
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BITS = [int(x) for x in os.environ.get("BITS", "64,128,256,512,1024").split(",")]
TAG = os.environ.get("TAG", f"{HEAD_NORM}-{LOSS}-s{SEED}")
CSV = os.environ.get("CSV", "/tmp/lever_results.csv")
BS = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

# hp best-params (same source train_1024.py uses)
import json
P = json.load(open("/tmp/hp_results.json"))["best_params"]


def nrm(x):
    return F.normalize(x, dim=1)


# ----------------------------- configurable head -----------------------------
class ConfigHead(nn.Module):
    """NestedHashLayer with a swappable per-bit normalization (Lever B).

    HEAD_NORM=bn reproduces src/models/nested_hash_layer.py byte-for-byte.
    """

    def __init__(self, input_dim, hidden_dim, bit_list, dropout, norm):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]
        self.norm = norm
        self.hash_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_bit),
        )
        nn.init.xavier_uniform_(self.hash_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.hash_head[-1].bias)

        if norm == "bn":
            self.pbn = nn.ModuleList([nn.BatchNorm1d(b) for b in self.bit_list])
        elif norm == "ln":
            self.pbn = nn.ModuleList([nn.LayerNorm(b) for b in self.bit_list])
        elif norm == "none_scale":
            # learnable per-dim affine WITHOUT batch statistics (isolates BN's affine
            # from BN's normalization). One affine over max_bit, prefix-sliced.
            self.aff_w = nn.Parameter(torch.ones(self.max_bit))
            self.aff_b = nn.Parameter(torch.zeros(self.max_bit))
        elif norm == "rotation":
            # learnable rotation on raw (ITQ-style), softly constrained orthogonal
            self.rot = nn.Parameter(torch.eye(self.max_bit))
        elif norm == "none":
            pass
        else:
            raise ValueError(f"unknown HEAD_NORM {norm}")

    def ortho_penalty(self):
        if self.norm != "rotation":
            return torch.tensor(0.0, device=self.hash_head[0].weight.device)
        R = self.rot
        I = torch.eye(R.size(0), device=R.device, dtype=R.dtype)
        return ((R.t() @ R - I) ** 2).mean()

    def forward(self, embeddings):
        raw = self.hash_head(embeddings)  # (B, max_bit)
        if self.norm == "rotation":
            raw = raw @ self.rot
        outputs = []
        for i, length in enumerate(self.bit_list):
            sliced = raw[:, :length]
            if self.norm == "bn":
                z = F.normalize(self.pbn[i](sliced), p=2, dim=1)
            elif self.norm == "ln":
                z = F.normalize(self.pbn[i](sliced), p=2, dim=1)
            elif self.norm == "none_scale":
                z = sliced * self.aff_w[:length] + self.aff_b[:length]
                z = F.normalize(z, p=2, dim=1)
            else:  # none, rotation
                z = F.normalize(sliced, p=2, dim=1)
            outputs.append({"continuous": torch.tanh(z), "binary": SignSTE.apply(z)})
        return outputs


# ----------------------------- negsep losses (Lever A) -----------------------------
def soft_sim(ci, ct):
    """Per-bit soft-code similarity in [-1,1]; sim = mean_d tanh-code product.
    Proportional to (1 - 2*normalized_Hamming): higher sim = smaller Hamming."""
    d = ci.size(1)
    return (ci @ ct.t()) / d


def hardneg_loss(ci, ct, tau=0.1, beta=1.0):
    """Hamming-space contrastive with hard-negative reweighting (Robinson et al. 2021):
    negatives are upweighted by exp(beta * sim) so the hardest (closest-in-Hamming)
    wrong items dominate the denominator. Symmetric over both directions."""
    s = soft_sim(ci, ct) / tau                       # (B,B)
    B = s.size(0)
    eye = torch.eye(B, device=s.device, dtype=torch.bool)

    def one_dir(logits):
        pos = logits[eye].view(B, 1)                 # (B,1) diagonal
        neg = logits[~eye].view(B, B - 1)            # (B,B-1) off-diagonal
        w = torch.softmax(beta * neg.detach(), dim=1)  # hardness weights (stop-grad)
        neg_lse = torch.logsumexp(neg + torch.log(w * (B - 1) + 1e-12), dim=1, keepdim=True)
        denom = torch.logsumexp(torch.cat([pos, neg_lse], dim=1), dim=1)
        return (denom - pos.squeeze(1)).mean()

    return 0.5 * (one_dir(s) + one_dir(s.t()))


def hmargin_loss(ci, ct, margin_frac=0.1):
    """Margin-in-Hamming triplet: directly maximize the gap between the paired
    (correct) soft-Hamming and the nearest-wrong soft-Hamming. margin in fraction
    of bits. Hfrac = (1 - sim)/2 in [0,1]. Symmetric over both directions."""
    s = soft_sim(ci, ct)                              # (B,B) in [-1,1]
    H = (1.0 - s) / 2.0                               # soft normalized Hamming [0,1]
    B = H.size(0)
    eye = torch.eye(B, device=H.device, dtype=torch.bool)

    def one_dir(Hm):
        pos = Hm[eye]                                 # (B,) paired Hamming
        neg = Hm.masked_fill(eye, float("inf"))       # mask self
        hardneg = neg.min(dim=1).values               # (B,) nearest-wrong Hamming
        return F.relu(margin_frac - (hardneg - pos)).mean()

    return 0.5 * (one_dir(H) + one_dir(H.t()))


# ----------------------------- data -----------------------------
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
clean, weak, strong, txt = nrm(TC["clean"]), nrm(TC["weak"]), nrm(TC["strong"]), nrm(TC["txt"])
te_i, te_t = nrm(EC["test"]["img"]), nrm(EC["test"]["txt"])
te_ko = nrm(KO["txt_emb"])
ids = EC["test"]["ids"]
labels = ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])
ko_ids = KO["ids"]
ko_ids = ko_ids if torch.is_tensor(ko_ids) else torch.tensor([int(x) for x in ko_ids])
assert torch.equal(labels, ko_ids), "KO ids must align to EN test order"
embed, N = clean.shape[1], clean.shape[0]
KS = [1, 5, 10]


def hdist(q, db):
    return (q.size(1) - q @ db.t()) / 2


def retr(qcodes, gcodes, larger=False):
    d = hdist(qcodes, gcodes)
    order = d.argsort(dim=1, descending=larger)
    rel = (labels[order] == labels[:, None])
    return {k: round((rel[:, :k].sum(1) > 0).float().mean().item() * 100, 2) for k in KS}


def realized_hmargin(qcodes, gcodes):
    """test-set hmargin in BITS at this bit length: mean(nearest-wrong - paired)."""
    d = hdist(qcodes, gcodes)
    B = d.size(0)
    eye = torch.eye(B, device=d.device, dtype=torch.bool)
    pos = d[eye]
    neg = d.masked_fill(eye, float("inf")).min(dim=1).values
    diff = (neg - pos)
    return round(diff.mean().item(), 3), round(diff.std().item(), 3)


def train():
    torch.manual_seed(SEED)
    img_h = ConfigHead(embed, P["hidden"], BITS, P["dropout"], HEAD_NORM).to(dev)
    txt_h = ConfigHead(embed, P["hidden"], BITS, P["dropout"], HEAD_NORM).to(dev)
    lf = CombinedHashLoss(
        BITS, contrastive_weight=1.0,
        ortho_weight=P["ortho"] * AUX_SCALE, quantization_weight=P["quant"] * AUX_SCALE,
        balance_weight=P["balance"] * AUX_SCALE, consistency_weight=P["cons"] * AUX_SCALE,
        lcs_weight=P["lcs"] * AUX_SCALE, temperature=P["temperature"],
    ).to(dev)
    opt = torch.optim.AdamW(
        list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"]
    )
    steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    use_recipe = LOSS in ("infonce", "infonce_hardneg", "infonce_hmargin")
    use_hardneg = LOSS in ("hardneg", "infonce_hardneg")
    use_hmargin = LOSS in ("hmargin", "infonce_hmargin")

    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    naninf = 0
    for ep in range(EPOCHS):
        pc = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = pc[s:s + BS]
            ci, wi, si, ti = clean[idx].to(dev), weak[idx].to(dev), strong[idx].to(dev), txt[idx].to(dev)
            io, to = img_h(ci), txt_h(ti)
            total = torch.tensor(0.0, device=dev)
            if use_recipe:
                total = total + lf(io, to, weak_image_outputs=img_h(wi),
                                   aug_image_outputs=img_h(si), progress=g / max(steps, 1))["total"]
            if use_hardneg or use_hmargin:
                # negsep on continuous codes, averaged over bits
                ns = torch.tensor(0.0, device=dev)
                for k in range(len(BITS)):
                    cik, ctk = io[k]["continuous"], to[k]["continuous"]
                    if use_hardneg:
                        ns = ns + hardneg_loss(cik, ctk)
                    if use_hmargin:
                        ns = ns + hmargin_loss(cik, ctk, HMARGIN_M)
                ns = ns / len(BITS)
                w = 1.0 if LOSS in ("hardneg", "hmargin") else NEGSEP_W
                total = total + w * ns
            if HEAD_NORM == "rotation":
                total = total + ORTHO_R_W * (img_h.ortho_penalty() + txt_h.ortho_penalty())
            if not torch.isfinite(total):
                naninf += 1
                opt.zero_grad(); g += 1; sched.step(); continue
            opt.zero_grad(); total.backward(); opt.step(); sched.step(); g += 1

    img_h.eval(); txt_h.eval()
    with torch.no_grad():
        io = img_h(te_i.to(dev)); eo = txt_h(te_t.to(dev)); kno = txt_h(te_ko.to(dev))
    rows = []
    for k, b in enumerate(BITS):
        ic = io[k]["binary"].cpu().float()
        ec = eo[k]["binary"].cpu().float()
        kc = kno[k]["binary"].cpu().float()
        en_t2i = retr(ec, ic)
        en_i2t = retr(ic, ec)
        ko_t2i = retr(kc, ic)
        hm, hmstd = realized_hmargin(ec, ic)  # EN T2I hmargin
        rows.append({
            "tag": TAG, "head_norm": HEAD_NORM, "loss": LOSS, "negsep_w": NEGSEP_W,
            "hmargin_m": HMARGIN_M, "aux_scale": AUX_SCALE, "seed": SEED, "bit": b,
            "en_t2i_r1": en_t2i[1], "en_t2i_r5": en_t2i[5], "en_t2i_r10": en_t2i[10],
            "en_i2t_r1": en_i2t[1], "en_i2t_r5": en_i2t[5], "en_i2t_r10": en_i2t[10],
            "ko_t2i_r1": ko_t2i[1], "ko_t2i_r5": ko_t2i[5], "ko_t2i_r10": ko_t2i[10],
            "hmargin_bits": hm, "hmargin_std": hmstd, "naninf": naninf,
            "secs": round(time.perf_counter() - t0, 1),
        })
        print(f"  [{TAG}] {b:>4}b EN T2I R@1/5/10 {en_t2i[1]}/{en_t2i[5]}/{en_t2i[10]} "
              f"| EN I2T R@10 {en_i2t[10]} | KO T2I R@10 {ko_t2i[10]} | hmargin {hm}±{hmstd}b "
              f"| naninf {naninf}", flush=True)
    return rows


def append_csv(rows):
    fields = list(rows[0].keys())
    exists = os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    print(f"=== {TAG} | HEAD_NORM={HEAD_NORM} LOSS={LOSS} NEGSEP_W={NEGSEP_W} "
          f"HMARGIN_M={HMARGIN_M} AUX_SCALE={AUX_SCALE} SEED={SEED} EPOCHS={EPOCHS} "
          f"BITS={BITS} ===", flush=True)
    rows = train()
    append_csv(rows)
    print(f"LEVER_DONE {TAG} -> {CSV}", flush=True)
