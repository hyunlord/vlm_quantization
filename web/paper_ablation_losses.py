"""(b) Hash-head loss-component ablation — head-only retrain (frozen SigLIP2 backbone,
cached COCO embeddings). Reproduces the EXACT scripts/train_1024.py recipe (both heads
img_h+txt_h trained from scratch on /tmp/emb_aug.pt COCO English embeddings, aug views ->
consistency, Matryoshka [8..1024] -> lcs, weights from /tmp/hp_results.json best_params).

InfoNCE is always on; one regularizer is removed at a time:
  full          all terms (the train_1024 recipe baseline)
  -quant        quantization (EAQL) weight -> 0
  -balance      bit-balance weight -> 0
  -ortho        orthogonality weight -> 0
  -consistency  aug-consistency (MSE) weight -> 0   (aug views still feed InfoNCE; only the
                                                     consistency objective is removed)
  -lcs          length/Matryoshka self-distillation weight -> 0
  full(seed=123) noise control

Per config we measure, on the eval_korean 5K protocol (gallery = img_h(test_img) 1024-bit codes,
query = txt_h(text_emb), faiss IndexBinaryFlat Hamming, gold = caption_i -> image_i):
  EN_R10, KO_R10
and code statistics at the 1024-bit code (img+txt test codes pooled):
  mean_bit_activation  mean over bits of P(bit=1)               (balance target ~0.5)
  bit_balance_gap      mean over bits of |P(bit=1) - 0.5|       (balance term effect; higher=worse)
  bit_decorr_metric    mean |off-diagonal| of the bit-bit corr  (ortho term effect; higher=worse)
  quant_margin         mean |tanh(pre-sign)| in [0,1]           (quant term effect; ->1 = saturated/good)

Frozen backbone, NO retraining of the backbone, NO index/common.py changes.
Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_ablation_losses.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256,512,1024").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = 512
MB = BITS[-1]  # max / headline bit (1024)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def pack_bits(codes):
    bits01 = (np.asarray(codes) > 0).astype(np.uint8)
    if bits01.ndim == 1:
        bits01 = bits01[None, :]
    return np.ascontiguousarray(np.packbits(bits01, axis=1, bitorder="big"), dtype=np.uint8)


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits)
    ix.add(np.ascontiguousarray(packed))
    return ix


def recall_ks(ix, q, gold, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]), 2) for k in ks}


# ---- data (identical sources/convention to train_1024.py; NORM_IN=1) ----
P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")


def inp(x):
    return F.normalize(x, dim=1)  # NORM_IN=1 (train_1024 default)


clean, weak, strong, txt = inp(TC["clean"]), inp(TC["weak"]), inp(TC["strong"]), inp(TC["txt"])
te_i = inp(EC["test"]["img"])
te_en = inp(EC["test"]["txt"])
te_ko = inp(KO["txt_emb"].float())
embed, N = clean.shape[1], clean.shape[0]
n_test = te_i.shape[0]
gold = list(range(n_test))

BASE_W = {"ortho": P["ortho"], "quant": P["quant"], "balance": P["balance"],
          "cons": P["cons"], "lcs": P["lcs"]}
print(f"[abl] N_train={N} n_test={n_test} embed={embed} hidden={P['hidden']} BITS={BITS} EPOCHS={EPOCHS}", flush=True)
print(f"[abl] baseline weights (from hp_results best_params): InfoNCE=1.0 "
      f"ortho={BASE_W['ortho']} quant={BASE_W['quant']} balance={BASE_W['balance']} "
      f"cons={BASE_W['cons']} lcs={BASE_W['lcs']} temp={P['temperature']}", flush=True)

# ablation configs: (label, weight overrides, seed)
CONFIGS = [
    ("full", {}, 42),
    ("-quant", {"quant": 0.0}, 42),
    ("-balance", {"balance": 0.0}, 42),
    ("-ortho", {"ortho": 0.0}, 42),
    ("-consistency", {"cons": 0.0}, 42),
    ("-lcs", {"lcs": 0.0}, 42),
    ("full (seed=123)", {}, 123),
]


def train_one(w, seed):
    torch.manual_seed(seed)
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=w["ortho"],
                          quantization_weight=w["quant"], balance_weight=w["balance"],
                          consistency_weight=w["cons"], lcs_weight=w["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()),
                            lr=P["lr"], weight_decay=P["wd"])
    steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        pc = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = pc[s:s + BS]
            ci, wi, si, ti = clean[idx].to(dev), weak[idx].to(dev), strong[idx].to(dev), txt[idx].to(dev)
            io = img_h(ci)
            out = lf(io, txt_h(ti), weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si),
                     progress=g / max(steps, 1))
            opt.zero_grad(); out["total"].backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    return img_h, txt_h, time.perf_counter() - t0


def _quant_err(cont):
    """EAQL objective: mean((continuous - sign(continuous))^2). Lower = cleaner quantization."""
    return ((cont - torch.sign(cont)) ** 2).mean().item()


@torch.no_grad()
def evaluate(img_h, txt_h):
    io = img_h(te_i.to(dev))
    to_en = txt_h(te_en.to(dev))
    to_ko = txt_h(te_ko.to(dev))
    bi = BITS.index(MB)
    gal = pack_bits(io[bi]["binary"].cpu().numpy())
    ix = faiss_bin(gal, MB)
    r_en = recall_ks(ix, pack_bits(to_en[bi]["binary"].cpu().numpy()), gold)
    r_ko = recall_ks(ix, pack_bits(to_ko[bi]["binary"].cpu().numpy()), gold)

    # ---- code statistics at the 1024-bit code (img + txt test codes pooled) ----
    binr = torch.cat([io[bi]["binary"], to_en[bi]["binary"], to_ko[bi]["binary"]], 0)  # (M,1024) ±1
    cont = torch.cat([io[bi]["continuous"], to_en[bi]["continuous"], to_ko[bi]["continuous"]], 0)
    act = (binr > 0).float().mean(0)                       # per-bit activation rate
    mean_act = act.mean().item()                           # balance: ~0.5 ideal
    balance_gap = (act - 0.5).abs().mean().item()          # balance: lower ideal
    X = binr - binr.mean(0, keepdim=True)                  # decorrelation (BitBalance decorr part)
    Xn = X / (X.std(0, keepdim=True) + 1e-8)
    C = (Xn.t() @ Xn) / X.shape[0]
    decorr = ((C.abs().sum() - C.abs().diagonal().sum()) / (C.shape[0] * (C.shape[0] - 1))).item()
    quant_err = _quant_err(cont)                           # quant @1024 (dim-bounded; small leverage)
    bi64 = BITS.index(64) if 64 in BITS else 0             # quant @short prefix (more leverage)
    cont64 = torch.cat([io[bi64]["continuous"], to_en[bi64]["continuous"], to_ko[bi64]["continuous"]], 0)
    quant_err64 = _quant_err(cont64)
    # ortho (cross-modal alignment): mean paired img<->txt(EN) cosine on continuous codes
    ic, tc = io[bi]["continuous"], to_en[bi]["continuous"]
    xpos = F.cosine_similarity(ic, tc, dim=1).mean().item()
    return dict(EN_R10=r_en[10], KO_R10=r_ko[10], EN_R1=r_en[1], KO_R1=r_ko[1],
                mean_bit_activation=round(mean_act, 4), bit_balance_gap=round(balance_gap, 4),
                bit_decorr_metric=round(decorr, 4), xmodal_pos_cos=round(xpos, 4),
                quant_err=round(quant_err, 4), quant_err64=round(quant_err64, 4))


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, override, seed in CONFIGS:
        w = dict(BASE_W); w.update(override)
        print(f"=== {label} (seed={seed}) weights={w} ===", flush=True)
        img_h, txt_h, dt = train_one(w, seed)
        m = evaluate(img_h, txt_h)
        row = {"config": label, "EN_R10": m["EN_R10"], "KO_R10": m["KO_R10"],
               "mean_bit_activation": m["mean_bit_activation"], "bit_balance_gap": m["bit_balance_gap"],
               "bit_decorr_metric": m["bit_decorr_metric"], "xmodal_pos_cos": m["xmodal_pos_cos"],
               "quant_err": m["quant_err"], "quant_err64": m["quant_err64"],
               "EN_R1": m["EN_R1"], "KO_R1": m["KO_R1"], "train_s": round(dt)}
        rows.append(row)
        print(f"[abl] {label}: EN_R10 {m['EN_R10']} KO_R10 {m['KO_R10']} | act {m['mean_bit_activation']} "
              f"gap {m['bit_balance_gap']} decorr {m['bit_decorr_metric']} xcos {m['xmodal_pos_cos']} "
              f"qerr {m['quant_err']}/{m['quant_err64']}(64b) ({dt:.0f}s)", flush=True)

    cols = ["config", "EN_R10", "KO_R10", "mean_bit_activation", "bit_balance_gap",
            "bit_decorr_metric", "xmodal_pos_cos", "quant_err", "quant_err64", "EN_R1", "KO_R1", "train_s"]
    with open(PAPER / "ablation_losses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[abl] RESULT_JSON " + json.dumps({"bits": MB, "epochs": EPOCHS, "base_weights": BASE_W,
          "rows": rows}, ensure_ascii=False), flush=True)
    print(f"[abl] DONE -> paper/ablation_losses.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
