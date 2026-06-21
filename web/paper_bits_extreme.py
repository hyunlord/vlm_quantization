"""(h) Extreme bit sweep — characterize behavior beyond the current 1024 max (→2048→4096) on the
instance task, with storage + latency cost. Tests the hypothesis that recall saturates near the frozen
embedding's effective dim (so400m ~1152-d) and bits beyond that are pure storage/latency cost.

Fresh Matryoshka head (clean COCO recipe = the train_1024 "coco" mode, frozen backbone, cached so400m
embeddings) trained ONCE with bit_list [16..4096], evaluated at every bit on:
  COCO 5K test  (EN, KO)            — cached emb_cache / coco_ko_test
  XM3600 avg36                      — cached /tmp/xm_so400m.pt (built by (g))
Cost columns per bit: bytes/img (=bits/8), index MB @50K, single-query Hamming top-10 latency @50K gallery.

Anchor: the 1024-bit point should track batch#1 (b) full (clean-COCO head, EN R@10 ≈ 79.98) — this is a
fresh clean-COCO head, NOT the deployed ft113 (so it won't equal bits_sweep.csv's ft113 exactly; the
SATURATION SHAPE is the finding).

Run on DGX: BITS=16,32,64,128,256,512,1024,2048,4096 .venv/bin/python web/paper_bits_extreme.py
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
BITS = [int(x) for x in os.environ.get("BITS", "16,32,64,128,256,512,1024,2048,4096").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"


def pack_bits(codes):
    b = (np.asarray(codes) > 0).astype(np.uint8)
    if b.ndim == 1:
        b = b[None, :]
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="big"), dtype=np.uint8)


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_map(ix, q, gold):
    _, I = ix.search(q, 10)
    r10 = round(100 * np.mean([gold[i] in I[i, :10] for i in range(len(gold))]), 2)
    aps = []
    for i in range(len(gold)):
        hit = np.where(I[i, :10] == gold[i])[0]
        aps.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return r10, round(100 * float(np.mean(aps)), 2)


P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
XM = torch.load("/tmp/xm_so400m.pt", map_location="cpu") if os.path.exists("/tmp/xm_so400m.pt") else None


def nrm(x):
    return F.normalize(x, dim=1)


clean, weak, strong, txt = nrm(TC["clean"]), nrm(TC["weak"]), nrm(TC["strong"]), nrm(TC["txt"])
te_i, te_en = nrm(EC["test"]["img"]), nrm(EC["test"]["txt"])
te_ko = nrm(KO["txt_emb"].float())
embed, N = clean.shape[1], clean.shape[0]
n_test = te_i.shape[0]
gold_coco = list(range(n_test))


def train_head():
    torch.manual_seed(42)
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = perm[s:s + BS]
            ci, wi, si, ti = clean[idx].to(dev), weak[idx].to(dev), strong[idx].to(dev), txt[idx].to(dev)
            io = img_h(ci)
            out = lf(io, txt_h(ti), weak_image_outputs=img_h(wi), aug_image_outputs=img_h(si),
                     progress=g / max(steps, 1))
            opt.zero_grad(); out["total"].backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    print(f"[h] trained [16..{BITS[-1]}] head in {time.perf_counter()-t0:.0f}s", flush=True)
    return img_h, txt_h


@torch.no_grad()
def codes_at(head, emb_t, bi):
    return head(emb_t.to(dev))[bi]["binary"].detach().cpu().numpy()


def latency_50k(gal_codes_bit, bits, n=50000, nq=500):
    """tile codes to ~50K, time faiss IndexBinaryFlat top-10 search of nq queries -> ms/query."""
    reps = (n + gal_codes_bit.shape[0] - 1) // gal_codes_bit.shape[0]
    big = np.tile(gal_codes_bit, (reps, 1))[:n]
    bigp = pack_bits(big)
    ix = faiss_bin(bigp, bits)
    q = bigp[:nq]
    t0 = time.perf_counter()
    ix.search(q, 10)
    return round(1000 * (time.perf_counter() - t0) / nq, 4)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    img_h, txt_h = train_head()
    with torch.no_grad():
        io_coco = img_h(te_i.to(dev)); to_en = txt_h(te_en.to(dev)); to_ko = txt_h(te_ko.to(dev))
        xm_io = xm_to = None
        if XM is not None:
            xm_img = nrm(XM["img_emb"]).to(dev)
            xm_io = img_h(xm_img)

    rows = []
    for bi, bits in enumerate(BITS):
        # COCO
        gal = pack_bits(io_coco[bi]["binary"].detach().cpu().numpy())
        ix = faiss_bin(gal, bits)
        en_r10, en_map = recall_map(ix, pack_bits(to_en[bi]["binary"].detach().cpu().numpy()), gold_coco)
        ko_r10, ko_map = recall_map(ix, pack_bits(to_ko[bi]["binary"].detach().cpu().numpy()), gold_coco)
        bytes_img = bits // 8
        idx_mb = round(50000 * bytes_img / 1e6, 2)
        lat = latency_50k(io_coco[bi]["binary"].detach().cpu().numpy(), bits)
        rows.append({"task": "instance", "dataset": "coco", "lang": "en", "bits": bits,
                     "bytes_per_img": bytes_img, "R10": en_r10, "mAP10": en_map, "idx_mb_50k": idx_mb,
                     "latency_ms_50k": lat})
        rows.append({"task": "instance", "dataset": "coco", "lang": "ko", "bits": bits,
                     "bytes_per_img": bytes_img, "R10": ko_r10, "mAP10": ko_map, "idx_mb_50k": idx_mb,
                     "latency_ms_50k": lat})
        # XM3600 avg36
        if XM is not None:
            galx = pack_bits(xm_io[bi]["binary"].detach().cpu().numpy())
            ixx = faiss_bin(galx, bits)
            r10s, maps = [], []
            for L, d in XM["per_lang"].items():
                with torch.no_grad():
                    qc = txt_h(nrm(d["text_emb"]).to(dev))[bi]["binary"].detach().cpu().numpy()
                r10, mp = recall_map(ixx, pack_bits(qc), d["gold"])
                r10s.append(r10); maps.append(mp)
            rows.append({"task": "instance", "dataset": "xm3600", "lang": "avg36", "bits": bits,
                         "bytes_per_img": bytes_img, "R10": round(float(np.mean(r10s)), 2),
                         "mAP10": round(float(np.mean(maps)), 2), "idx_mb_50k": idx_mb, "latency_ms_50k": lat})
        print(f"[h] bit {bits}: COCO EN R@10 {en_r10}/mAP {en_map} KO {ko_r10} | "
              f"{bytes_img}B/img idx {idx_mb}MB lat {lat}ms"
              + (f" | XM avg36 {rows[-1]['R10']}" if XM is not None else ""), flush=True)

    cols = ["task", "dataset", "lang", "bits", "bytes_per_img", "R10", "mAP10", "idx_mb_50k", "latency_ms_50k"]
    with open(PAPER / "bits_extreme.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[h] RESULT_JSON " + json.dumps({"bits": BITS, "rows": rows}), flush=True)
    print(f"[h] DONE -> paper/bits_extreme.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
