"""(3) Extreme bit sweep [8..4096] re-run with so400m text LOWERCASED (batch #2 (h) was orig-case).

Same fresh clean-COCO Matryoshka head as (h) (frozen backbone, cached emb_aug train embeddings — ORIG case,
mirroring how the deployed ft113 was orig-trained), trained ONCE, then evaluated on BOTH orig-case and
lowercased test text in the same run = a controlled orig-vs-lower snapshot at every bit. This matches how the
lowercased SERVER number was obtained (ft113 orig-trained + lowercased eval -> 81.24), so the 1024-bit
lowercased point should track ~81.2.

  COCO 5K EN: orig (emb_cache) vs lower (/tmp/coco_en_lc.pt)
  COCO 5K KO: caseless (coco_ko_test) -> single row, unchanged
  XM3600 avg36: orig (/tmp/xm_so400m.pt) vs lower (/tmp/xm_so400m_lc.pt)
Cost columns per bit: bytes/img, 50K index MB, single-query Hamming top-10 latency @50K. KO + curve SHAPE
must be invariant; only cased English/Latin XM langs shift up.

Run: BITS=8,16,32,64,128,256,512,1024,2048,4096 EPOCHS=12 .venv/bin/python web/paper_bits_extreme_lc.py
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
BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256,512,1024,2048,4096").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "12"))
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
    _, I = ix.search(np.ascontiguousarray(q), 10)
    r10 = round(100 * float(np.mean([gold[i] in I[i, :10] for i in range(len(gold))])), 2)
    aps = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0)
           for i in range(len(gold))]
    return r10, round(100 * float(np.mean(aps)), 2)


def nrm(x):
    return F.normalize(x, dim=1)


P = json.load(open("/tmp/hp_results.json"))["best_params"]
TC = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
XM = torch.load("/tmp/xm_so400m.pt", map_location="cpu") if os.path.exists("/tmp/xm_so400m.pt") else None
XML = torch.load("/tmp/xm_so400m_lc.pt", map_location="cpu") if os.path.exists("/tmp/xm_so400m_lc.pt") else None
EN_LOW = torch.load("/tmp/coco_en_lc.pt", map_location="cpu")

clean, weak, strong, txt = nrm(TC["clean"]), nrm(TC["weak"]), nrm(TC["strong"]), nrm(TC["txt"])
te_i = nrm(EC["test"]["img"])
te_en_orig = nrm(EC["test"]["txt"])
te_en_low = nrm(EN_LOW.float())
te_ko = nrm(KO["txt_emb"].float())
embed, N = clean.shape[1], clean.shape[0]
gold_coco = list(range(te_i.shape[0]))


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
    print(f"[h-lc] trained [{BITS[0]}..{BITS[-1]}] head {EPOCHS}ep in {time.perf_counter()-t0:.0f}s", flush=True)
    return img_h, txt_h


def latency_50k(gal_codes_bit, bits, n=50000, nq=500):
    reps = (n + gal_codes_bit.shape[0] - 1) // gal_codes_bit.shape[0]
    big = np.tile(gal_codes_bit, (reps, 1))[:n]
    ix = faiss_bin(pack_bits(big), bits)
    q = pack_bits(big)[:nq]
    t0 = time.perf_counter(); ix.search(q, 10)
    return round(1000 * (time.perf_counter() - t0) / nq, 4)


def xm_avg(xm_dict, txt_h, bi, img_io):
    galx = pack_bits(img_io[bi]["binary"].detach().cpu().numpy())
    ixx = faiss_bin(galx, BITS[bi])
    r10s, maps = [], []
    for L, d in xm_dict["per_lang"].items():
        with torch.no_grad():
            qc = txt_h(nrm(d["text_emb"]).to(dev))[bi]["binary"].detach().cpu().numpy()
        r10, mp = recall_map(ixx, pack_bits(qc), d["gold"])
        r10s.append(r10); maps.append(mp)
    return round(float(np.mean(r10s)), 2), round(float(np.mean(maps)), 2)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    img_h, txt_h = train_head()
    with torch.no_grad():
        io_coco = img_h(te_i.to(dev))
        to_en_o = txt_h(te_en_orig.to(dev)); to_en_l = txt_h(te_en_low.to(dev)); to_ko = txt_h(te_ko.to(dev))
        xm_io = img_h(nrm(XM["img_emb"]).to(dev)) if XM is not None else None
        xml_io = img_h(nrm(XML["img_emb"]).to(dev)) if XML is not None else None

    rows = []
    for bi, bits in enumerate(BITS):
        gal = pack_bits(io_coco[bi]["binary"].detach().cpu().numpy()); ix = faiss_bin(gal, bits)
        eno_r, eno_m = recall_map(ix, pack_bits(to_en_o[bi]["binary"].detach().cpu().numpy()), gold_coco)
        enl_r, enl_m = recall_map(ix, pack_bits(to_en_l[bi]["binary"].detach().cpu().numpy()), gold_coco)
        ko_r, ko_m = recall_map(ix, pack_bits(to_ko[bi]["binary"].detach().cpu().numpy()), gold_coco)
        bytes_img = max(1, bits // 8)
        idx_mb = round(50000 * bytes_img / 1e6, 2)
        lat = latency_50k(io_coco[bi]["binary"].detach().cpu().numpy(), bits)

        def add(dataset, lang, preproc, r10, mp):
            rows.append({"task": "instance", "dataset": dataset, "lang": lang, "preproc": preproc, "bits": bits,
                         "bytes_per_img": bytes_img, "R10": r10, "mAP10": mp, "idx_mb_50k": idx_mb,
                         "latency_ms_50k": lat})
        add("coco", "en", "orig", eno_r, eno_m)
        add("coco", "en", "lower", enl_r, enl_m)
        add("coco", "ko", "orig(caseless)", ko_r, ko_m)
        if XM is not None:
            xo_r, xo_m = xm_avg(XM, txt_h, bi, xm_io); add("xm3600", "avg36", "orig", xo_r, xo_m)
        if XML is not None:
            xl_r, xl_m = xm_avg(XML, txt_h, bi, xml_io); add("xm3600", "avg36", "lower", xl_r, xl_m)
        print(f"[h-lc] bit {bits}: COCO EN orig {eno_r} -> lower {enl_r} | KO {ko_r} | "
              f"XM avg36 orig {xo_r if XM else '-'} -> lower {xl_r if XML else '-'} | "
              f"{bytes_img}B idx {idx_mb}MB lat {lat}ms", flush=True)

    cols = ["task", "dataset", "lang", "preproc", "bits", "bytes_per_img", "R10", "mAP10", "idx_mb_50k",
            "latency_ms_50k"]
    with open(PAPER / "bits_extreme_lc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    en1024 = [r for r in rows if r["dataset"] == "coco" and r["lang"] == "en" and r["bits"] == 1024]
    print(f"[h-lc] ANCHOR 1024 COCO EN: {en1024} (lower should ~81.2)", flush=True)
    print("[h-lc] RESULT_JSON " + json.dumps({"epochs": EPOCHS, "bits": BITS, "rows": rows}), flush=True)
    print(f"[h-lc] DONE -> paper/bits_extreme_lc.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
