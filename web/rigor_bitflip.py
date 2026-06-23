"""AAAI rigor §2 (Pillar 1) — precision / bit-flip sensitivity, eval-only.

Where does numeric precision matter for a deployed binary code? We quantize at two points and
measure (a) bit-flip% vs the fp32 reference code and (b) R@10 loss (COCO 5K, I2T, deployed ft113):
  - "encoder-output": quantize the input SigLIP2 feature (int8/fp16/fp32), then fp32 head → sign.
  - "head-output":    fp32 feature → fp32 head → quantize the pre-sign z (int8/fp16/fp32) → sign.
Quant: fp16 = round-trip half; int8 = symmetric per-tensor (scale=max|x|/127). Hypothesis: encoder-output
int8 is tolerable, head-output int8 is destructive (pre-sign margins are tiny → many near-zero bits flip).

Run on DGX:
  .venv/bin/python web/rigor_bitflip.py --out paper/rigor_bitflip.csv
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
TEST_CACHE = os.environ.get("TEST_CACHE", "/tmp/emb_cache.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def head_z1024(head, x):
    raw = head.hash_head(x)
    return F.normalize(head.batch_norms[-1](raw[:, :head.bit_list[-1]]), p=2, dim=1)


def q_int8(x):
    scale = x.abs().max() / 127.0
    return torch.round(x / scale).clamp(-127, 127) * scale


def q_fp16(x):
    return x.half().float()


def quant(x, prec):
    return {"fp32": x, "fp16": q_fp16(x), "int8": q_int8(x)}[prec]


def pack(b01):
    return np.ascontiguousarray(np.packbits(b01.astype(np.uint8), axis=1, bitorder="big"), dtype=np.uint8)


def r10(qb, gb):
    import faiss
    ix = faiss.IndexBinaryFlat(qb.shape[1]); ix.add(pack(gb))
    _, I = ix.search(pack(qb), 10)
    return round(100 * np.mean([r in I[r] for r in range(qb.shape[0])]), 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/rigor_bitflip.csv")
    args = ap.parse_args()
    ck = torch.load(HEAD_PATH, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    img_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0).to(dev).eval()
    img_h.load_state_dict(ck["img_h"]); txt_h.load_state_dict(ck["txt_h"])
    EC = torch.load(TEST_CACHE, map_location="cpu")["test"]
    img = F.normalize(EC["img"].float(), dim=1).to(dev)
    txt = F.normalize(EC["txt"].float(), dim=1).to(dev)
    with torch.no_grad():
        # fp32 reference codes
        zi_ref = (head_z1024(img_h, img) > 0).cpu().numpy()
        gt_ref = (head_z1024(txt_h, txt) > 0).cpu().numpy()
    base_r10 = r10(zi_ref, gt_ref)
    print(f"[ref] fp32 R@10 I2T {base_r10}", flush=True)

    rows = []
    for point in ("encoder-output", "head-output"):
        for prec in ("fp32", "fp16", "int8"):
            with torch.no_grad():
                if point == "encoder-output":
                    qi = F.normalize(quant(img, prec), dim=1)
                    qt = F.normalize(quant(txt, prec), dim=1)
                    zi = (head_z1024(img_h, qi) > 0).cpu().numpy()
                    gt = (head_z1024(txt_h, qt) > 0).cpu().numpy()
                else:  # head-output: quantize the pre-sign z
                    zi = (quant(head_z1024(img_h, img), prec) > 0).cpu().numpy()
                    gt = (quant(head_z1024(txt_h, txt), prec) > 0).cpu().numpy()
            flip_i = round(100 * float((zi != zi_ref).mean()), 3)
            flip_t = round(100 * float((gt != gt_ref).mean()), 3)
            rr = r10(zi, gt)
            rows.append({"quant_point": point, "precision": prec,
                         "bitflip_pct_img": flip_i, "bitflip_pct_txt": flip_t,
                         "R10_I2T": rr, "R10_drop": round(base_r10 - rr, 2)})
            print(f"[bf] {point:>14} {prec}: flip img {flip_i}% txt {flip_t}% | R@10 {rr} (drop {round(base_r10-rr,2)})", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quant_point", "precision", "bitflip_pct_img", "bitflip_pct_txt", "R10_I2T", "R10_drop"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
