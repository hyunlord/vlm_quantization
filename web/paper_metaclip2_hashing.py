"""(B) MetaCLIP2 backbone hashing — does our 1-bit Matryoshka recipe ride the current SoTA multilingual
backbone? Same structure + budget as the NLLB-CLIP experiment (batch #2 f) for a fair head-to-head.

MetaCLIP2-worldwide (ViT-H-14-worldwide, the multilingual variant, XM3600 SoTA) loaded via open_clip
(same path as NLLB-CLIP). Frozen backbone, head-only: encode a COCO-train subset (MetaCLIP2 image tower +
en/ko text) -> train Matryoshka hash head (img_h+txt_h, [8..1024], same arch/loss/budget as NLLB) -> eval
text->image R@{1,5,10}+mAP@10 on COCO 5K (en,ko) + XM3600 36-lang, plus MetaCLIP2 float (no head) ceiling.

Native preprocessing (open_clip's multilingual tokenizer handles casing itself — NOT force-lowercased; only
SigLIP-family text gets `.lower()`). Reuses NLLB script's shared helpers (data, train_head, eval).

Run on DGX: TRAIN_N=30000 EPOCHS=25 .venv/bin/python web/paper_metaclip2_hashing.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.paper_nllb_hashing import (pack_bits, faiss_bin, recall_map_bin, rk_map_float,  # noqa: E402
                                    coco_train_pairs, coco_test, train_head, head_codes, BITS, MB)
from web.paper_baselines_multiling import xm3600_data  # noqa: E402

PAPER = Path(REPO) / "paper"
REPR = ["de", "te", "th", "hi", "en", "ko"]
TRAIN_N = int(os.environ.get("TRAIN_N", "30000"))
TRAIN_CACHE = "/tmp/metaclip2_coco_train.pt"
MC_NAME = os.environ.get("MC_NAME", "ViT-H-14-worldwide")
MC_PRET = os.environ.get("MC_PRET", "metaclip2_worldwide")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_metaclip2():
    import open_clip
    # metaclip2_worldwide weights are trained with QuickGELU; the bare ViT-H-14-worldwide config builds
    # exact GELU -> open_clip only WARNS and would silently use the wrong activation (corrupt embeddings).
    # force_quick_gelu=True matches the pretrained tag.
    model, _, preprocess = open_clip.create_model_and_transforms(MC_NAME, pretrained=MC_PRET,
                                                                 force_quick_gelu=True)
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer(MC_NAME)
    print(f"[B] loaded MetaCLIP2 {MC_NAME}/{MC_PRET} (force_quick_gelu=True)", flush=True)
    return model, preprocess, tok


def _ac():
    # bf16 autocast on cuda — ViT-H is large + the GB10 (sm_121) exceeds this torch's max cap; bf16 uses
    # tensor cores and is harmless for retrieval embeddings (we hash to 1-bit anyway).
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else \
        __import__("contextlib").nullcontext()


@torch.no_grad()
def mc_img(model, preprocess, paths, batch=64, tag="img"):
    out, t0 = [], time.perf_counter()
    for s in range(0, len(paths), batch):
        px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(dev)
        with _ac():
            e = model.encode_image(px)
        out.append(e.float().cpu())
        if (s // batch) % 20 == 0:
            print(f"  [mc {tag}] {min(s+batch,len(paths))}/{len(paths)} "
                  f"({(s+batch)/(time.perf_counter()-t0+1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


@torch.no_grad()
def mc_txt(model, tok, strings, batch=256):
    """native multilingual tokenizer (no per-lang src code, no forced lowercase)."""
    out = []
    for s in range(0, len(strings), batch):
        toks = tok(strings[s:s + batch]).to(dev)
        with _ac():
            e = model.encode_text(toks)
        out.append(e.float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xm3600")
    args = ap.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)
    model, preprocess, tok = load_metaclip2()
    bi = BITS.index(MB)

    # ---- training data: MetaCLIP2 emb of COCO-train subset (cache) ----
    if os.path.exists(TRAIN_CACHE):
        tc = torch.load(TRAIN_CACHE, map_location="cpu")
        img_e, en_e, ko_rows, ko_e_p = tc["img"], tc["en"], tc["ko_rows"], tc["ko"]
        print(f"[B] loaded train cache img {tuple(img_e.shape)}", flush=True)
    else:
        paths, en, ko = coco_train_pairs(TRAIN_N)
        print(f"[B] encoding {len(paths)} COCO-train images with MetaCLIP2 ...", flush=True)
        img_e = mc_img(model, preprocess, paths, tag="train")
        en_e = mc_txt(model, tok, en)
        ko_rows = [i for i, k in enumerate(ko) if k]
        ko_e_p = mc_txt(model, tok, [ko[i] for i in ko_rows])
        torch.save({"img": img_e, "en": en_e, "ko_rows": ko_rows, "ko": ko_e_p}, TRAIN_CACHE)
    embed = img_e.shape[1]
    tr_img = torch.cat([img_e, img_e[torch.tensor(ko_rows)]], 0)
    tr_txt = torch.cat([en_e, ko_e_p], 0)
    print(f"[B] train pairs: {tr_img.shape[0]} (en {img_e.shape[0]} + ko {len(ko_rows)}) embed={embed}", flush=True)
    img_h, txt_h = train_head(tr_img, tr_txt, embed)

    rows = []

    def emit(model_lbl, space, dataset, lang, ncaps, tup):
        rows.append({"model": model_lbl, "space": space, "dataset": dataset, "lang": lang,
                     "n_caps": ncaps, "R1": tup[0], "R5": tup[1], "R10": tup[2], "mAP10": tup[3]})

    # ---- COCO 5K ----
    cpaths, ccaps = coco_test()
    print("[B] encoding COCO test 5K images with MetaCLIP2 ...", flush=True)
    cimg = mc_img(model, preprocess, cpaths, tag="cocotest")
    cimg_n = F.normalize(cimg, dim=1).to(dev)
    gal = pack_bits(head_codes(img_h, cimg, bi)); ix = faiss_bin(gal, MB)
    gold = list(range(len(cpaths)))
    for L in ("en", "ko"):
        te = mc_txt(model, tok, ccaps[L])
        emit("MetaCLIP2+head", "1bit", "coco", L, len(ccaps[L]),
             recall_map_bin(ix, pack_bits(head_codes(txt_h, te, bi)), gold))
        emit("MetaCLIP2 float", "float", "coco", L, len(ccaps[L]),
             rk_map_float(F.normalize(te, dim=1).to(dev), cimg_n, gold))

    # ---- XM3600 36-lang ----
    ds = xm3600_data(args.data)
    print(f"[B] encoding XM3600 {ds['n_img']} images with MetaCLIP2 ...", flush=True)
    ximg = mc_img(model, preprocess, ds["paths"], tag="xm")
    ximg_n = F.normalize(ximg, dim=1).to(dev)
    galx = pack_bits(head_codes(img_h, ximg, bi)); ixx = faiss_bin(galx, MB)
    h_all, f_all = [], []
    for L in ds["langs"]:
        caps, g = ds["caps"][L], ds["gold"][L]
        te = mc_txt(model, tok, caps)
        hb = recall_map_bin(ixx, pack_bits(head_codes(txt_h, te, bi)), g)
        ff = rk_map_float(F.normalize(te, dim=1).to(dev), ximg_n, g)
        h_all.append(hb); f_all.append(ff)
        if L in REPR:
            emit("MetaCLIP2+head", "1bit", "xm3600", L, len(caps), hb)
            emit("MetaCLIP2 float", "float", "xm3600", L, len(caps), ff)
        print(f"[B] xm {L}: MetaCLIP2+head R@10 {hb[2]} | float {ff[2]}", flush=True)

    def avg(rs):
        return tuple(round(float(np.mean([r[j] for r in rs])), 2) for j in range(4))
    h_avg, f_avg = avg(h_all), avg(f_all)
    emit("MetaCLIP2+head", "1bit", "xm3600", "avg36", "", h_avg)
    emit("MetaCLIP2 float", "float", "xm3600", "avg36", "", f_avg)

    cols = ["model", "space", "dataset", "lang", "n_caps", "R1", "R5", "R10", "mAP10"]
    with open(PAPER / "metaclip2_hashing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[B] SANITY MetaCLIP2 float XM3600: avg36 R@10 {f_avg[2]} | de/te/th/hi = "
          f"{[round(float(np.mean([f_all[i][2]])),1) for i,L in enumerate(ds['langs']) if L in ('de','te','th','hi')]} "
          f"(multilingual float should be strong on low-resource where SigLIP2 collapses)", flush=True)
    print("[B] RESULT_JSON " + json.dumps({"embed": embed, "rows": rows, "float_avg36": f_avg,
          "head_avg36": h_avg}, ensure_ascii=False), flush=True)
    print(f"[B] DONE -> paper/metaclip2_hashing.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
