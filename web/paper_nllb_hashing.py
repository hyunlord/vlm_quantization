"""(f) NLLB-CLIP backbone hashing — "does our 1-bit Matryoshka recipe ride a multilingual backbone?"

Hypothesis: applying our hash-head recipe to a multilingual-native backbone (NLLB-CLIP) yields a
multilingual 1-bit system that inherits NLLB's low-resource-language strength — and a head trained on
en(+ko) ONLY preserves de/te/th/hi multilinguality after binarization (the head adapts to the embedding
SPACE, not the language). NO new-language training data (multilingual alignment is already in frozen NLLB).

Pipeline (frozen NLLB-CLIP backbone, head-only):
  1. Encode a COCO-train subset with NLLB image tower (cache) + NLLB text emb of en(+ko) captions.
  2. Train Matryoshka hash head (img_h+txt_h, bit_list [8..1024], same arch/loss as ft113; no aug views
     -> consistency term inactive) on the (NLLB img, NLLB en/ko txt) InfoNCE pairs.
  3. Eval text->image R@{1,5,10}+mAP@10: COCO 5K (en,ko) and XM3600 36-lang (NLLB text->head->1bit vs
     NLLB image gallery codes). Also NLLB float (no head) for the binarization-loss / anchor check.

Compare: NLLB+head(1bit) vs NLLB float (batch#1 a) vs SigLIP2+ft113(1bit) (batch#1 a).
NLLB load reuses batch#1: open_clip create_model_from_pretrained("nllb-clip-base-siglip","v1") +
NllbTokenizerFast (open_clip's tokenizer is tf5.1-broken).

Run on DGX: TRAIN_N=30000 .venv/bin/python web/paper_nllb_hashing.py --data data/xm3600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.paper_baselines_multiling import xm3600_data, NLLB_LANG  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
REPR = ["de", "te", "th", "hi", "en", "ko"]
BITS = [int(x) for x in os.environ.get("BITS", "8,16,32,64,128,256,512,1024").split(",")]
MB = BITS[-1]
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = 512
TRAIN_N = int(os.environ.get("TRAIN_N", "30000"))
TRAIN_CACHE = "/tmp/nllb_coco_train.pt"
_ID = re.compile(r"_0*(\d+)\.jpg")
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


def recall_map_bin(ix, q, gold):
    _, I = ix.search(q, 10)
    r = {k: round(100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]), 2) for k in KS}
    aps = [(1.0 / (np.where(I[i, :10] == gold[i])[0][0] + 1) if gold[i] in I[i, :10] else 0.0)
           for i in range(len(gold))]
    return r[1], r[5], r[10], round(100 * float(np.mean(aps)), 2)


def rk_map_float(txt_n, img_n, gold, chunk=512):
    g = np.asarray(gold); R = {k: 0 for k in KS}; aps = []
    for s in range(0, txt_n.shape[0], chunk):
        idx = (txt_n[s:s + chunk] @ img_n.t()).topk(max(KS), dim=1).indices.cpu().numpy()
        for j in range(idx.shape[0]):
            gj = g[s + j]
            for k in KS:
                R[k] += gj in idx[j, :k]
            aps.append(1.0 / (np.where(idx[j, :10] == gj)[0][0] + 1) if gj in idx[j, :10] else 0.0)
    n = txt_n.shape[0]
    return (round(100 * R[1] / n, 2), round(100 * R[5] / n, 2), round(100 * R[10] / n, 2),
            round(100 * float(np.mean(aps)), 2))


# ---- NLLB-CLIP loader + encoders ----
def load_nllb():
    import open_clip
    from transformers import NllbTokenizerFast
    model, preprocess = open_clip.create_model_from_pretrained("nllb-clip-base-siglip", "v1")
    model = model.to(dev).eval()
    tok = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    return model, preprocess, tok


@torch.no_grad()
def nllb_img(model, preprocess, paths, batch=64, tag="img"):
    out, t0 = [], time.perf_counter()
    for s in range(0, len(paths), batch):
        px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(dev)
        out.append(model.encode_image(px).float().cpu())
        if (s // batch) % 20 == 0:
            print(f"  [nllb {tag}] {min(s+batch,len(paths))}/{len(paths)} "
                  f"({(s+batch)/(time.perf_counter()-t0+1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


@torch.no_grad()
def nllb_txt(model, tok, strings, lang, batch=256):
    tok.src_lang = NLLB_LANG.get(lang, "eng_Latn")
    out = []
    for s in range(0, len(strings), batch):
        enc = tok(strings[s:s + batch], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        out.append(model.encode_text(enc["input_ids"].to(dev)).float().cpu())
    return torch.cat(out)


# ---- COCO data ----
def coco_train_pairs(n):
    """first n COCO train/restval images: paths + en caption + ko caption (by cocoid)."""
    D = json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        if m and e.get("captions"):
            kf[int(m.group(1))] = e["captions"][0]
    paths, en, ko = [], [], []
    for im in D:
        if im["split"] not in ("train", "restval"):
            continue
        cid = im["cocoid"]
        paths.append(f'{REPO}/data/coco/{im["filepath"]}/{im["filename"]}')
        en.append(im["sentences"][0]["raw"])
        ko.append(kf.get(cid, ""))
        if len(paths) >= n:
            break
    return paths, en, ko


def coco_test():
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        if m:
            kf[int(m.group(1))] = e.get("captions", [])
    ko = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    paths = [f'{REPO}/data/coco/{dco[i]["filepath"]}/{dco[i]["filename"]}' for i in te_ids]
    return paths, {"en": en, "ko": ko}


def train_head(img_e, txt_e, embed):
    """img_e,txt_e row-aligned NLLB emb pairs -> img_h,txt_h (no aug -> consistency inactive)."""
    P = json.load(open("/tmp/hp_results.json"))["best_params"]
    torch.manual_seed(42)
    img_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    txt_h = NestedHashLayer(embed, P["hidden"], BITS, P["dropout"]).to(dev)
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    I, T = F.normalize(img_e, dim=1), F.normalize(txt_e, dim=1)
    Ntr = I.shape[0]; steps = EPOCHS * (Ntr // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=max(steps, 1), pct_start=0.3)
    img_h.train(); txt_h.train(); lf.train(); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        perm = torch.randperm(Ntr)
        for s in range(0, Ntr - BS + 1, BS):
            idx = perm[s:s + BS]
            out = lf(img_h(I[idx].to(dev)), txt_h(T[idx].to(dev)), progress=g / max(steps, 1))
            opt.zero_grad(); out["total"].backward(); opt.step(); sched.step(); g += 1
    img_h.eval(); txt_h.eval()
    print(f"[f] trained NLLB head ({Ntr} pairs) in {time.perf_counter()-t0:.0f}s", flush=True)
    return img_h, txt_h


@torch.no_grad()
def head_codes(head, emb, bi):
    return head(F.normalize(emb, dim=1).to(dev))[bi]["binary"].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/xm3600")
    args = ap.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)
    model, preprocess, tok = load_nllb()
    bi = BITS.index(MB)

    # ---- training data: NLLB emb of COCO-train subset (cache) ----
    if os.path.exists(TRAIN_CACHE):
        tc = torch.load(TRAIN_CACHE, map_location="cpu")
        img_e, en_e, ko_rows, ko_e_p = tc["img"], tc["en"], tc["ko_rows"], tc["ko"]
        print(f"[f] loaded train cache img {tuple(img_e.shape)}", flush=True)
    else:
        paths, en, ko = coco_train_pairs(TRAIN_N)
        print(f"[f] encoding {len(paths)} COCO-train images with NLLB ...", flush=True)
        img_e = nllb_img(model, preprocess, paths, tag="train")
        en_e = nllb_txt(model, tok, en, "en")
        ko_rows = [i for i, k in enumerate(ko) if k]
        ko_e_p = nllb_txt(model, tok, [ko[i] for i in ko_rows], "ko")
        torch.save({"img": img_e, "en": en_e, "ko_rows": ko_rows, "ko": ko_e_p}, TRAIN_CACHE)
    embed = img_e.shape[1]
    # paired train tensors: (img,en) for all rows + (img,ko) for rows that have a ko caption
    tr_img = torch.cat([img_e, img_e[torch.tensor(ko_rows)]], 0)
    tr_txt = torch.cat([en_e, ko_e_p], 0)
    print(f"[f] train pairs: {tr_img.shape[0]} (en {img_e.shape[0]} + ko {len(ko_rows)}) embed={embed}", flush=True)
    img_h, txt_h = train_head(tr_img, tr_txt, embed)

    rows = []

    def emit(model_lbl, space, path, dataset, lang, ncaps, tup):
        rows.append({"model": model_lbl, "space": space, "path": path, "dataset": dataset, "lang": lang,
                     "n_caps": ncaps, "R1": tup[0], "R5": tup[1], "R10": tup[2], "mAP10": tup[3]})

    # ---- COCO 5K eval ----
    cpaths, ccaps = coco_test()
    print(f"[f] encoding COCO test 5K images with NLLB ...", flush=True)
    cimg = nllb_img(model, preprocess, cpaths, tag="cocotest")
    cimg_n = F.normalize(cimg, dim=1).to(dev)
    gal = pack_bits(head_codes(img_h, cimg, bi)); ix = faiss_bin(gal, MB)
    gold = list(range(len(cpaths)))
    for L in ("en", "ko"):
        te = nllb_txt(model, tok, ccaps[L], L)
        emit("NLLB+head", "1bit", "server", "coco", L, len(ccaps[L]),
             recall_map_bin(ix, pack_bits(head_codes(txt_h, te, bi)), gold))
        emit("NLLB float", "float", "server", "coco", L, len(ccaps[L]),
             rk_map_float(F.normalize(te, dim=1).to(dev), cimg_n, gold))

    # ---- XM3600 eval ----
    ds = xm3600_data(args.data)
    print(f"[f] encoding XM3600 {ds['n_img']} images with NLLB ...", flush=True)
    ximg = nllb_img(model, preprocess, ds["paths"], tag="xm")
    ximg_n = F.normalize(ximg, dim=1).to(dev)
    galx = pack_bits(head_codes(img_h, ximg, bi)); ixx = faiss_bin(galx, MB)
    h_all, f_all = [], []
    for L in ds["langs"]:
        caps, g = ds["caps"][L], ds["gold"][L]
        te = nllb_txt(model, tok, caps, L)
        hb = recall_map_bin(ixx, pack_bits(head_codes(txt_h, te, bi)), g)
        ff = rk_map_float(F.normalize(te, dim=1).to(dev), ximg_n, g)
        h_all.append(hb); f_all.append(ff)
        if L in REPR:
            emit("NLLB+head", "1bit", "server", "xm3600", L, len(caps), hb)
            emit("NLLB float", "float", "server", "xm3600", L, len(caps), ff)
        print(f"[f] xm {L}: NLLB+head R@10 {hb[2]} | NLLB float {ff[2]}", flush=True)

    def avg(rs):
        return tuple(round(float(np.mean([r[j] for r in rs])), 2) for j in range(4))
    emit("NLLB+head", "1bit", "server", "xm3600", "avg36", "", avg(h_all))
    emit("NLLB float", "float", "server", "xm3600", "avg36", "", avg(f_all))

    cols = ["model", "space", "path", "dataset", "lang", "n_caps", "R1", "R5", "R10", "mAP10"]
    with open(PAPER / "nllb_hashing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[f] RESULT_JSON " + json.dumps({"rows": rows}, ensure_ascii=False), flush=True)
    print(f"[f] DONE -> paper/nllb_hashing.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
