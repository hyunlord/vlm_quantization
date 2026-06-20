"""Reviewer experiment item #1 — off-the-shelf baseline. "How far does naive binarization of a stock
multilingual CLIP get, without SigLIP2 + our hash head?" EVAL only, eval_korean 5K, own gallery per model.

(a) naive: sign() on L2-normalized RAW image/text embeddings (NO trained head) -> 1-bit at the model's
    native dim. "just binarize the stock model."
(b) head: that model as a backbone for our hash head -> 1024-bit (= Ext③ backbone result, reused).

Stock models (M-CLIP / MobileCLIP were the requested picks but both fail to load under the env's
transformers 5.1 — M-CLIP meta-device init, like jina; per the work order we substitute transformers-
native stock CLIPs):
  - AltCLIP-m18  (multilingual, XLM-R+CLIP-ViT-L, dim 1024) — naive from the Ext③ cache; head = Ext③.
  - openai/clip-vit-base-patch32 (small, EN-only, dim 512) — naive (embed test); the MobileCLIP-role aux.

Reference rows (existing pipeline): So400m server 1-bit 79.92/71.08; SigLIP2+MiniLM head-adapt 78.16/65.44;
e5-small deployed offline 74.0/66.2; AltCLIP head (Ext③) 77.96/70.52.

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_baseline.py
"""
from __future__ import annotations

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

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
_ID = re.compile(r"_0*(\d+)\.jpg")


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_ks(ix, q, gold, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]), 2) for k in ks}


def naive_eval(img_emb, txt_en, txt_ko, gold):
    """sign() on L2-normalized raw emb -> 1-bit; own gallery (image codes); text->image R@K."""
    dim = img_emb.shape[1]
    gal = pack_bits(F.normalize(torch.as_tensor(img_emb).float(), dim=1).numpy())
    ix = faiss_bin(gal, dim)
    out = {}
    for L, t in (("EN", txt_en), ("KO", txt_ko)):
        if t is None:
            continue
        q = pack_bits(F.normalize(torch.as_tensor(t).float(), dim=1).numpy())
        out[L] = recall_ks(ix, q, gold)
    return dim, out


def coco_test_paths(te_ids):
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    return [f'{REPO}/data/coco/{dco[i]["filepath"]}/{dco[i]["filename"]}' for i in te_ids]


def main():
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); mm = _ID.search(e["image_path"])
        if mm:
            kf[int(mm.group(1))] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    gold = list(range(len(te_ids)))
    rows = []

    def add_naive(label, family, img, en, ko):
        dim, R = naive_eval(img, en, ko, gold)
        row = {"model": label, "family": family, "variant": "naive (sign, no head)", "dim": dim,
               "EN_R10": R.get("EN", {}).get(10, ""), "KO_R10": R.get("KO", {}).get(10, ""),
               "EN_R1": R.get("EN", {}).get(1, ""), "KO_R1": R.get("KO", {}).get(1, "")}
        rows.append(row)
        print(f"[base] {label} naive (dim {dim}): EN R@10 {row['EN_R10']} | KO R@10 {row['KO_R10']}", flush=True)

    # so400m naive (reference: raw SigLIP2 binarized, NO head) — from cache
    add_naive("SigLIP2-So400m", "SigLIP2-so400m", EC["test"]["img"].float().numpy(),
              EC["test"]["txt"].float().numpy(), KO["txt_emb"].float().numpy())

    # AltCLIP-m18 naive — from Ext③ cache (free)
    alt = "/tmp/bb_altclip-m18_test.pt"
    if os.path.exists(alt):
        A = torch.load(alt, map_location="cpu")
        add_naive("AltCLIP-m18 (stock mCLIP)", "AltCLIP-XLMR", A["img"].float().numpy(),
                  A["en"].float().numpy(), A["ko"].float().numpy())
    else:
        print("[base] SKIP AltCLIP naive — cache missing", flush=True)

    # openai/clip-vit-base-patch32 naive — small EN-only stock CLIP (embed test 5K)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizerFast
    from PIL import Image
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    cip = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ctok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    paths = coco_test_paths(te_ids)

    @torch.no_grad()
    def clip_img(paths, batch=64):
        out, t0 = [], time.perf_counter()
        for s in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
            pv = cip(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            out.append(cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float().cpu())
        print(f"  clip img {len(paths)} in {time.perf_counter()-t0:.0f}s", flush=True)
        return torch.cat(out).numpy()

    @torch.no_grad()
    def clip_txt(strings, batch=256):
        out = []
        for s in range(0, len(strings), batch):
            t = ctok(strings[s:s + batch], padding=True, truncation=True, max_length=77, return_tensors="pt")
            out.append(cm.text_projection(cm.text_model(
                input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).pooler_output).float().cpu())
        return torch.cat(out).numpy()

    add_naive("CLIP-ViT-B/32 (stock, EN-only)", "openai-CLIP", clip_img(paths), clip_txt(en_caps), clip_txt(ko_caps))

    # reference rows (existing pipeline)
    for label, fam, var, en, ko in (
        ("SigLIP2-So400m", "SigLIP2-so400m", "server (head, 1024-bit)", 79.92, 71.08),
        ("SigLIP2+MiniLM", "head-adapt offline", "head-adapt (1024-bit)", 78.16, 65.44),
        ("SigLIP2+e5-small", "head-adapt offline (deployed)", "head-adapt (1024-bit)", 74.0, 66.2),
        ("AltCLIP-m18", "AltCLIP-XLMR", "head (1024-bit, Ext③)", 77.96, 70.52),
    ):
        rows.append({"model": label, "family": fam, "variant": var, "dim": 1024,
                     "EN_R10": en, "KO_R10": ko, "EN_R1": "", "KO_R1": ""})

    cols = ["model", "family", "variant", "dim", "EN_R10", "KO_R10", "EN_R1", "KO_R1"]
    with open(PAPER / "baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[base] RESULT_JSON " + json.dumps(rows, ensure_ascii=False), flush=True)
    print(f"[base] DONE -> paper/baseline.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
