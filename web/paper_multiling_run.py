"""Extension ② — multilingual eval (XM3600 primary; XTD10 fallback). EVAL ONLY — no retraining,
no index rebuild, no img_h / common.py / existing-head changes.

A NEW image gallery is built from the benchmark images via the so400m vision tower -> FROZEN ft113
img_h -> packed 1024-bit codes. The offline text heads are aligned to that ft113 img_h code space,
so the SAME gallery serves every encoder. Then for each (encoder x language) the language's captions
are encoded with that encoder's TRAINED config (native pooling + the prefix stored in its head file)
-> packed codes -> Hamming top-K vs the gallery -> text->image R@{1,5,10}.

Encoders (rows):
  so400m-float (ceiling)    so400m text vs so400m image cosine (no hash) — calibration ceiling
  server (so400m+ft113)     so400m text -> ft113 txt_h (deployed server head)
  e5-small (C1, no-prefix)  intfloat/multilingual-e5-small                 -> /tmp/txt_h_e5.pt
  MiniLM-L12-v2             sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -> /tmp/txt_h_paraphrase-...pt
  e5-base (query:)          intfloat/multilingual-e5-base ("query: ")      -> /tmp/txt_h_multilingual-e5-base.pt

The offline heads were trained in Extension ① (same recipe/hp); each .pt records its student model
and the prefix it was trained with, so this script reuses them verbatim — NO retraining.

Data (DGX) — see web/paper_multiling_fetch.sh:
  # XM3600 (primary): captions.jsonl + images/{key}.jpg under data/xm3600
  # XTD10  (fallback): caption .txt files; images are local COCO (data/coco/{train,val}2014)

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_multiling_run.py --bench xm3600 --data data/xm3600
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_multiling_run.py --bench xtd \
      --data data/xtd10_src/XTD10 --coco-root data/coco
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob as _glob
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

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)

# offline heads trained in Extension ①; each .pt stores student + prefix + embed/hidden/bits.
OFFLINE = [
    {"label": "e5-small (C1, no-prefix)", "head": "/tmp/txt_h_e5.pt", "family": "e5/XLM-R"},
    {"label": "MiniLM-L12-v2", "head": "/tmp/txt_h_paraphrase-multilingual-MiniLM-L12-v2.pt",
     "family": "paraphrase/XLM-R"},
    {"label": "e5-base (query:)", "head": "/tmp/txt_h_multilingual-e5-base.pt", "family": "e5/XLM-R"},
]
SERVER = "server (so400m+ft113)"
FLOAT = "so400m-float (ceiling)"

# representative subset highlighted in the paper table; the full CSV keeps every language present.
REPR_LANGS = ["en", "ko", "ja", "zh", "ar", "ru", "hi", "th", "de", "fr", "es", "tr", "vi", "id"]


# ---- index + recall (mirrors web/eval_paper.py exactly) ----
def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits)
    ix.add(np.ascontiguousarray(packed))
    return ix


def recall_ks(ix, q, gold_rows, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold_rows[i] in I[i, :k] for i in range(len(gold_rows))]), 2) for k in ks}


def _pool(out):
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(1)


# ---- data loaders -> (image_paths: list[Path], per_lang: dict[lang] -> (caps, cap_img_rows)) ----
def load_xm3600(data_dir, langs=None, limit_images=0):
    data = Path(data_dir)
    capf = data / "captions.jsonl"
    if not capf.exists():
        cands = list(data.rglob("captions.jsonl"))
        if cands:
            capf = cands[0]
    imgmap = {p.stem: p for p in (data / "images").rglob("*.jpg")} if (data / "images").exists() \
        else {p.stem: p for p in data.rglob("*.jpg")}
    recs = [json.loads(l) for l in open(capf, encoding="utf-8") if l.strip()]
    if limit_images:
        recs = recs[:limit_images]
    image_paths, key2idx, per_lang = [], {}, {}
    for rec in recs:
        key = rec.get("image/key")
        p = imgmap.get(key)
        if p is None:
            continue
        key2idx[key] = len(image_paths)
        image_paths.append(p)
    for rec in recs:
        key = rec.get("image/key")
        if key not in key2idx:
            continue
        idx = key2idx[key]
        for lang, val in rec.items():
            if not isinstance(val, dict):
                continue
            caps = val.get("caption") or []
            if langs and lang not in langs:
                continue
            d = per_lang.setdefault(lang, ([], []))
            for cap in caps:
                d[0].append(cap)
                d[1].append(idx)
    return image_paths, per_lang


def load_xtd(data_dir, coco_root, langs=None, limit_images=0):
    data, coco = Path(data_dir), Path(coco_root)
    names = [l.strip() for l in open(data / "test_image_names.txt", encoding="utf-8") if l.strip()]
    if limit_images:
        names = names[:limit_images]

    def coco_path(fn):
        split = "train2014" if "train2014" in fn else "val2014"
        return coco / split / fn

    image_paths = [coco_path(fn) for fn in names]
    per_lang = {}
    for capfile in sorted(_glob.glob(str(data / "test_1kcaptions_*.txt"))):
        lang = Path(capfile).stem.replace("test_1kcaptions_", "")
        if langs and lang not in langs:
            continue
        caps = [l.strip() for l in open(capfile, encoding="utf-8")][:len(names)]
        per_lang[lang] = (caps, list(range(len(caps))))
    return image_paths, per_lang


# ---- embedding ----
@torch.no_grad()
def so400m_image_emb(model, processor, image_paths, dev, dtype, batch=64):
    out, t0, M = [], time.perf_counter(), len(image_paths)
    for s in range(0, M, batch):
        imgs = [Image.open(p).convert("RGB") for p in image_paths[s:s + batch]]
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype=dtype)
        out.append(_pool(model.vision_model(pixel_values=px)).float().cpu())
        if (s // batch) % 10 == 0:
            print(f"  img {min(s + batch, M)}/{M} ({(s + len(imgs)) / (time.perf_counter() - t0 + 1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


@torch.no_grad()
def so400m_text_emb(model, tok, strings, dev, dtype, batch=256, maxlen=64):
    out = []
    for s in range(0, len(strings), batch):
        t = tok(strings[s:s + batch], padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        e = _pool(model.text_model(input_ids=t["input_ids"].to(dev),
                                   attention_mask=am.to(dev) if am is not None else None)).float().cpu()
        out.append(e)
    return torch.cat(out)


@torch.no_grad()
def offline_emb(model, tok, strings, prefix, dev, batch=256, maxlen=64):
    out = []
    for s in range(0, len(strings), batch):
        txt = [prefix + x for x in strings[s:s + batch]]
        t = tok(txt, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        o = model(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
        msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
        e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=1).float().cpu())
    return torch.cat(out)


def float_t2i(txt_norm, img_norm, gold, ks=KS):
    topk = (txt_norm @ img_norm.t()).topk(max(ks), dim=1).indices.numpy()
    g = np.asarray(gold)
    return {k: round(100 * np.mean([g[i] in topk[i, :k] for i in range(len(g))]), 2) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["xm3600", "xtd"], default="xm3600")
    ap.add_argument("--data", required=True)
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--langs", default="", help="comma list; default = all present")
    ap.add_argument("--limit-images", type=int, default=0, help="smoke test: cap gallery size")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default=str(PAPER / "multiling.csv"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if dev == "cpu":
        dtype = torch.float32
    langs = [x for x in args.langs.split(",") if x] or None

    if args.bench == "xm3600":
        image_paths, per_lang = load_xm3600(args.data, langs, args.limit_images)
    else:
        image_paths, per_lang = load_xtd(args.data, args.coco_root, langs, args.limit_images)
    n_img = len(image_paths)
    all_langs = sorted(per_lang.keys())
    if not n_img or not all_langs:
        raise SystemExit(f"[ml] no data: images={n_img} langs={all_langs} (check --data)")
    print(f"[ml] bench={args.bench} images={n_img} langs={len(all_langs)}: {all_langs}", flush=True)
    print("[ml] caps/lang: " + ", ".join(f"{L}:{len(per_lang[L][0])}" for L in all_langs), flush=True)

    enc = Encoder()  # ft113 (HEAD_PATH=/tmp/ft_ko_113.pt): img_h + server txt_h

    # so400m backbone — gallery image emb + server text emb + float ceiling
    from transformers import AutoModel
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    except Exception:
        from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
        processor = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(SIGLIP_MODEL),
                                    tokenizer=GemmaTokenizer.from_pretrained(SIGLIP_MODEL))
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()
    tok_s = processor.tokenizer

    img_emb = so400m_image_emb(bb, processor, image_paths, dev, dtype)  # (n_img, 1152) raw float32
    print(f"[ml] gallery image emb {tuple(img_emb.shape)}", flush=True)
    gal = enc.image_codes_packed(img_emb.numpy())          # _prep(L2)->img_h->pack (frozen ft113)
    gal_ix = faiss_bin(gal, CODE_BITS)
    img_norm = F.normalize(img_emb, dim=1)                 # float ceiling reference

    results = {}  # (encoder_label, lang) -> {k: R@k}
    for L in all_langs:
        caps, cap_img = per_lang[L]
        te = so400m_text_emb(bb, tok_s, caps, dev, dtype)  # (C, 1152) raw
        results[(SERVER, L)] = recall_ks(gal_ix, pack_bits(enc._codes_pm1(enc.txt_h, te)), cap_img)
        results[(FLOAT, L)] = float_t2i(F.normalize(te, dim=1), img_norm, cap_img)
        print(f"[ml] {L}: server R@10 {results[(SERVER, L)][10]} | float R@10 {results[(FLOAT, L)][10]}", flush=True)

    del bb
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    for spec in OFFLINE:
        head = spec["head"]
        if not os.path.exists(head):
            print(f"[ml] SKIP {spec['label']} — head missing: {head}", flush=True)
            continue
        ck = torch.load(head, map_location="cpu")
        student, prefix = ck["student"], (ck.get("prefix") or "")
        bits = [int(b) for b in ck["bits"]]
        obi = bits.index(CODE_BITS)
        txt_h = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0)
        txt_h.load_state_dict(ck["txt_h"])
        txt_h.to(dev).eval()
        from transformers import AutoModel as _AM, AutoTokenizer
        m = _AM.from_pretrained(student).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(student)
        print(f"[ml] === {spec['label']} ({student}, prefix={prefix!r}) ===", flush=True)
        for L in all_langs:
            caps, cap_img = per_lang[L]
            e = offline_emb(m, tok, caps, prefix, dev)
            with torch.no_grad():
                codes = pack_bits(txt_h(e.to(dev))[obi]["binary"].cpu().numpy())
            results[(spec["label"], L)] = recall_ks(gal_ix, codes, cap_img)
            print(f"[ml]   {spec['label']} {L}: R@10 {results[(spec['label'], L)][10]}", flush=True)
        del m, txt_h
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- write CSV (language x encoder) + figure JSON ----
    PAPER.mkdir(parents=True, exist_ok=True)
    enc_order = [FLOAT, SERVER] + [s["label"] for s in OFFLINE]
    fam = {FLOAT: "SigLIP2-so400m", SERVER: "SigLIP2-so400m"}
    fam.update({s["label"]: s["family"] for s in OFFLINE})
    cols = ["language", "encoder", "family", "n_captions", "n_images", "R1", "R5", "R10"]
    out_rows = []
    for L in all_langs:
        for lab in enc_order:
            if (lab, L) not in results:
                continue
            r = results[(lab, L)]
            out_rows.append({"language": L, "encoder": lab, "family": fam[lab],
                             "n_captions": len(per_lang[L][0]), "n_images": n_img,
                             "R1": r[1], "R5": r[5], "R10": r[10]})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    figj = {"bench": args.bench, "n_images": n_img, "langs": all_langs, "repr": REPR_LANGS,
            "data": {lab: {L: results[(lab, L)] for L in all_langs if (lab, L) in results} for lab in enc_order}}
    json.dump(figj, open(str(Path(args.out).with_suffix(".json")), "w", encoding="utf-8"), ensure_ascii=False)
    print("[ml] RESULT_JSON " + json.dumps(figj, ensure_ascii=False), flush=True)
    print(f"[ml] DONE -> {args.out} ({len(out_rows)} rows, {len(all_langs)} langs)", flush=True)


if __name__ == "__main__":
    main()
