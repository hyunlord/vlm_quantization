"""(a) Absolute multilingual baselines on XM3600 (+COCO 5K). text->image R@{1,5,10} + mAP@10.

Pins our claim ("offline head-adapt beats SigLIP2's own text tower on de/te/th/hi") to the
ABSOLUTE numbers of real multilingual retrieval models, each scored in its OWN native float-cosine
space with its OWN image + text towers on the SAME image set.

Rows
  space=1bit (ours, frozen ft113 1024-bit codes):
    Ours: server (so400m+ft113)        so400m text -> ft113 txt_h
    Ours: offline (MiniLM head-adapt)  MiniLM text -> txt_h_MiniLM (best offline, Ext①)
  space=float (cosine):
    SigLIP2 text tower (so400m float)  the encoder our heads sit on — the thing we claim to beat
    NLLB-CLIP-base-siglip              purpose-built 201-lang CLIP (open_clip)        [real SoTA]
    AltCLIP-m18                        XLM-R + CLIP-ViT-L, 18-lang (transformers)     [real SoTA]
    M-CLIP (XLM-R-L / ViT-L-14)        best-effort (tf5.1 meta-device history)        [try/flag]
    jina-clip-v2                       best-effort (tf5.1 meta-device history)        [try/flag]

Datasets: COCO 5K test (en, ko — the langs we have captions for; reuses cached so400m/AltCLIP
embeddings) and XM3600 3600 (all 36 langs; emit en/ko/de/te/th/hi + avg36). COCO server EN/KO R@10
must reproduce the 79.92/71.08 anchor; XM3600 so400m/server/offline must reproduce Ext② multiling.json.

Frozen backbone, EVAL only. Run on DGX (after open_clip install):
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_baselines_multiling.py [--models ours,nllb,altclip,mclip,jina]
"""
from __future__ import annotations

import argparse
import csv
import gc
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

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
_ID = re.compile(r"_0*(\d+)\.jpg")
REPR = ["en", "ko", "de", "te", "th", "hi"]
MINILM_HEAD = "/tmp/txt_h_paraphrase-multilingual-MiniLM-L12-v2.pt"
# XM3600 2-letter -> NLLB FLORES-200 code (for NLLB-CLIP tokenizer; default used if absent)
NLLB_LANG = {"ar": "arb_Arab", "bn": "ben_Beng", "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn",
             "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
             "fil": "tgl_Latn", "fr": "fra_Latn", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn",
             "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
             "mi": "mri_Latn", "nl": "nld_Latn", "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn",
             "quz": "quy_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn",
             "te": "tel_Telu", "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl", "vi": "vie_Latn",
             "zh": "zho_Hans"}


# ---------- metrics ----------
def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def rk_map_bin(ix, q, gold):
    _, I = ix.search(q, max(KS))
    R = {k: 100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]) for k in KS}
    ap = []
    for i in range(len(gold)):
        hit = np.where(I[i, :10] == gold[i])[0]
        ap.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return {1: round(R[1], 2), 5: round(R[5], 2), 10: round(R[10], 2), "map10": round(100 * float(np.mean(ap)), 2)}


def rk_map_float(txt_n, img_n, gold, chunk=512):
    """text->image float cosine R@{1,5,10} + mAP@10 (single gold per query)."""
    g = np.asarray(gold)
    R = {k: 0 for k in KS}; aps = []
    for s in range(0, txt_n.shape[0], chunk):
        sims = txt_n[s:s + chunk] @ img_n.t()
        idx = sims.topk(max(KS), dim=1).indices.cpu().numpy()
        for j in range(idx.shape[0]):
            gj = g[s + j]
            for k in KS:
                if gj in idx[j, :k]:
                    R[k] += 1
            hit = np.where(idx[j, :10] == gj)[0]
            aps.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    n = txt_n.shape[0]
    return {1: round(100 * R[1] / n, 2), 5: round(100 * R[5] / n, 2), 10: round(100 * R[10] / n, 2),
            "map10": round(100 * float(np.mean(aps)), 2)}


# ---------- data ----------
def coco_data():
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
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    paths = [f'{REPO}/data/coco/{dco[i]["filepath"]}/{dco[i]["filename"]}' for i in te_ids]
    gold = list(range(len(te_ids)))
    return dict(EC=EC, KO=KO, paths=paths, caps={"en": en_caps, "ko": ko_caps}, gold={"en": gold, "ko": gold},
                langs=["en", "ko"], n_img=len(te_ids))


def xm3600_data(data_dir):
    data = Path(data_dir)
    capf = data / "captions.jsonl"
    if not capf.exists():
        capf = next(data.rglob("captions.jsonl"))
    imgmap = {p.stem: p for p in (data / "images").rglob("*.jpg")} if (data / "images").exists() \
        else {p.stem: p for p in data.rglob("*.jpg")}
    recs = [json.loads(l) for l in open(capf, encoding="utf-8") if l.strip()]
    paths, key2idx = [], {}
    for rec in recs:
        p = imgmap.get(rec.get("image/key"))
        if p is not None:
            key2idx[rec["image/key"]] = len(paths); paths.append(p)
    caps, gold = {}, {}
    for rec in recs:
        idx = key2idx.get(rec.get("image/key"))
        if idx is None:
            continue
        for lang, val in rec.items():
            if not isinstance(val, dict):
                continue
            for cap in (val.get("caption") or []):
                caps.setdefault(lang, []).append(cap); gold.setdefault(lang, []).append(idx)
    return dict(paths=paths, caps=caps, gold=gold, langs=sorted(caps.keys()), n_img=len(paths))


# ---------- emit rows ----------
def add_rows(rows, model, space, dataset, ds, per_lang_metrics):
    """per_lang_metrics: lang -> {1,5,10,map10}. Emit REPR langs present + avg over all langs."""
    present = [L for L in ds["langs"] if L in per_lang_metrics]
    for L in present:
        if dataset == "coco" or L in REPR:
            m = per_lang_metrics[L]
            rows.append({"model": model, "space": space, "dataset": dataset, "lang": L,
                         "n_caps": len(ds["caps"][L]), "n_imgs": ds["n_img"],
                         "R1": m[1], "R5": m[5], "R10": m[10], "mAP10": m["map10"]})
    if dataset == "xm3600" and present:
        for tag, sub in (("avg36", present),):
            arr = {k: round(float(np.mean([per_lang_metrics[L][k] for L in sub])), 2) for k in (1, 5, 10, "map10")}
            rows.append({"model": model, "space": space, "dataset": dataset, "lang": tag,
                         "n_caps": "", "n_imgs": ds["n_img"], "R1": arr[1], "R5": arr[5],
                         "R10": arr[10], "mAP10": arr["map10"]})


# ---------- so400m image encode ----------
@torch.no_grad()
def so400m_imgemb(paths, dev, dtype, batch=64):
    from transformers import AutoModel
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    except Exception:
        from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
        proc = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(SIGLIP_MODEL),
                               tokenizer=GemmaTokenizer.from_pretrained(SIGLIP_MODEL))
    bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=dtype).to(dev).eval()
    tok = proc.tokenizer
    out, t0 = [], time.perf_counter()
    for s in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype)
        o = bb.vision_model(pixel_values=px)
        e = (o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1))
        out.append(e.float().cpu())
    print(f"  [so400m] {len(paths)} imgs in {time.perf_counter()-t0:.0f}s", flush=True)
    return torch.cat(out), bb, tok


@torch.no_grad()
def so400m_txtemb(bb, tok, strings, dev, dtype, batch=256):
    out = []
    for s in range(0, len(strings), batch):
        t = tok(strings[s:s + batch], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        o = bb.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)
        e = (o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1))
        out.append(e.float().cpu())
    return torch.cat(out)


# ---------- our rows (so400m float tower + ft113 server bit + MiniLM offline bit) ----------
def run_ours(rows, ds, dataset, enc, dev, dtype):
    # image emb: cached for COCO, fresh for XM3600
    if dataset == "coco":
        img_emb = ds["EC"]["test"]["img"].float()
        bb, tok = None, None
    else:
        img_emb, bb, tok = so400m_imgemb(ds["paths"], dev, dtype)
    img_n = F.normalize(img_emb, dim=1)
    gal = enc.image_codes_packed(img_emb.numpy())  # ft113 img_h -> packed codes (no_grad inside)
    gal_ix = faiss_bin(gal, CODE_BITS)
    # MiniLM offline head
    ck = torch.load(MINILM_HEAD, map_location="cpu")
    mm_th = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
    mm_th.load_state_dict(ck["txt_h"]); mm_th.to(dev).eval()
    mm_obi = [int(b) for b in ck["bits"]].index(CODE_BITS)
    from transformers import AutoModel, AutoTokenizer
    mm = AutoModel.from_pretrained(ck["student"]).to(dev).eval()
    mmtok = AutoTokenizer.from_pretrained(ck["student"])
    mm_prefix = ck.get("prefix") or ""

    @torch.no_grad()
    def minilm_emb(strings, batch=256):
        o = []
        for s in range(0, len(strings), batch):
            t = mmtok([mm_prefix + x for x in strings[s:s + batch]], padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt")
            h = mm(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            e = (h * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            o.append(F.normalize(e, dim=1).float().cpu())
        return torch.cat(o)

    M_float, M_server, M_off = {}, {}, {}
    for L in ds["langs"]:
        caps, gold = ds["caps"][L], ds["gold"][L]
        if dataset == "coco":
            # cached so400m text emb (EN) / KO emb
            te = ds["EC"]["test"]["txt"].float() if L == "en" else ds["KO"]["txt_emb"].float()
        else:
            te = so400m_txtemb(bb, tok, caps, dev, dtype)
        M_float[L] = rk_map_float(F.normalize(te, dim=1).to(dev), img_n.to(dev), gold)
        M_server[L] = rk_map_bin(gal_ix, pack_bits(enc._codes_pm1(enc.txt_h, te)), gold)
        with torch.no_grad():
            off_codes = mm_th(minilm_emb(caps).to(dev))[mm_obi]["binary"].detach().cpu().numpy()
        M_off[L] = rk_map_bin(gal_ix, pack_bits(off_codes), gold)
        if dataset == "coco" or L in REPR:
            print(f"  [ours {dataset} {L}] float R@10 {M_float[L][10]} | server {M_server[L][10]} | "
                  f"offline {M_off[L][10]}", flush=True)
    add_rows(rows, "SigLIP2 text tower (so400m float)", "float", dataset, ds, M_float)
    add_rows(rows, "Ours: server (so400m+ft113)", "1bit", dataset, ds, M_server)
    add_rows(rows, "Ours: offline (MiniLM head-adapt)", "1bit", dataset, ds, M_off)
    del mm, mm_th
    if bb is not None:
        del bb
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ---------- NLLB-CLIP (open_clip) ----------
def run_nllb(rows, ds, dataset, dev, variant="nllb-clip-base-siglip", pretrained="v1"):
    # NLLB-CLIP model loads via open_clip (no meta-device issue); its open_clip tokenizer wrapper
    # is broken under transformers 5.1 (AutoTokenizer -> None model_type), so we use the NLLB
    # tokenizer class directly (FLORES-200 src_lang per language).
    import open_clip
    from transformers import NllbTokenizerFast
    model, preprocess = open_clip.create_model_from_pretrained(variant, pretrained)
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    model = model.to(dev).eval()

    @torch.no_grad()
    def img_emb(paths, batch=64):
        out, t0 = [], time.perf_counter()
        for s in range(0, len(paths), batch):
            px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(dev)
            out.append(F.normalize(model.encode_image(px), dim=1).float().cpu())
        print(f"  [nllb] {len(paths)} imgs in {time.perf_counter()-t0:.0f}s", flush=True)
        return torch.cat(out)

    @torch.no_grad()
    def txt_emb(strings, lang, batch=256):
        tokenizer.src_lang = NLLB_LANG.get(lang, "eng_Latn")
        out = []
        for s in range(0, len(strings), batch):
            enc = tokenizer(strings[s:s + batch], return_tensors="pt", padding="max_length",
                            max_length=64, truncation=True)
            out.append(F.normalize(model.encode_text(enc["input_ids"].to(dev)), dim=1).float().cpu())
        return torch.cat(out)

    imn = img_emb(ds["paths"]).to(dev)
    M = {}
    for L in ds["langs"]:
        M[L] = rk_map_float(txt_emb(ds["caps"][L], L).to(dev), imn, ds["gold"][L])
        if dataset == "coco" or L in REPR:
            print(f"  [nllb {dataset} {L}] R@10 {M[L][10]} mAP10 {M[L]['map10']}", flush=True)
    add_rows(rows, f"NLLB-CLIP ({variant})", "float", dataset, ds, M)
    del model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ---------- AltCLIP-m18 (transformers; explicit projection path for tf5.1) ----------
def run_altclip(rows, ds, dataset, dev):
    from transformers import AltCLIPModel, AltCLIPProcessor
    m = AltCLIPModel.from_pretrained("BAAI/AltCLIP-m18").to(dev).eval()
    proc = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP-m18")

    @torch.no_grad()
    def img_emb(paths, batch=64):
        out, t0 = [], time.perf_counter()
        for s in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            e = m.visual_projection(m.vision_model(pixel_values=px).pooler_output)
            out.append(F.normalize(e, dim=1).float().cpu())
        print(f"  [altclip] {len(paths)} imgs in {time.perf_counter()-t0:.0f}s", flush=True)
        return torch.cat(out)

    @torch.no_grad()
    def txt_emb(strings, batch=128):
        out = []
        for s in range(0, len(strings), batch):
            t = proc(text=strings[s:s + batch], padding=True, truncation=True, max_length=77, return_tensors="pt")
            e = m.text_projection(m.text_model(input_ids=t["input_ids"].to(dev),
                                               attention_mask=t["attention_mask"].to(dev)).pooler_output)
            out.append(F.normalize(e, dim=1).float().cpu())
        return torch.cat(out)

    if dataset == "coco" and os.path.exists("/tmp/bb_altclip-m18_test.pt"):
        A = torch.load("/tmp/bb_altclip-m18_test.pt", map_location="cpu")
        imn = F.normalize(A["img"].float(), dim=1).to(dev)
        cached = {"en": A.get("en"), "ko": A.get("ko")}
    else:
        imn = img_emb(ds["paths"]).to(dev); cached = {}
    M = {}
    for L in ds["langs"]:
        if cached.get(L) is not None:
            te = F.normalize(cached[L].float(), dim=1)
        else:
            te = txt_emb(ds["caps"][L])
        M[L] = rk_map_float(te.to(dev), imn, ds["gold"][L])
        if dataset == "coco" or L in REPR:
            print(f"  [altclip {dataset} {L}] R@10 {M[L][10]} mAP10 {M[L]['map10']}", flush=True)
    add_rows(rows, "AltCLIP-m18", "float", dataset, ds, M)
    del m
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ---------- M-CLIP (multilingual_clip text + open_clip ViT-L-14 image) ----------
def run_mclip(rows, ds, dataset, dev):
    import open_clip
    from multilingual_clip import pt_multilingual_clip
    import transformers
    name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    tmodel = pt_multilingual_clip.MultilingualCLIP.from_pretrained(name).to(dev).eval()
    ttok = transformers.AutoTokenizer.from_pretrained(name)
    imodel, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    imodel = imodel.to(dev).eval()

    @torch.no_grad()
    def img_emb(paths, batch=64):
        out = []
        for s in range(0, len(paths), batch):
            px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(dev)
            out.append(F.normalize(imodel.encode_image(px), dim=1).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def txt_emb(strings, batch=128):
        out = []
        for s in range(0, len(strings), batch):
            out.append(F.normalize(tmodel.forward(strings[s:s + batch], ttok), dim=1).float().cpu())
        return torch.cat(out)

    imn = img_emb(ds["paths"]).to(dev)
    M = {}
    for L in ds["langs"]:
        M[L] = rk_map_float(txt_emb(ds["caps"][L]).to(dev), imn, ds["gold"][L])
        if dataset == "coco" or L in REPR:
            print(f"  [mclip {dataset} {L}] R@10 {M[L][10]}", flush=True)
    add_rows(rows, "M-CLIP (XLM-R-L/ViT-L-14)", "float", dataset, ds, M)
    del tmodel, imodel
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ---------- jina-clip-v2 (transformers, trust_remote_code) ----------
def run_jina(rows, ds, dataset, dev):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(dev).eval()

    @torch.no_grad()
    def img_emb(paths, batch=32):
        out = []
        for s in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
            out.append(F.normalize(torch.as_tensor(m.encode_image(imgs)), dim=1).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def txt_emb(strings, batch=128):
        out = []
        for s in range(0, len(strings), batch):
            out.append(F.normalize(torch.as_tensor(m.encode_text(strings[s:s + batch])), dim=1).float().cpu())
        return torch.cat(out)

    imn = img_emb(ds["paths"]).to(dev)
    M = {}
    for L in ds["langs"]:
        M[L] = rk_map_float(txt_emb(ds["caps"][L]).to(dev), imn, ds["gold"][L])
        if dataset == "coco" or L in REPR:
            print(f"  [jina {dataset} {L}] R@10 {M[L][10]}", flush=True)
    add_rows(rows, "jina-clip-v2", "float", dataset, ds, M)
    del m
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ours,nllb,altclip,mclip,jina")
    ap.add_argument("--xm3600", default="data/xm3600")
    ap.add_argument("--datasets", default="coco,xm3600")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dev == "cuda" and args.dtype or "fp32"]
    want = args.models.split(",")
    PAPER.mkdir(parents=True, exist_ok=True)

    datasets = {}
    if "coco" in args.datasets:
        datasets["coco"] = coco_data()
    if "xm3600" in args.datasets:
        datasets["xm3600"] = xm3600_data(args.xm3600)
    for name, ds in datasets.items():
        print(f"[a] dataset {name}: n_img={ds['n_img']} langs={len(ds['langs'])}", flush=True)

    enc = Encoder()  # ft113 img_h + server txt_h
    rows, flags = [], []
    runners = [("ours", run_ours, True), ("nllb", run_nllb, False), ("altclip", run_altclip, False),
               ("mclip", run_mclip, False), ("jina", run_jina, False)]
    for key, fn, needs_enc in runners:
        if key not in want:
            continue
        for name, ds in datasets.items():
            try:
                if needs_enc:
                    fn(rows, ds, name, enc, dev, dtype)
                else:
                    fn(rows, ds, name, dev)
            except Exception as e:
                msg = f"{key}/{name}: FAILED {repr(e)[:160]}"
                flags.append(msg)
                print(f"[a] FLAG {msg}", flush=True)

    cols = ["model", "space", "dataset", "lang", "n_caps", "n_imgs", "R1", "R5", "R10", "mAP10"]
    with open(PAPER / "baselines_multiling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[a] RESULT_JSON " + json.dumps({"rows": rows, "flags": flags}, ensure_ascii=False), flush=True)
    print(f"[a] DONE -> paper/baselines_multiling.csv ({len(rows)} rows); flags={len(flags)}", flush=True)
    for fl in flags:
        print(f"[a]   FLAG: {fl}", flush=True)


if __name__ == "__main__":
    main()
