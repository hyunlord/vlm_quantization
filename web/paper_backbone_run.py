"""Extension ③ — backbone generalization. For ONE cross-modal backbone B, reproduce the core
pipeline to test whether contributions ②/③/④ hold beyond SigLIP2-So400m. EVAL+TRAIN(head only,
no backbone finetune), 1024-bit, eval_korean 5K protocol. Does NOT touch COCO ft113 / index.bin /
common.py — all B artifacts are separate (/tmp/bb_<tag>_*).

Stages (per backbone B):
  1. Embed COCO Karpathy (train+restval clean image+EN-text; test image+EN-text; KO test text) via B
     (B's image/text towers + MAP-head pooler_output). Cached to /tmp/bb_<tag>_{train,test}.pt.
  2. Train the FULL hash head (img_h + txt_h jointly) on B's clean train embeddings — repo recipe
     (train_1024: NestedHashLayer(dim->hidden->1024) x2, CombinedHashLoss, AdamW(both), OneCycleLR,
     25ep BS512, hp_results params, norm_in=1). NOTE: clean-only (weak/strong aug transforms are not
     in git); COCO-EN only (no OI/RKD/KO-finetune) — a documented simplification vs So400m=ft113.
  3. server 1-bit text->image R@{1,5,10} (EN/KO) + head-continuous ceiling (cosine on pre-sign
     continuous[1024]) = B's binarization cost. B's own anchor (absolute != So400m).
  4. head-adapt MiniLM (Ext① recipe: txt_h' on MiniLM emb aligned to B's FROZEN img_h codes) -> offline R@K.
  5. precision: B head/emb at fp32/bf16/int8 -> bitflip/1024 + R@10 (is the head 0-near sensitive: int8 head>emb?).

Writes one row to paper/backbones.csv (+ So400m reference row passed via repo data).
Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_backbone_run.py \
      --model google/siglip2-base-patch16-256 --tag siglip2-base --epochs 25
"""
from __future__ import annotations

import argparse
import copy
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
from web.common import CODE_BITS, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
BITS = [8, 16, 32, 64, 128, 256, 512, 1024]
_ID = re.compile(r"_0*(\d+)\.jpg")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------- index + recall (mirror web/eval_paper.py) ----------
def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_ks(ix, q, gold_rows, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold_rows[i] in I[i, :k] for i in range(len(gold_rows))]), 2) for k in ks}


def cosine_recall(q_emb, db_emb, gold_rows, ks=KS):
    """Float cosine top-K (the binarization ceiling when q/db are the head's pre-sign continuous)."""
    q = F.normalize(torch.as_tensor(q_emb).float(), dim=1)
    db = F.normalize(torch.as_tensor(db_emb).float(), dim=1)
    topk = (q @ db.t()).topk(max(ks), dim=1).indices.numpy()
    return {k: round(100 * np.mean([gold_rows[i] in topk[i, :k] for i in range(len(gold_rows))]), 2) for k in ks}


def _pool(o):
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


# ---------- SigLIP-family backbone (image + text via MAP-head pooler_output) ----------
def load_siglip(name, dev, dtype):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, dtype=dtype).to(dev).eval()
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(name)
    except Exception:
        from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
        proc = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(name),
                               tokenizer=GemmaTokenizer.from_pretrained(name))
    return m, proc


@torch.no_grad()
def emb_images(m, proc, paths, dev, dtype, batch=64):
    out, t0, M = [], time.perf_counter(), len(paths)
    for s in range(0, M, batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype=dtype)
        out.append(_pool(m.vision_model(pixel_values=px)).float().cpu())
        if (s // batch) % 25 == 0:
            print(f"  img {min(s + batch, M)}/{M} ({(s + len(imgs)) / (time.perf_counter() - t0 + 1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


@torch.no_grad()
def emb_texts_siglip(m, proc, strings, dev, dtype, batch=256, maxlen=64):
    tok, out = proc.tokenizer, []
    for s in range(0, len(strings), batch):
        t = tok(strings[s:s + batch], padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        am = t.get("attention_mask")
        o = m.text_model(input_ids=t["input_ids"].to(dev), attention_mask=am.to(dev) if am is not None else None)
        out.append(_pool(o).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def emb_texts_xlmr(name, strings, dev, batch=256, maxlen=64):
    """mean-pool + L2 (e5/MiniLM offline encoder convention, no prefix)."""
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(name).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(name)
    out = []
    for s in range(0, len(strings), batch):
        t = tok(strings[s:s + batch], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        o = m(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
        msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
        e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=1).float().cpu())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out)


# ---------- data ----------
def load_coco():
    data = json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]
    tr, te = [], []
    for im in data:
        if not im.get("sentences"):
            continue
        rec = (im["cocoid"], f'{im.get("filepath","")}/{im["filename"]}'.lstrip("/"), im["sentences"][0]["raw"])
        if im["split"] in ("train", "restval"):
            tr.append(rec)
        elif im["split"] == "test":
            te.append(rec)
    return tr, te


def load_ko_test(test_ids):
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); mm = _ID.search(e["image_path"])
        if mm:
            kf[int(mm.group(1))] = e.get("captions", [])
    return [(kf.get(i, [""]) or [""])[0] for i in test_ids]


# ---------- full head training (train_1024 recipe; clean-only) ----------
def train_full_head(tr_img, tr_txt, dim, P, dev, epochs, bs=512):
    img_h = NestedHashLayer(dim, P["hidden"], BITS, P["dropout"]).to(dev).train()
    txt_h = NestedHashLayer(dim, P["hidden"], BITS, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(list(img_h.parameters()) + list(txt_h.parameters()), lr=P["lr"], weight_decay=P["wd"])
    N = tr_img.shape[0]; steps = epochs * (N // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    torch.manual_seed(42); g = 0; t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N - bs + 1, bs):
            idx = perm[s:s + bs]
            io = img_h(tr_img[idx].to(dev)); to = txt_h(tr_txt[idx].to(dev))
            loss = lf(io, to, progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [head] ep {ep+1}/{epochs} loss {loss.item():.4f} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)
    img_h.eval(); txt_h.eval(); return img_h, txt_h


def train_head_adapt(tr_emb, tr_img_codes_src, img_h, dim, P, dev, epochs, bs=512):
    """Ext① head-adapt: txt_h' on tr_emb aligned to FROZEN img_h codes (image branch detached)."""
    txt_h = NestedHashLayer(dim, P["hidden"], BITS, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(txt_h.parameters(), lr=P["lr"], weight_decay=P["wd"])
    N = tr_emb.shape[0]; steps = epochs * (N // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    torch.manual_seed(42); g = 0
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N - bs + 1, bs):
            idx = perm[s:s + bs]
            with torch.no_grad():
                io = img_h(tr_img_codes_src[idx].to(dev))
            to = txt_h(tr_emb[idx].to(dev))
            loss = lf(io, to, progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
    txt_h.eval(); return txt_h


# ---------- code helpers (norm_in=1 contract) ----------
def head_binary(head, emb, dev, bi):
    with torch.no_grad():
        o = head(F.normalize(emb.to(dev), dim=1))
    return o[bi]["binary"].cpu().numpy()


def head_continuous(head, emb, dev, bi):
    with torch.no_grad():
        o = head(F.normalize(emb.to(dev), dim=1))
    return o[bi]["continuous"].cpu().numpy()


# ---------- precision (mirror web/eval_paper.py "C": dynamic-quant head, storage-cast emb;
#            bitflips/1024 measured on PACKED TEXT codes, EN+KO averaged) ----------
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pair_bitflip(a, b):
    return float(_LUT[np.bitwise_xor(a, b)].sum(1).mean())


def cast_emb_np(x, dt):
    """storage-cast a raw (N,dim) float32 embedding to dt then back to float32 (eval_paper.py cast_emb)."""
    if dt == "int8":
        s = np.abs(x).max(axis=1, keepdims=True) / 127.0 + 1e-9
        return (np.round(x / s).clip(-127, 127) * s).astype(np.float32)
    if dt == "fp16":
        return x.astype(np.float16).astype(np.float32)
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", default="SigLIP2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--adapt-epochs", type=int, default=25)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--limit-train", type=int, default=0, help="smoke: cap train pairs")
    ap.add_argument("--limit-test", type=int, default=0, help="smoke: cap test images")
    ap.add_argument("--out", default=str(PAPER / "backbones.csv"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype if dev == "cuda" else "fp32"]
    bi = BITS.index(CODE_BITS)
    P = json.load(open("/tmp/hp_results.json"))["best_params"]
    PAPER.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    tr_cache = f"/tmp/bb_{tag}_train.pt"; te_cache = f"/tmp/bb_{tag}_test.pt"

    # ---- Stage 1: embeddings (cached) ----
    tr, te = load_coco()
    if args.limit_train:
        tr = tr[:args.limit_train]
    if args.limit_test:
        te = te[:args.limit_test]
    te_ids = [r[0] for r in te]
    ko_caps = load_ko_test(te_ids)
    print(f"[bb:{tag}] train {len(tr)} / test {len(te)} | model {args.model} dim?", flush=True)

    if os.path.exists(tr_cache) and os.path.exists(te_cache):
        TR = torch.load(tr_cache, map_location="cpu"); TE = torch.load(te_cache, map_location="cpu")
        print(f"[bb:{tag}] embeddings from cache", flush=True)
    else:
        m, proc = load_siglip(args.model, dev, dtype)
        t0 = time.perf_counter()
        tr_img = emb_images(m, proc, [f"{REPO}/data/coco/{r[1]}" for r in tr], dev, dtype)
        tr_txt = emb_texts_siglip(m, proc, [r[2] for r in tr], dev, dtype)
        te_img = emb_images(m, proc, [f"{REPO}/data/coco/{r[1]}" for r in te], dev, dtype)
        te_txt = emb_texts_siglip(m, proc, [r[2] for r in te], dev, dtype)
        ko_txt = emb_texts_siglip(m, proc, ko_caps, dev, dtype)
        TR = {"img": tr_img, "txt": tr_txt}
        TE = {"img": te_img, "en": te_txt, "ko": ko_txt, "ids": te_ids}
        torch.save(TR, tr_cache); torch.save(TE, te_cache)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"[bb:{tag}] embedded in {(time.perf_counter()-t0)/60:.1f}min -> cached", flush=True)
    dim = TR["img"].shape[1]
    print(f"[bb:{tag}] dim={dim}", flush=True)

    # ---- Stage 2: full head ----
    head_cache = f"/tmp/bb_{tag}_head.pt"
    if os.path.exists(head_cache):
        ck = torch.load(head_cache, map_location="cpu")
        img_h = NestedHashLayer(dim, ck["hidden"], BITS, 0.0); img_h.load_state_dict(ck["img_h"]); img_h.to(dev).eval()
        txt_h = NestedHashLayer(dim, ck["hidden"], BITS, 0.0); txt_h.load_state_dict(ck["txt_h"]); txt_h.to(dev).eval()
        print(f"[bb:{tag}] head from cache", flush=True)
    else:
        img_h, txt_h = train_full_head(F.normalize(TR["img"].float(), dim=1), F.normalize(TR["txt"].float(), dim=1),
                                       dim, P, dev, args.epochs)
        torch.save({"img_h": img_h.state_dict(), "txt_h": txt_h.state_dict(), "bits": BITS,
                    "hidden": P["hidden"], "embed": dim, "norm_in": 1, "mode": f"bb_{tag}"}, head_cache)

    # ---- Stage 3: server 1-bit + continuous ceiling ----
    n = len(te_ids); gold = list(range(n))
    gal = pack_bits(head_binary(img_h, TE["img"], dev, bi))
    gal_ix = faiss_bin(gal, CODE_BITS)
    img_cont = head_continuous(img_h, TE["img"], dev, bi)
    R1bit, Rceil = {}, {}
    for L, key in (("EN", "en"), ("KO", "ko")):
        codes = pack_bits(head_binary(txt_h, TE[key], dev, bi))
        R1bit[L] = recall_ks(gal_ix, codes, gold)
        Rceil[L] = cosine_recall(head_continuous(txt_h, TE[key], dev, bi), img_cont, gold)
        print(f"[bb:{tag}] {L}: 1bit R@10 {R1bit[L][10]} | head-cont ceiling R@10 {Rceil[L][10]}", flush=True)

    # ---- Stage 4: head-adapt MiniLM to B's frozen img_h codes ----
    mm_tr = emb_texts_xlmr(MINILM, [r[2] for r in tr], dev)
    mm_dim = mm_tr.shape[1]
    txt_h_mm = train_head_adapt(F.normalize(mm_tr, dim=1), F.normalize(TR["img"].float(), dim=1),
                                img_h, mm_dim, P, dev, args.adapt_epochs)
    Rmm = {}
    for L, key in (("EN", "en"), ("KO", "ko")):
        caps = [r[2] for r in te] if key == "en" else ko_caps
        mm_te = emb_texts_xlmr(MINILM, caps, dev)
        codes = pack_bits(head_binary(txt_h_mm, mm_te, dev, bi))
        Rmm[L] = recall_ks(gal_ix, codes, gold)
        print(f"[bb:{tag}] MiniLM head-adapt {L}: R@10 {Rmm[L][10]}", flush=True)

    # ---- Stage 5: precision (mirror eval_paper.py C) — bitflips/1024 on TEXT codes (EN+KO avg) ----
    ref_p = {"en": pack_bits(head_binary(txt_h, TE["en"], dev, bi)), "ko": pack_bits(head_binary(txt_h, TE["ko"], dev, bi))}

    def txt_flip(codes_of):
        return round(float(np.mean([pair_bitflip(codes_of("en"), ref_p["en"]), pair_bitflip(codes_of("ko"), ref_p["ko"])])), 2)

    flips = {}
    th_bf = copy.deepcopy(txt_h).to(dev, torch.bfloat16).eval()
    flips["head_bf16"] = txt_flip(lambda key: pack_bits(
        th_bf(F.normalize(TE[key].float(), dim=1).to(dev, torch.bfloat16))[bi]["binary"].detach().float().cpu().numpy()))
    import torch.ao.quantization as _aoq
    th_q = _aoq.quantize_dynamic(copy.deepcopy(txt_h).cpu(), {torch.nn.Linear}, dtype=torch.qint8).eval()
    flips["head_int8"] = txt_flip(lambda key: pack_bits(
        th_q(F.normalize(TE[key].float(), dim=1).cpu())[bi]["binary"].detach().cpu().numpy()))
    flips["emb_bf16"] = txt_flip(lambda key: pack_bits(
        head_binary(txt_h, torch.from_numpy(cast_emb_np(TE[key].float().numpy(), "bf16")), dev, bi)))
    flips["emb_int8"] = txt_flip(lambda key: pack_bits(
        head_binary(txt_h, torch.from_numpy(cast_emb_np(TE[key].float().numpy(), "int8")), dev, bi)))
    print(f"[bb:{tag}] flips/1024 (text EN+KO): head_bf16 {flips['head_bf16']} head_int8 {flips['head_int8']} | "
          f"emb_bf16 {flips['emb_bf16']} emb_int8 {flips['emb_int8']}", flush=True)

    # ---- write/append row ----
    cols = ["backbone", "family", "dim", "server_1bit_EN_R10", "server_1bit_KO_R10",
            "head_cont_ceiling_EN_R10", "head_cont_ceiling_KO_R10",
            "MiniLM_headadapt_EN_R10", "MiniLM_headadapt_KO_R10",
            "head_bf16_flip", "head_int8_flip", "emb_bf16_flip", "emb_int8_flip",
            "server_1bit_EN_R1", "server_1bit_KO_R1"]
    row = {"backbone": args.model, "family": args.family, "dim": dim,
           "server_1bit_EN_R10": R1bit["EN"][10], "server_1bit_KO_R10": R1bit["KO"][10],
           "head_cont_ceiling_EN_R10": Rceil["EN"][10], "head_cont_ceiling_KO_R10": Rceil["KO"][10],
           "MiniLM_headadapt_EN_R10": Rmm["EN"][10], "MiniLM_headadapt_KO_R10": Rmm["KO"][10],
           "head_bf16_flip": flips["head_bf16"], "head_int8_flip": flips["head_int8"],
           "emb_bf16_flip": flips["emb_bf16"], "emb_int8_flip": flips["emb_int8"],
           "server_1bit_EN_R1": R1bit["EN"][1], "server_1bit_KO_R1": R1bit["KO"][1]}
    rows = []
    if os.path.exists(args.out):
        rows = [r for r in csv.DictReader(open(args.out)) if r["backbone"] != args.model]
    rows.append({k: row.get(k, "") for k in cols})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("[bb] RESULT_JSON " + json.dumps(row, ensure_ascii=False), flush=True)
    print(f"[bb:{tag}] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
