"""Reviewer experiment batch — items #2 (variance/seeds), #3 (mAP), #5 (index build cost/memory).
EVAL/measurement only on existing so400m caches; 1024-bit, eval_korean 5K protocol. Does NOT touch
ft113 / index.bin / common.py. (Item #1 off-the-shelf baseline = web/paper_baseline_mclip.py.)

  --task seeds      MiniLM head-adapt x 3 seeds (0/1/2) -> EN/KO R@10 mean±std (training variance).
  --task map        mAP@10 (+R@10) for server-1bit / MiniLM head-adapt / e5-small deployed offline, EN/KO.
  --task buildcost  50K index build: so400m vision throughput + img_h+pack timing + index size + JS mem.

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_review_exp.py --task seeds
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

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, CODE_BYTES, MODEL as SIGLIP_MODEL, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
KS = (1, 5, 10)
_ID = re.compile(r"_0*(\d+)\.jpg")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-small"


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_ks(ix, q, gold, ks=KS):
    _, I = ix.search(q, max(ks))
    return {k: round(100 * np.mean([gold[i] in I[i, :k] for i in range(len(gold))]), 2) for k in ks}


def map_at10(ix, q, gold, k=10):
    """single-gold mAP@k = mean AP@k = mean(1/rank if gold in top-k else 0)."""
    _, I = ix.search(q, k)
    aps = []
    for i in range(len(gold)):
        hit = np.where(I[i, :k] == gold[i])[0]
        aps.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return round(100 * float(np.mean(aps)), 2)


def load_enc(name, dev):
    from transformers import AutoModel, AutoTokenizer
    return AutoModel.from_pretrained(name).to(dev).eval(), AutoTokenizer.from_pretrained(name)


@torch.no_grad()
def embed(m, tok, strings, prefix, dev, cache=None, batch=256, maxlen=64):
    if cache and os.path.exists(cache):
        return torch.load(cache)
    out = []
    for s in range(0, len(strings), batch):
        txt = [prefix + x for x in strings[s:s + batch]]
        t = tok(txt, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        o = m(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
        msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
        e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        out.append(F.normalize(e, dim=1).float().cpu())
    emb = torch.cat(out)
    if cache:
        torch.save(emb, cache)
    return emb


def train_head(enc_tr, img_tr, img_h, bits, hidden, P, dev, seed, epochs=25, bs=512):
    txt_h = NestedHashLayer(enc_tr.shape[1], hidden, bits, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(bits, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(txt_h.parameters(), lr=P["lr"], weight_decay=P["wd"])
    N = enc_tr.shape[0]; steps = epochs * (N // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    torch.manual_seed(seed); g = 0
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N - bs + 1, bs):
            idx = perm[s:s + bs]
            with torch.no_grad():
                io = img_h(img_tr[idx].to(dev))
            loss = lf(io, txt_h(enc_tr[idx].to(dev)), progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
    txt_h.eval(); return txt_h


def load_common(enc):
    """test gallery + so400m server text codes + EN/KO captions + ids (eval_korean 5K)."""
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()
    te_en_so = EC["test"]["txt"].float().numpy()
    ko_so = KO["txt_emb"].float().numpy()
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); mm = _ID.search(e["image_path"])
        if mm:
            kf[int(mm.group(1))] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    gold = list(range(len(te_ids)))
    gal = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    return dict(te_img=te_img, te_en_so=te_en_so, ko_so=ko_so, en_caps=en_caps, ko_caps=ko_caps,
                gold=gold, gal=gal, gal_ix=faiss_bin(gal, CODE_BITS))


def build_train_pairs():
    """reconstruct the head-adapt training pairs (EN+KO captions <-> matched so400m img), as in Ext①."""
    emb_aug = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in emb_aug["ids"].tolist()]; id2row = {c: i for i, c in enumerate(tr_ids)}
    img_tr = F.normalize(emb_aug["clean"].float(), dim=1)
    ko_rows = []
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line); mm = _ID.search(e["image_path"])
        if mm and int(mm.group(1)) in id2row and e.get("captions"):
            ko_rows.append(id2row[int(mm.group(1))])
    train_img = torch.cat([img_tr, img_tr[torch.tensor(ko_rows)]], 0)
    return train_img  # (226k, 1152); aligns row-for-row with the cached MiniLM train emb (EN then KO)


def codes(head, emb_np, dev, bi):
    with torch.no_grad():
        o = head(F.normalize(torch.from_numpy(emb_np).to(dev), dim=1))
    return pack_bits(o[bi]["binary"].detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seeds", "map", "buildcost"], required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(); bi = enc.bit_index; bits = enc.bit_list; hidden = enc.hidden
    P = json.load(open("/tmp/hp_results.json"))["best_params"]
    PAPER.mkdir(parents=True, exist_ok=True)

    if args.task == "seeds":
        C = load_common(enc)
        train_img = build_train_pairs()
        mm_tr = torch.load("/tmp/enc_paraphrase-multilingual-MiniLM-L12-v2_train.pt", map_location="cpu")
        assert mm_tr.shape[0] == train_img.shape[0], f"mismatch {mm_tr.shape} {train_img.shape}"
        m, tok = load_enc(MINILM, dev)
        mm_en = embed(m, tok, C["en_caps"], "", dev).numpy()
        mm_ko = embed(m, tok, C["ko_caps"], "", dev).numpy()
        rows = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            txt_h = train_head(mm_tr, train_img, enc.img_h, bits, hidden, P, dev, seed, args.epochs)
            r_en = recall_ks(C["gal_ix"], codes(txt_h, mm_en, dev, bi), C["gold"])
            r_ko = recall_ks(C["gal_ix"], codes(txt_h, mm_ko, dev, bi), C["gold"])
            rows.append({"seed": seed, "EN_R10": r_en[10], "KO_R10": r_ko[10]})
            print(f"[seeds] seed {seed}: EN R@10 {r_en[10]} | KO R@10 {r_ko[10]}", flush=True)
        en = np.array([r["EN_R10"] for r in rows]); ko = np.array([r["KO_R10"] for r in rows])
        summ = {"EN_mean": round(en.mean(), 2), "EN_std": round(en.std(ddof=0), 2),
                "KO_mean": round(ko.mean(), 2), "KO_std": round(ko.std(ddof=0), 2), "n": int(en.size)}
        with open(PAPER / "review_seeds.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "EN_R10", "KO_R10"]); w.writeheader(); w.writerows(rows)
        print(f"[seeds] MiniLM head-adapt R@10 EN {summ['EN_mean']}±{summ['EN_std']} | "
              f"KO {summ['KO_mean']}±{summ['KO_std']} (n={en.size})", flush=True)
        print("[seeds] RESULT_JSON " + json.dumps({"rows": rows, "summary": summ}), flush=True)

    elif args.task == "map":
        C = load_common(enc)
        out = []
        # server 1-bit (so400m text -> ft113 txt_h)
        for L, emb_np in (("EN", C["te_en_so"]), ("KO", C["ko_so"])):
            c = pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(emb_np)))
            out.append({"row": "server (so400m+ft113)", "lang": L,
                        "R10": recall_ks(C["gal_ix"], c, C["gold"])[10], "mAP10": map_at10(C["gal_ix"], c, C["gold"])})
        # offline heads: MiniLM head-adapt + e5-small deployed (both no-prefix)
        for label, model, head_path in (("MiniLM head-adapt", MINILM, "/tmp/txt_h_paraphrase-multilingual-MiniLM-L12-v2.pt"),
                                        ("e5-small deployed offline", E5, "/tmp/txt_h_e5.pt")):
            ck = torch.load(head_path, map_location="cpu")
            th = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
            th.load_state_dict(ck["txt_h"]); th.to(dev).eval()
            obi = [int(b) for b in ck["bits"]].index(CODE_BITS)
            m, tok = load_enc(model, dev)
            for L, caps in (("EN", C["en_caps"]), ("KO", C["ko_caps"])):
                c = codes(th, embed(m, tok, caps, "", dev).numpy(), dev, obi)
                out.append({"row": label, "lang": L, "R10": recall_ks(C["gal_ix"], c, C["gold"])[10],
                            "mAP10": map_at10(C["gal_ix"], c, C["gold"])})
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        with open(PAPER / "review_map.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["row", "lang", "R10", "mAP10"]); w.writeheader(); w.writerows(out)
        for r in out:
            print(f"[map] {r['row']} {r['lang']}: R@10 {r['R10']} | mAP@10 {r['mAP10']}", flush=True)
        print("[map] RESULT_JSON " + json.dumps(out), flush=True)

    elif args.task == "buildcost":
        # (a) so400m vision throughput on a 300-image sample
        from PIL import Image
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        except Exception:
            from transformers import GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
            proc = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(SIGLIP_MODEL),
                                   tokenizer=GemmaTokenizer.from_pretrained(SIGLIP_MODEL))
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=torch.bfloat16).to(dev).eval()
        paths = sorted(__import__("glob").glob(f"{REPO}/data/coco/val2014/*.jpg"))[:300]
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(paths), 64):
                imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 64]]
                pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, torch.bfloat16)
                _ = bb.vision_model(pixel_values=pv).pooler_output
        vis_ips = len(paths) / (time.perf_counter() - t0)
        del bb
        if dev == "cuda":
            torch.cuda.empty_cache()
        # (b) img_h + pack on 50K cached embeddings
        ix = np.load("/tmp/demo_index.npz", allow_pickle=True)
        emb50 = ix["emb"].astype(np.float32)[:50000]
        t1 = time.perf_counter()
        packed = enc.image_codes_packed(emb50)
        head_pack_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        (Path("/tmp/_idx_bench.bin")).write_bytes(packed.tobytes())
        serialize_s = time.perf_counter() - t2
        n = len(emb50)
        idx_bytes = n * CODE_BYTES
        # JS footprint: index + query code + popcount LUT(256*2) + dist Int32(n*4) + xor scratch(CODE_BYTES)
        js_bytes = idx_bytes + CODE_BYTES + 256 * 2 + n * 4 + CODE_BYTES
        res = {"n": n, "so400m_vision_img_per_s_bf16": round(vis_ips, 1),
               "est_vision_50k_min": round(n / vis_ips / 60, 1),
               "img_h+pack_50k_s": round(head_pack_s, 2), "img_h+pack_img_per_s": round(n / head_pack_s, 0),
               "index_serialize_s": round(serialize_s, 3),
               "index_MB": round(idx_bytes / 1e6, 2), "bytes_per_img": CODE_BYTES,
               "js_search_footprint_MB": round(js_bytes / 1e6, 2)}
        os.remove("/tmp/_idx_bench.bin")
        with open(PAPER / "review_buildcost.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
        print("[buildcost] " + json.dumps(res), flush=True)
        print("[buildcost] RESULT_JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
