"""Extension ① — generalize the C1 head-adapt recipe across small multilingual text encoders.

For each encoder E: train a NEW head txt_h'_E = NestedHashLayer(E_dim -> hidden -> 1024) on E's
NATIVE (frozen) embeddings, aligned to the FROZEN ft113 img_h codes via the repo CombinedHashLoss
— SAME recipe / hp (hp_results) / split (train+restval) / gallery (5K img_h codes) as C1; the ONLY
variable is the encoder. Eval on the eval_korean 5K protocol. Writes paper/encoders.csv.

Fair comparison: each encoder uses mean-pooling + its documented prefix (e5 family: "query: ";
sentence-transformers MiniLM: none). NOTE the C1 reference row (e5-small) used raw text (no prefix)
— reported as-is. img_h / index.bin / common.py / C1 outputs are untouched (read-only).

Rows: server (so400m+ft113) | e5-small (C1, existing head) | MiniLM-L12-v2 | e5-base.
Run on DGX:  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_encoder_run.py
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

from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, CODE_BYTES, Encoder, pack_bits  # noqa: E402
from web.eval_paper import faiss_bin, recall_ks  # noqa: E402

PAPER = Path(REPO) / "paper"
_ID = re.compile(r"_0*(\d+)\.jpg")
KS = (1, 5, 10)
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = 512

ENCODERS = [
    {"label": "server (so400m+ft113)", "server": True, "family": "SigLIP2-so400m", "params": 707.8, "deploy": "n"},
    {"label": "e5-small (C1)", "model": "intfloat/multilingual-e5-small", "prefix": "",
     "head": "/tmp/txt_h_e5.pt", "family": "e5/XLM-R", "deploy": "y"},
    {"label": "MiniLM-L12-v2", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "prefix": "", "head": None, "family": "paraphrase/XLM-R", "deploy": "y"},
    {"label": "e5-base", "model": "intfloat/multilingual-e5-base", "prefix": "query: ",
     "head": None, "family": "e5/XLM-R", "deploy": "y"},
]


def load_enc(name, dev):
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(name).to(dev).eval()
    return m, AutoTokenizer.from_pretrained(name), m.config.hidden_size, sum(p.numel() for p in m.parameters())


def embed(m, tok, strings, prefix, dev, cache=None, batch=256, maxlen=64):
    if cache and os.path.exists(cache):
        return torch.load(cache)
    out = []
    with torch.no_grad():
        for s in range(0, len(strings), batch):
            txt = [prefix + x for x in strings[s:s + batch]]
            t = tok(txt, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
            o = m(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(F.normalize(e, dim=1).float().cpu())
    emb = torch.cat(out, 0)
    if cache:
        torch.save(emb, cache)
    return emb


def train_head(enc_tr, img_tr, img_h, bits, hidden, P, dev):
    txt_h = NestedHashLayer(enc_tr.shape[1], hidden, bits, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(bits, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(txt_h.parameters(), lr=P["lr"], weight_decay=P["wd"])
    N = enc_tr.shape[0]; steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    torch.manual_seed(42); g = 0; t0 = time.perf_counter()
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = perm[s:s + BS]
            with torch.no_grad():
                io = img_h(img_tr[idx].to(dev))
            to = txt_h(enc_tr[idx].to(dev))
            loss = lf(io, to, progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
    print(f"      trained head {EPOCHS}ep in {(time.perf_counter()-t0)/60:.1f}min", flush=True)
    txt_h.eval(); return txt_h


def main():
    enc = Encoder()
    dev = enc.device
    bits = [int(b) for b in enc.bit_list]
    bi = bits.index(CODE_BITS)
    hidden = enc.hidden
    P = json.load(open("/tmp/hp_results.json"))["best_params"]
    PAPER.mkdir(parents=True, exist_ok=True)

    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    emb_aug = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in emb_aug["ids"].tolist()]
    id2row = {c: i for i, c in enumerate(tr_ids)}
    img_tr = F.normalize(emb_aug["clean"].float(), dim=1)  # so400m image (shared anchor input)
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    en_tr = [(dco[c]["sentences"][0]["raw"] if c in dco and dco[c].get("sentences") else "") for c in tr_ids]
    ko_tr, ko_rows = [], []
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        if m and int(m.group(1)) in id2row and e.get("captions"):
            ko_tr.append(e["captions"][0]); ko_rows.append(id2row[int(m.group(1))])
    train_caps = en_tr + ko_tr
    train_img = torch.cat([img_tr, img_tr[torch.tensor(ko_rows)]], 0)  # EN rows + KO matched rows

    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()
    te_en = EC["test"]["txt"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        m = _ID.search(json.loads(line)["image_path"]); kf[int(m.group(1)) if m else -1] = json.loads(line).get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    n = len(te_ids); gold = list(range(n))
    test_caps = {"EN": en_caps, "KO": ko_caps}

    # frozen gallery (5K) + server codes + index.bin (50K, for overlap, same as C1)
    gal = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    gal_ix = faiss_bin(gal, CODE_BITS)
    srv = {L: pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(t))) for L, t in (("EN", te_en), ("KO", ko_emb))}
    packed50 = np.fromfile(Path(REPO) / "web/static/data/index.bin", dtype=np.uint8).reshape(-1, CODE_BYTES)
    ix50 = faiss_bin(np.ascontiguousarray(packed50), CODE_BITS)
    rng = np.random.default_rng(42); osel = np.sort(rng.choice(n, 1000, replace=False))
    srv_top = {L: ix50.search(srv[L][osel], 10)[1] for L in ("EN", "KO")}
    print(f"[enc] anchor server R@10 EN {recall_ks(gal_ix, srv['EN'], gold)[10]} / "
          f"KO {recall_ks(gal_ix, srv['KO'], gold)[10]}", flush=True)

    def codes_from_head(txt_h, enc_test_emb):
        with torch.no_grad():
            o = txt_h(enc_test_emb.to(dev))
        return pack_bits(o[bi]["binary"].cpu().numpy())

    rows = []
    for spec in ENCODERS:
        lab = spec["label"]
        print(f"\n[enc] === {lab} ===", flush=True)
        if spec.get("server"):
            R = {L: recall_ks(gal_ix, srv[L], gold) for L in ("EN", "KO")}
            row = dict(encoder=lab, family=spec["family"], dim=1152, params=spec["params"],
                       int8_MB=708, browser_deployable=spec["deploy"], overlap_EN=1.0, overlap_KO=1.0)
        else:
            m, tok, dim, npar = load_enc(spec["model"], dev)
            tag = spec["model"].split("/")[-1]
            test_emb = {L: embed(m, tok, test_caps[L], spec["prefix"], dev) for L in ("EN", "KO")}
            if spec["head"] and os.path.exists(spec["head"]):
                ck = torch.load(spec["head"], map_location="cpu")
                txt_h = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
                txt_h.load_state_dict(ck["txt_h"]); txt_h.to(dev).eval()
                print(f"      loaded existing head {spec['head']}", flush=True)
            else:
                tr_emb = embed(m, tok, train_caps, spec["prefix"], dev, cache=f"/tmp/enc_{tag}_train.pt")
                print(f"      train emb {tuple(tr_emb.shape)} | training head…", flush=True)
                txt_h = train_head(tr_emb, train_img, enc.img_h, bits, hidden, P, dev)
                torch.save({"txt_h": txt_h.state_dict(), "bits": bits, "hidden": hidden, "embed": dim,
                            "student": spec["model"], "prefix": spec["prefix"]}, f"/tmp/txt_h_{tag}.pt")
            R, ov = {}, {}
            for L in ("EN", "KO"):
                c = codes_from_head(txt_h, test_emb[L])
                R[L] = recall_ks(gal_ix, c, gold)
                top = ix50.search(c[osel], 10)[1]
                ov[L] = round(float(np.mean([len(set(top[i]) & set(srv_top[L][i])) / 10 for i in range(len(osel))])), 3)
            row = dict(encoder=lab, family=spec["family"], dim=dim, params=round(npar / 1e6, 1),
                       int8_MB=round(npar / 1e6), browser_deployable=spec["deploy"],
                       overlap_EN=ov["EN"], overlap_KO=ov["KO"])
        for L in ("EN", "KO"):
            for k in KS:
                row[f"{L}_R{k}"] = R[L][k]
        rows.append(row)
        print(f"[enc] {lab}: EN R@10 {row['EN_R10']} / KO {row['KO_R10']} | "
              f"overlap {row['overlap_EN']}/{row['overlap_KO']}", flush=True)
        cols = ["encoder", "family", "dim", "params", "int8_MB", "browser_deployable",
                "EN_R1", "EN_R5", "EN_R10", "KO_R1", "KO_R5", "KO_R10", "overlap_EN", "overlap_KO"]
        with open(PAPER / "encoders.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    print("[enc] RESULT_JSON " + json.dumps(rows), flush=True)
    print("[enc] DONE -> paper/encoders.csv", flush=True)


if __name__ == "__main__":
    main()
