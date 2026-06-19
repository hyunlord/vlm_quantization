"""Cleanup — re-measure e5-small WITH the e5-recommended "query: " prefix (head-only), to
de-confound the encoder sweep (the C1 e5-small row used raw text / no prefix). ONLY variable
vs C1 = the prefix; same recipe/hp/split/frozen-img_h gallery as paper_encoder_run.py.

Trains txt_h'_e5p = NestedHashLayer(384->hidden->1024) on e5-small embeddings of "query: "+caption,
aligned to frozen ft113 img_h codes; evals on eval_korean 5K (eval caps also "query: "-prefixed).
APPENDS an "e5-small +query:" row to paper/encoders.csv (existing rows preserved). img_h / index.bin /
common.py / existing heads untouched.

Run on DGX:  HEAD_PATH=/tmp/ft_ko_113.pt EPOCHS=25 .venv/bin/python web/paper_encoder_e5prefix.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, CODE_BYTES, Encoder, pack_bits  # noqa: E402
from web.eval_paper import faiss_bin, recall_ks  # noqa: E402
from web.paper_encoder_run import embed, load_enc, train_head  # noqa: E402

PAPER = Path(REPO) / "paper"
_ID = re.compile(r"_0*(\d+)\.jpg")
MODEL, PREFIX, LABEL = "intfloat/multilingual-e5-small", "query: ", "e5-small +query:"
KS = (1, 5, 10)


def main():
    enc = Encoder(); dev = enc.device
    bits = [int(b) for b in enc.bit_list]; bi = bits.index(CODE_BITS); hidden = enc.hidden
    P = json.load(open("/tmp/hp_results.json"))["best_params"]

    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    emb_aug = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in emb_aug["ids"].tolist()]; id2row = {c: i for i, c in enumerate(tr_ids)}
    img_tr = F.normalize(emb_aug["clean"].float(), dim=1)
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    en_tr = [(dco[c]["sentences"][0]["raw"] if c in dco and dco[c].get("sentences") else "") for c in tr_ids]
    ko_tr, ko_rows = [], []
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"])
        if m and int(m.group(1)) in id2row and e.get("captions"):
            ko_tr.append(e["captions"][0]); ko_rows.append(id2row[int(m.group(1))])
    train_caps = en_tr + ko_tr
    train_img = torch.cat([img_tr, img_tr[torch.tensor(ko_rows)]], 0)

    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy(); te_en = EC["test"]["txt"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); m = _ID.search(e["image_path"]); kf[int(m.group(1)) if m else -1] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]
    n = len(te_ids); gold = list(range(n)); test_caps = {"EN": en_caps, "KO": ko_caps}

    gal = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    gal_ix = faiss_bin(gal, CODE_BITS)
    srv = {L: pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(t))) for L, t in (("EN", te_en), ("KO", ko_emb))}
    print(f"[e5p] anchor server R@10 EN {recall_ks(gal_ix, srv['EN'], gold)[10]} / "
          f"KO {recall_ks(gal_ix, srv['KO'], gold)[10]} (expect 79.92/71.08)", flush=True)
    packed50 = np.fromfile(Path(REPO) / "web/static/data/index.bin", dtype=np.uint8).reshape(-1, CODE_BYTES)
    ix50 = faiss_bin(np.ascontiguousarray(packed50), CODE_BITS)
    rng = np.random.default_rng(42); osel = np.sort(rng.choice(n, 1000, replace=False))
    srv_top = {L: ix50.search(srv[L][osel], 10)[1] for L in ("EN", "KO")}

    m, tok, dim, npar = load_enc(MODEL, dev)
    tr_emb = embed(m, tok, train_caps, PREFIX, dev, cache="/tmp/enc_e5small_qprefix_train.pt")
    print(f"[e5p] train emb {tuple(tr_emb.shape)} (prefix={PREFIX!r}) | training head…", flush=True)
    txt_h = train_head(tr_emb, train_img, enc.img_h, bits, hidden, P, dev)
    torch.save({"txt_h": txt_h.state_dict(), "bits": bits, "hidden": hidden, "embed": dim,
                "student": MODEL, "prefix": PREFIX}, "/tmp/txt_h_e5p.pt")

    R, ov = {}, {}
    for L in ("EN", "KO"):
        emb_t = embed(m, tok, test_caps[L], PREFIX, dev)
        with torch.no_grad():
            c = pack_bits(txt_h(emb_t.to(dev))[bi]["binary"].cpu().numpy())
        R[L] = recall_ks(gal_ix, c, gold)
        top = ix50.search(c[osel], 10)[1]
        ov[L] = round(float(np.mean([len(set(top[i]) & set(srv_top[L][i])) / 10 for i in range(len(osel))])), 3)
    print(f"[e5p] e5-small +query: EN R@10 {R['EN'][10]} / KO {R['KO'][10]} | overlap {ov['EN']}/{ov['KO']}", flush=True)

    # append/replace row in encoders.csv
    csvp = PAPER / "encoders.csv"
    rows = list(csv.DictReader(open(csvp))); cols = list(rows[0].keys())
    rows = [r for r in rows if r["encoder"] != LABEL]  # idempotent
    new = {"encoder": LABEL, "family": "e5/XLM-R", "dim": dim, "params": round(npar / 1e6, 1),
           "int8_MB": round(npar / 1e6), "browser_deployable": "y",
           "overlap_EN": ov["EN"], "overlap_KO": ov["KO"]}
    for L in ("EN", "KO"):
        for k in KS:
            new[f"{L}_R{k}"] = R[L][k]
    # insert right after the e5-small (C1) row for readability
    out, inserted = [], False
    for r in rows:
        out.append(r)
        if r["encoder"] == "e5-small (C1)" and not inserted:
            out.append(new); inserted = True
    if not inserted:
        out.append(new)
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(out)
    print("[e5p] RESULT_JSON " + json.dumps(new), flush=True)
    print("[e5p] DONE -> paper/encoders.csv", flush=True)


if __name__ == "__main__":
    main()
