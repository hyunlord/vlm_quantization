"""(d) mAP@10 extension for the bit-length sweep. Adds mAP@10 next to R@10 for
64/128/256/512/1024-bit, server (so400m+ft113) and deployed offline (e5-small), EN/KO,
on the eval_korean 5K protocol (same rankings as web/eval_paper.py section B).

(The multilingual mAP@10 is produced by (a) web/paper_baselines_multiling.py -> the mAP10
column of paper/baselines_multiling.csv, so this script only extends the bit sweep.)

Cross-check: R@10 here must match paper/bits_sweep.csv (e.g. 1024 server EN 79.92 / KO 71.08,
offline EN 74.0 / KO 66.2). server mAP@10 at 1024 must match review_map.csv (EN 53.34 / KO 42.49).

Run on DGX: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_map_extension.py
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

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, Encoder, pack_bits  # noqa: E402

PAPER = Path(REPO) / "paper"
_ID = re.compile(r"_0*(\d+)\.jpg")
SWEEP_DEFAULT = "64,128,256,512,1024"


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def recall_at10(ix, q, gold):
    _, I = ix.search(q, 10)
    return round(100 * np.mean([gold[i] in I[i, :10] for i in range(len(gold))]), 2)


def map_at10(ix, q, gold):
    _, I = ix.search(q, 10)
    aps = []
    for i in range(len(gold)):
        hit = np.where(I[i, :10] == gold[i])[0]
        aps.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return round(100 * float(np.mean(aps)), 2)


def head_codes(head, emb_np, enc, bit):
    et = enc._prep(torch.from_numpy(np.ascontiguousarray(emb_np, dtype=np.float32)).to(enc.device))
    with torch.no_grad():
        outs = head(et)
    bi = [int(b) for b in enc.bit_list].index(bit)
    return outs[bi]["binary"].float().cpu().numpy()


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    bit_list = [int(b) for b in enc.bit_list]
    sweep = [b for b in (int(x) for x in os.environ.get("SWEEP", SWEEP_DEFAULT).split(",")) if b in bit_list]

    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()
    TXT = {"EN": EC["test"]["txt"].float().numpy(), "KO": KO["txt_emb"].float().numpy()}
    gold = list(range(len(te_ids)))

    # deployed offline head (e5-small, C1 no-prefix)
    ck = torch.load(os.environ.get("E5_HEAD", "/tmp/txt_h_e5.pt"), map_location="cpu")
    e5_th = NestedHashLayer(ck["embed"], ck["hidden"], [int(b) for b in ck["bits"]], 0.0)
    e5_th.load_state_dict(ck["txt_h"]); e5_th.to(enc.device).eval()
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F
    e5m = AutoModel.from_pretrained(ck["student"]).to(enc.device).eval()
    e5tok = AutoTokenizer.from_pretrained(ck["student"])
    en_caps = [str(c) for c in EC["test"]["captions"]]
    kf = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); mm = _ID.search(e["image_path"])
        if mm:
            kf[int(mm.group(1))] = e.get("captions", [])
    ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]

    @torch.no_grad()
    def e5_emb(strings, batch=256):
        out = []
        for s in range(0, len(strings), batch):
            t = e5tok([(ck.get("prefix") or "") + x for x in strings[s:s + batch]], padding="max_length",
                      max_length=64, truncation=True, return_tensors="pt")
            o = e5m(input_ids=t["input_ids"].to(enc.device),
                    attention_mask=t["attention_mask"].to(enc.device)).last_hidden_state
            msk = t["attention_mask"].to(enc.device).unsqueeze(-1).float()
            e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(F.normalize(e, dim=1).float().cpu())
        return torch.cat(out).numpy()

    off_emb = {"EN": e5_emb(en_caps), "KO": e5_emb(ko_caps)}

    rows = []
    for bit in sweep:
        gal = pack_bits(head_codes(enc.img_h, te_img, enc, bit))
        ix = faiss_bin(gal, bit)
        row = {"bits": bit}
        for L in ("EN", "KO"):
            q = pack_bits(head_codes(enc.txt_h, TXT[L], enc, bit))
            row[f"server_{L}_R10"] = recall_at10(ix, q, gold)
            row[f"server_{L}_mAP10"] = map_at10(ix, q, gold)
            oq = pack_bits(head_codes(e5_th, off_emb[L], enc, bit))
            row[f"off_{L}_R10"] = recall_at10(ix, oq, gold)
            row[f"off_{L}_mAP10"] = map_at10(ix, oq, gold)
        rows.append(row)
        print(f"[d] bit {bit}: server EN R@10 {row['server_EN_R10']} mAP@10 {row['server_EN_mAP10']} | "
              f"KO R@10 {row['server_KO_R10']} mAP@10 {row['server_KO_mAP10']} || "
              f"offline EN mAP@10 {row['off_EN_mAP10']} KO mAP@10 {row['off_KO_mAP10']}", flush=True)

    cols = ["bits", "server_EN_R10", "server_EN_mAP10", "server_KO_R10", "server_KO_mAP10",
            "off_EN_R10", "off_EN_mAP10", "off_KO_R10", "off_KO_mAP10"]
    with open(PAPER / "map_extension.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("[d] RESULT_JSON " + json.dumps({"anchor_check": {"1024_server_EN_R10": rows[-1]["server_EN_R10"],
          "1024_server_EN_mAP10": rows[-1]["server_EN_mAP10"]}, "rows": rows}), flush=True)
    print(f"[d] DONE -> paper/map_extension.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
