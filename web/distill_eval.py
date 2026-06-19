"""v2 distill spike (cycle 4) — parity evaluation of the distilled student text encoder.

Gate = RETRIEVAL parity (Top-K / R@K), not cosine. The student replaces ONLY the
text->1152-d embedding; txt_h, img_h, index.bin stay frozen (reused via web.common.Encoder).

Reports, per precision (fp32/fp16[/int8]), KO and EN separately:
  - R@10 on the 5K COCO test gallery (eval_korean protocol -> apples-to-apples with 71/80):
    student-text-code vs frozen image codes; also the so400m SERVER baseline measured here.
  - Top-10 overlap vs the so400m server query over the real web/static/data/index.bin (50K).
  - cosine(student emb, teacher emb) — diagnostic only (distill won't hit 0.999).

Hold-out queries only: EN = emb_cache['test']['captions'], KO = coco_ko.jsonl test caps
(rebuilt) — neither is in the train+restval distill set.

Run on DGX (after distill_train.py):
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/distill_eval.py --student /tmp/distill_e5.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import CODE_BITS, CODE_BYTES, Encoder, pack_bits  # noqa: E402
from web.distill_train import Student  # noqa: E402

_ID = re.compile(r"_0*(\d+)\.jpg")


def _cocoid(p: str) -> int:
    m = _ID.search(p)
    return int(m.group(1)) if m else -1


def _faiss_index(packed):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(CODE_BITS)
    ix.add(packed)
    return ix


def _recall_at(gallery_ix, query_codes, gold_ids, gallery_ids, k=10):
    _, I = gallery_ix.search(query_codes, k)
    hit = sum(1 for q in range(len(gold_ids))
              if gold_ids[q] in [gallery_ids[j] for j in I[q]])
    return round(100 * hit / len(gold_ids), 2)


class StudentRunner:
    def __init__(self, path, device=None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        ck = torch.load(path, map_location="cpu")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.maxlen = ck.get("maxlen", 64)
        self.model = Student(ck["student"])
        self.model.backbone.load_state_dict(ck["backbone"])
        self.model.proj.load_state_dict(ck["proj"])
        self.model.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(ck["student"])
        self.name = ck["student"]

    def embed(self, strings, batch=256, half=False):
        torch = self.torch
        m = self.model.half() if half else self.model.float()
        out = []
        with torch.no_grad():
            for s in range(0, len(strings), batch):
                t = self.tok(strings[s:s + batch], padding="max_length", max_length=self.maxlen,
                             truncation=True, return_tensors="pt")
                e = m(t["input_ids"].to(self.device), t["attention_mask"].to(self.device))
                out.append(e.float().cpu().numpy())
        self.model.float()
        return np.concatenate(out, 0)


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--student", default="/tmp/distill_e5.pt")
    p.add_argument("--static", default="web/static")
    p.add_argument("--n-overlap", type=int, default=1000, help="held-out queries for index.bin overlap")
    p.add_argument("--precisions", default="fp32,fp16")
    args = p.parse_args()

    enc = Encoder()  # frozen txt_h/img_h + packing (the demo's path)
    stu = StudentRunner(args.student)

    # ---- hold-out test queries (5K) ----
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    te_en = EC["test"]["txt"].float().numpy()
    te_img = EC["test"]["img"].float().numpy()
    ko_emb = KO["txt_emb"].float().numpy()
    ko_full = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line)
        ko_full[_cocoid(e["image_path"])] = e.get("captions", [])
    ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]
    print(f"[eval] hold-out test: {len(te_ids)} | EN caps {len(en_caps)} | KO caps {len(ko_caps)} "
          f"(missing {sum(1 for c in ko_caps if not c)})", flush=True)

    # frozen gallery image codes (5K) and server (teacher) text codes
    gal = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    gal_ix = _faiss_index(gal)
    srv = {"EN": pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(te_en))),
           "KO": pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(ko_emb)))}
    srv_R = {L: _recall_at(gal_ix, srv[L], te_ids, te_ids) for L in ("EN", "KO")}
    print(f"[eval] SERVER (so400m) R@10 on 5K gallery: EN {srv_R['EN']} | KO {srv_R['KO']} "
          f"(published ft113 ~80/~71)", flush=True)

    # real serving index for Top-10 overlap
    packed = np.fromfile(Path(args.static) / "data/index.bin", dtype=np.uint8).reshape(-1, CODE_BYTES)
    idx50 = _faiss_index(np.ascontiguousarray(packed))
    rng = np.random.default_rng(42)
    osel = np.sort(rng.choice(len(te_ids), min(args.n_overlap, len(te_ids)), replace=False))

    results = {"student": stu.name, "server_R10": srv_R}
    teacher = {"EN": te_en, "KO": ko_emb}
    caps = {"EN": en_caps, "KO": ko_caps}
    for prec in args.precisions.split(","):
        prec = prec.strip()
        half = prec == "fp16"
        if prec not in ("fp32", "fp16"):
            continue
        res = {}
        for L in ("EN", "KO"):
            try:
                t0 = time.perf_counter()
                semb = stu.embed(caps[L], half=half)
                lat = (time.perf_counter() - t0) / len(caps[L]) * 1e3
                scode = pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(semb)))
                r10 = _recall_at(gal_ix, scode, te_ids, te_ids)
                a = semb / (np.linalg.norm(semb, axis=1, keepdims=True) + 1e-9)
                b = teacher[L] / (np.linalg.norm(teacher[L], axis=1, keepdims=True) + 1e-9)
                cos = float((a * b).sum(1).mean())
                _, Is = idx50.search(scode[osel], 10)
                _, Iv = idx50.search(srv[L][osel], 10)
                ov = float(np.mean([len(set(Is[i]) & set(Iv[i])) / 10 for i in range(len(osel))]))
                res[L] = {"R10": r10, "R10_delta_vs_server": round(r10 - srv_R[L], 2),
                          "top10_overlap": round(ov, 3), "cos_vs_teacher": round(cos, 4),
                          "ms_per_query": round(lat, 1)}
                print(f"[eval] {prec} {L}: R@10 {r10} (Δserver {res[L]['R10_delta_vs_server']:+}) | "
                      f"overlap {res[L]['top10_overlap']} | cos {res[L]['cos_vs_teacher']} | "
                      f"{res[L]['ms_per_query']} ms/q", flush=True)
            except Exception as e:  # keep the run alive (e.g., fp16 unsupported op on GB10)
                res[L] = {"error": repr(e)[:160]}
                print(f"[eval] {prec} {L}: FAILED {res[L]['error']}", flush=True)
        results[prec] = res

    npar = sum(p.numel() for p in stu.model.parameters())
    results["size_mb"] = {"fp32": round(npar * 4 / 1e6), "fp16": round(npar * 2 / 1e6),
                          "int8": round(npar * 1 / 1e6)}
    print(f"[eval] student size: {results['size_mb']} MB (params {npar/1e6:.1f}M)", flush=True)
    print("[eval] RESULT_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
