"""v2 distill spike (cycle 4) — train a small browser-deployable text encoder to match
SigLIP2-so400m text embeddings (the 1152-d `txt_h` input).

ONLY the text->1152-d embedding function is being replaced. txt_h, img_h, index.bin,
common.py, /encode_query are untouched (no retraining of the heads). If the student's
text embedding matches the teacher's *direction* (norm_in=1 -> direction is all that
matters), the frozen txt_h + cross-modal retrieval keep working.

Teacher targets are REUSED from cache (verified aligned, cos>=0.9999):
  EN: /tmp/emb_aug.pt['train']['txt'] (113K, so400m text emb of dataset_coco caption[0])
  KO: /tmp/coco_ko_pairs.pt['txt_emb'] (113K, so400m text emb of coco_ko_train caption[0])
Both are Karpathy train+restval -> the 5K test eval queries (eval_korean) are NOT here.

Student: intfloat/multilingual-e5-small (117.7M, XLM-R, transformers.js-compatible) +
mean-pool + trainable Linear(384->1152) -> L2. Loss = 1 - cos(student, teacher).

Run on DGX (GPU):
  HEAD_PATH=/tmp/ft_ko_113.pt MAX_EPOCHS=3 .venv/bin/python web/distill_train.py
Outputs /tmp/distill_e5.pt  (student backbone + projection; NOT committed).
"""
from __future__ import annotations

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

STUDENT = os.environ.get("STUDENT", "intfloat/multilingual-e5-small")
OUT = os.environ.get("OUT", "/tmp/distill_e5.pt")
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "3e-5"))
PROJ_LR = float(os.environ.get("PROJ_LR", "3e-4"))
MAX_MIN = float(os.environ.get("MAX_MIN", "45"))   # wall-clock cap (spike)
MAXLEN = int(os.environ.get("MAXLEN", "64"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
_ID = re.compile(r"_0*(\d+)\.jpg")


def _cocoid(p: str) -> int:
    m = _ID.search(p)
    return int(m.group(1)) if m else -1


def load_pairs():
    """Return (strings, teacher_emb float32 (N,1152) L2-normalized)."""
    emb_aug = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in emb_aug["ids"].tolist()]
    id2row = {c: i for i, c in enumerate(tr_ids)}
    en_teacher = emb_aug["txt"].float()
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    en_str = [(dco[c]["sentences"][0]["raw"] if c in dco and dco[c].get("sentences") else "")
              for c in tr_ids]

    ko_teacher = torch.load("/tmp/coco_ko_pairs.pt", map_location="cpu")["txt_emb"].float()
    ko_str = []
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line)
        if _cocoid(e["image_path"]) in id2row and e.get("captions"):
            ko_str.append(e["captions"][0])
    n_ko = min(len(ko_str), ko_teacher.shape[0])
    ko_str, ko_teacher = ko_str[:n_ko], ko_teacher[:n_ko]

    strings = en_str + ko_str
    teacher = torch.cat([en_teacher, ko_teacher], 0)
    teacher = F.normalize(teacher, dim=1)
    print(f"[distill] pairs: EN={len(en_str):,} KO={len(ko_str):,} total={len(strings):,}", flush=True)
    return strings, teacher


class Student(torch.nn.Module):
    def __init__(self, name: str, out_dim: int = 1152):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(name)
        h = self.backbone.config.hidden_size
        self.proj = torch.nn.Linear(h, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)   # mean pool (e5 convention)
        return F.normalize(self.proj(pooled), dim=1)


def main() -> None:
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    strings, teacher = load_pairs()
    tok = AutoTokenizer.from_pretrained(STUDENT)
    print(f"[distill] tokenizing {len(strings):,} (maxlen={MAXLEN})", flush=True)
    enc = tok(strings, padding="max_length", max_length=MAXLEN, truncation=True, return_tensors="pt")
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]

    n = len(strings)
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)
    n_val = 2000
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = Student(STUDENT).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR},
        {"params": model.proj.parameters(), "lr": PROJ_LR},
    ], weight_decay=0.01)

    def val_cos():
        model.eval()
        cs = []
        with torch.no_grad():
            for s in range(0, n_val, 512):
                b = val_idx[s:s + 512]
                p = model(ids_all[b].to(dev), mask_all[b].to(dev))
                cs.append((p * teacher[b].to(dev)).sum(1).cpu())
        model.train()
        return float(torch.cat(cs).mean())

    print(f"[distill] start: {len(tr_idx):,} train / {n_val} val | epochs<={MAX_EPOCHS} "
          f"cap={MAX_MIN}min dev={dev}", flush=True)
    print(f"[distill] val cos (init): {val_cos():.4f}", flush=True)
    step = 0
    stop = False
    for ep in range(MAX_EPOCHS):
        order = tr_idx[torch.randperm(len(tr_idx), generator=rng)]
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            p = model(ids_all[b].to(dev), mask_all[b].to(dev))
            loss = (1 - (p * teacher[b].to(dev)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 100 == 0:
                el = (time.perf_counter() - t0) / 60
                print(f"  ep{ep} step{step} loss{loss.item():.4f} ({el:.1f}min)", flush=True)
            if (time.perf_counter() - t0) / 60 > MAX_MIN:
                stop = True; break
        print(f"[distill] ep{ep} done | val cos {val_cos():.4f} | "
              f"{(time.perf_counter()-t0)/60:.1f}min", flush=True)
        if stop:
            print("[distill] hit time cap, stopping", flush=True); break

    torch.save({"student": STUDENT, "backbone": model.backbone.state_dict(),
                "proj": model.proj.state_dict(), "out_dim": 1152, "maxlen": MAXLEN}, OUT)
    print(f"[distill] DONE val_cos={val_cos():.4f} steps={step} "
          f"{(time.perf_counter()-t0)/60:.1f}min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
