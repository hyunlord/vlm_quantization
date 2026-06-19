"""Recover image paths for the existing cc12m_pairs_big.pt embeddings WITHOUT
re-embedding. Replays embed_cc12m.py's EXACT iteration (sorted tars -> tar member
order -> keep only key-in-captions AND PIL+processor decode succeeds, in
DataLoader order, with the same LIMIT stop), collecting the key/tar per kept row.
The i-th recovered path then aligns to the i-th embedding row.

Batch order matches the embed runs:
  rows [0:403280]      <- /tmp/cc12m2  (original cc12m_pairs.pt)
  rows [403280:806560] <- /tmp/cc12m   (APPEND_TO big run)
Out: /tmp/cc12m_paths.npy  (object array "tarstem/key", len == embedding rows)
CPU only. Verify afterwards: per-batch count must equal 403,280 + cosine spot-check.
"""
from __future__ import annotations
import os, sys, io, glob, json, gzip, tarfile
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
CAPS = os.environ.get("CAPS", "/tmp/cc12m_caps"); CAPCOL = os.environ.get("CAPCOL", "caption_llava")
LIMIT = int(os.environ.get("LIMIT", "400000"))
# NOTE: embed used proc(images=im) but decode-success is determined by
# Image.open().convert("RGB") (a valid RGB image never makes the resize throw),
# so we replicate that condition only. Alignment is verified by cosine afterward.

key2cap = {}
for cf in sorted(glob.glob(CAPS + "/**/*.jsonl.gz", recursive=True)) + sorted(glob.glob(CAPS + "/**/*.jsonl", recursive=True)):
    op = gzip.open if cf.endswith(".gz") else open
    with op(cf, "rt", encoding="utf-8") as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            k = d.get("key", d.get("__key__")); c = d.get(CAPCOL) or d.get("caption_llava") or d.get("caption")
            if k is not None and c: key2cap[str(k)] = c
print(f"captions: {len(key2cap):,}", flush=True)


class DS(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        b, key = self.s[i]
        try:
            Image.open(io.BytesIO(b)).convert("RGB")
            return key, 1
        except Exception:
            return key, 0
def collate(batch): return [b[0] for b in batch], torch.tensor([b[1] for b in batch], dtype=torch.bool)


def replay(tdir):
    db = os.path.basename(tdir.rstrip("/"))   # "cc12m" or "cc12m2" (source dir)
    tars = sorted(glob.glob(tdir + "/*.tar")); rows = []; n = 0
    for tf in tars:
        if n >= LIMIT: break
        stem = os.path.splitext(os.path.basename(tf))[0]
        samples = []
        try:
            with tarfile.open(tf) as tar:
                for m in tar:
                    if not m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): continue
                    key = os.path.splitext(os.path.basename(m.name))[0]
                    if key not in key2cap: continue
                    fh = tar.extractfile(m)
                    if fh is not None: samples.append((fh.read(), key))
        except Exception as e:
            print(f"  {stem} read err: {e}", flush=True); continue
        if not samples: continue
        dl = DataLoader(DS(samples), batch_size=192, num_workers=16, collate_fn=collate)
        for keys, ok in dl:
            for k, o in zip(keys, ok.tolist()):
                if o: rows.append(f"{db}/{stem}/{k}"); n += 1
        print(f"  {stem}: n={n:,}", flush=True)
    return rows


r1 = replay("/tmp/cc12m");  print(f"BATCH1 (cc12m):  {len(r1):,}", flush=True)   # rows [0:403280]
r2 = replay("/tmp/cc12m2"); print(f"BATCH2 (cc12m2): {len(r2):,}", flush=True)   # rows [403280:]
paths = r1 + r2
np.save("/tmp/cc12m_paths.npy", np.array(paths, dtype=object))
print(f"RECOVER_DONE total={len(paths):,} (expect 806,560) -> /tmp/cc12m_paths.npy", flush=True)
