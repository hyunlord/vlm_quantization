"""Asymmetric search: keep the QUERY float, DB stays binary (144x storage kept).

Recovers the float hash logits H (pre-sign) for every item, then compares 4 rankings
against the float-embedding gold (emb cosine = the demo's "float search"):

  float-hash  : H · H[q]            (both float — hash-space upper bound)
  symmetric   : sign(H) · sign(H[q])  (both binary — current bit search)
  asymmetric  : sign(H) · H[q]        (DB binary, query float — ADC-style)
  asym-256    : asymmetric on first 256 bits only

DB is stored as bits in all bit cases (asym only needs the float QUERY vector, ~0 cost).
Env: INDEX, K(10), NQ(50), BIT(1024)
"""
from __future__ import annotations
import os, sys, time
import numpy as np, torch

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

INDEX = os.environ.get("INDEX", "/tmp/oi_index_167k.npz")
HEADS = os.environ.get("HEADS", "/tmp/demo_hashheads.pt")
K = int(os.environ.get("K", "10")); NQ = int(os.environ.get("NQ", "50"))

hh = torch.load(HEADS, map_location="cpu"); BITS = hh["bits"]; B = 1024
ki = BITS.index(B)
img_h = NestedHashLayer(hh["embed"], hh["hidden"], BITS, 0.1)
img_h.load_state_dict(hh["img_h"]); img_h.eval()

idx = np.load(INDEX, allow_pickle=True)
emb = np.ascontiguousarray(idx["emb"].astype(np.float32))      # (N,1152) normalized
N = len(emb)
print(f"N={N:,} | recovering {B}-bit float hash logits ...", flush=True)

H = np.empty((N, B), np.float32)
with torch.no_grad():
    for i in range(0, N, 8192):
        outs = img_h(torch.from_numpy(emb[i:i+8192]))
        H[i:i+8192] = outs[ki]["binary"].numpy()
pm1 = np.ascontiguousarray(np.sign(H).astype(np.float32))      # ±1 DB bits (binary)
pm1_256 = np.ascontiguousarray(pm1[:, :256])

rng = np.random.default_rng(0); QI = [int(x) for x in rng.choice(N, NQ, replace=False)]
def topk(sc): o = np.argpartition(-sc, K-1)[:K]; return set(o.tolist())
GOLD = {qi: topk(emb @ emb[qi]) for qi in QI}                  # float-emb gold (semantic target)

def recall(score_fn):
    return float(np.mean([len(score_fn(qi) & GOLD[qi]) / K for qi in QI]))

methods = {
    "float-hash (H·H)         ": lambda qi: topk(H @ H[qi]),
    "symmetric  (±1·±1) [현재] ": lambda qi: topk(pm1 @ pm1[qi]),
    "asymmetric (±1·Hq)       ": lambda qi: topk(pm1 @ H[qi]),
    "asymmetric-256bit        ": lambda qi: topk(pm1_256 @ H[qi, :256]),
    "symmetric-256bit         ": lambda qi: topk(pm1_256 @ np.sign(H[qi, :256])),
}
print(f"\n{'method':28} | recall@{K} vs float-emb gold")
print("-" * 56)
for name, fn in methods.items():
    print(f"{name} | {recall(fn)*100:6.1f}%")
