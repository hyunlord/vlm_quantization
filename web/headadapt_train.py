"""C1 (cycle 5) — train a NEW text head txt_h' that maps NATIVE e5-small text embeddings
into the FROZEN image-code space, fixing cycle 4's mismatch.

Cycle 4 forced e5 -> so400m embedding, then used the so400m-tuned txt_h: the error
compounded ((e5!=so400m) x head bit-sensitivity) -> R@10 -9pt, overlap ~0.5. C1 instead
fits the HEAD to e5: native e5 text -> new txt_h' -> 1024-bit code, aligned directly to
the FIXED image codes (ft113 img_h — the exact head that built index.bin).

Frozen: ft113 img_h (image branch + the code space), index.bin, common.py, query_server.
Trained: ONLY txt_h' = NestedHashLayer(e5_dim -> hidden -> 1024). Loss = the repo's
CombinedHashLoss (InfoNCE + EAQL + Ortho + BitBalance + LCS) with the image branch
detached (frozen img_h outputs as fixed anchors). Recipe/weights from /tmp/hp_results.json
(same as train_1024.py / ft113). Inputs are cached embeddings -> head trains fast.

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt EPOCHS=25 .venv/bin/python web/headadapt_train.py
Outputs /tmp/txt_h_e5.pt (txt_h' only; native e5 is stock/frozen -> not saved).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.losses.combined import CombinedHashLoss
from src.models.nested_hash_layer import NestedHashLayer

STUDENT = os.environ.get("STUDENT", "intfloat/multilingual-e5-small")
HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
OUT = os.environ.get("OUT", "/tmp/txt_h_e5.pt")
E5_CACHE = os.environ.get("E5_CACHE", "/tmp/e5_train_emb.pt")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = int(os.environ.get("BS", "512"))
MAXLEN = int(os.environ.get("MAXLEN", "64"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
_ID = re.compile(r"_0*(\d+)\.jpg")


def _cocoid(p):
    m = _ID.search(p)
    return int(m.group(1)) if m else -1


def e5_embed_all(strings):
    """Native e5-small mean-pool embeddings (frozen). Cached to E5_CACHE."""
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(STUDENT).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(STUDENT)
    out = []
    with torch.no_grad():
        for s in range(0, len(strings), 256):
            t = tok(strings[s:s + 256], padding="max_length", max_length=MAXLEN,
                    truncation=True, return_tensors="pt")
            o = m(t["input_ids"].to(dev), t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            e = (o * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(e.float().cpu())
            if (s // 256) % 100 == 0:
                print(f"  e5 embed {s}/{len(strings)}", flush=True)
    return torch.cat(out, 0)


def build_data():
    """Return (e5_text (N,384) L2, so400m_img (N,1152) L2) aligned pairs (EN then KO)."""
    emb_aug = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in emb_aug["ids"].tolist()]
    id2row = {c: i for i, c in enumerate(tr_ids)}
    img = emb_aug["clean"].float()  # so400m IMAGE emb (the image branch input)
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    en_caps = [(dco[c]["sentences"][0]["raw"] if c in dco and dco[c].get("sentences") else "")
               for c in tr_ids]
    en_rows = list(range(len(tr_ids)))
    ko_caps, ko_rows = [], []
    for line in open(f"{REPO}/data/coco_ko/coco_ko_train.jsonl"):
        e = json.loads(line); cid = _cocoid(e["image_path"])
        if cid in id2row and e.get("captions"):
            ko_caps.append(e["captions"][0]); ko_rows.append(id2row[cid])
    caps = en_caps + ko_caps
    rows = torch.tensor(en_rows + ko_rows)

    if os.path.exists(E5_CACHE):
        e5 = torch.load(E5_CACHE, map_location="cpu")
        if e5.shape[0] != len(caps):
            e5 = None
    else:
        e5 = None
    if e5 is None:
        print(f"[c1] computing native e5 emb for {len(caps):,} caps", flush=True)
        e5 = e5_embed_all(caps)
        torch.save(e5, E5_CACHE)
    img_pairs = img[rows]
    print(f"[c1] pairs: EN={len(en_caps):,} KO={len(ko_caps):,} total={len(caps):,} "
          f"| e5={tuple(e5.shape)} img={tuple(img_pairs.shape)}", flush=True)
    return F.normalize(e5, dim=1), F.normalize(img_pairs, dim=1)


def main():
    t0 = time.perf_counter()
    P = json.load(open("/tmp/hp_results.json"))["best_params"]
    ck = torch.load(HEAD_PATH, map_location="cpu")
    BITS = [int(b) for b in ck["bits"]]
    hidden, embed = ck["hidden"], ck["embed"]

    img_h = NestedHashLayer(embed, hidden, BITS, 0.0).to(dev).eval()
    img_h.load_state_dict(ck["img_h"])
    for p in img_h.parameters():
        p.requires_grad_(False)

    e5, img = build_data()
    e5_dim = e5.shape[1]
    N = e5.shape[0]

    txt_h = NestedHashLayer(e5_dim, hidden, BITS, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"],
                          quantization_weight=P["quant"], balance_weight=P["balance"],
                          consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(txt_h.parameters(), lr=P["lr"], weight_decay=P["wd"])
    spe = N // BS
    steps = EPOCHS * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=steps, pct_start=0.3)
    print(f"[c1] train txt_h' (e5_dim={e5_dim}->hidden={hidden}->{BITS[-1]}b) | "
          f"N={N:,} epochs={EPOCHS} steps={steps} dev={dev}", flush=True)

    torch.manual_seed(42)
    g = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = perm[s:s + BS]
            with torch.no_grad():
                io = img_h(img[idx].to(dev))            # frozen image anchors
            to = txt_h(e5[idx].to(dev))                 # trainable text head on native e5
            loss = lf(io, to, progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep{ep} loss {loss.item():.4f} ({(time.perf_counter()-t0)/60:.1f}min)", flush=True)

    txt_h.eval()
    torch.save({"txt_h": txt_h.state_dict(), "bits": BITS, "hidden": hidden,
                "embed": e5_dim, "student": STUDENT, "norm_in": 1, "mode": "c1_headadapt_e5"}, OUT)
    print(f"[c1] DONE steps={g} {(time.perf_counter()-t0)/60:.1f}min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
