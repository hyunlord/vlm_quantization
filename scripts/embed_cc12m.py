"""Embed CC12M (pixparse-wds images + llavanext clean captions) -> (img_emb, txt_emb).

Join hosted images (tar shards, __key__) with clean dense captions (parquet, `key`
column) on key. Frozen SigLIP2 both towers. Shard-by-shard, checkpoints every few
shards so a multi-hour run is crash-safe and resumable.

Env: TARS(/tmp/cc12m), CAPS(/tmp/cc12m_caps), OUT(/tmp/cc12m_pairs.pt),
     CAPCOL(caption_llava), LIMIT(400000), BATCH(192), WORKERS(16), CKPT_EVERY(15)
"""
from __future__ import annotations
import os, sys, io, glob, time, tarfile
import gzip, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
TARS = os.environ.get("TARS", "/tmp/cc12m"); CAPS = os.environ.get("CAPS", "/tmp/cc12m_caps")
OUT = os.environ.get("OUT", "/tmp/cc12m_pairs.pt"); CAPCOL = os.environ.get("CAPCOL", "caption_llava")
LIMIT = int(os.environ.get("LIMIT", "400000")); BATCH = int(os.environ.get("BATCH", "192"))
WORKERS = int(os.environ.get("WORKERS", "16")); CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "15"))
APPEND_TO = os.environ.get("APPEND_TO", "")   # concat new pairs onto an existing pairs .pt
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
IMG = 384

from transformers import AutoProcessor, AutoModel
try:
    proc = AutoProcessor.from_pretrained(MODEL)
except Exception:
    from transformers import SiglipImageProcessor, GemmaTokenizer, SiglipProcessor
    proc = SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(MODEL),
                           tokenizer=GemmaTokenizer.from_pretrained(MODEL))
tok = getattr(proc, "tokenizer", None) or AutoProcessor.from_pretrained(MODEL).tokenizer

# key -> clean caption (llavanext ships train.jsonl.gz: one JSON per line)
key2cap = {}
cap_files = sorted(glob.glob(CAPS + "/**/*.jsonl.gz", recursive=True)) + \
            sorted(glob.glob(CAPS + "/**/*.jsonl", recursive=True))
for cf in cap_files:
    op = gzip.open if cf.endswith(".gz") else open
    with op(cf, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            k = d.get("key", d.get("__key__"))
            c = d.get(CAPCOL) or d.get("caption_llava") or d.get("caption")
            if k is not None and c:
                key2cap[str(k)] = c
print(f"captions loaded: {len(key2cap):,} (col={CAPCOL}, files={len(cap_files)})", flush=True)


class ShardDS(Dataset):
    def __init__(self, samples):
        self.s = samples
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        b, c = self.s[i]
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
            px = proc(images=im, return_tensors="pt")["pixel_values"][0]
            return px, c, 1
        except Exception:
            return torch.zeros(3, IMG, IMG), "", 0


def collate(batch):
    px = torch.stack([b[0] for b in batch]); caps = [b[1] for b in batch]
    ok = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    return px, caps, ok


bb = AutoModel.from_pretrained(MODEL, dtype=dt).to(dev).eval()
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)


def save(img_list, txt_list, n):
    ie = np.concatenate(img_list, 0) if img_list else np.zeros((0, 1152), np.float32)
    te = np.concatenate(txt_list, 0) if txt_list else np.zeros((0, 1152), np.float32)
    if APPEND_TO and os.path.exists(APPEND_TO):
        old = torch.load(APPEND_TO, map_location="cpu")
        ie = np.concatenate([old["img_emb"].numpy(), ie], 0)
        te = np.concatenate([old["txt_emb"].numpy(), te], 0)
    torch.save({"img_emb": torch.from_numpy(ie), "txt_emb": torch.from_numpy(te)}, OUT)
    print(f"  [ckpt] saved {len(ie):,} pairs total -> {OUT}", flush=True)


def main():
    tars = sorted(glob.glob(TARS + "/*.tar"))
    print(f"{len(tars)} tar shards | target {LIMIT:,} pairs", flush=True)
    img_list, txt_list, n, t0 = [], [], 0, time.perf_counter()
    for si, tf in enumerate(tars):
        if n >= LIMIT:
            break
        samples = []
        try:
            with tarfile.open(tf) as tar:
                for m in tar:
                    if not m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        continue
                    key = os.path.splitext(os.path.basename(m.name))[0]
                    cap = key2cap.get(key)
                    if not cap:
                        continue
                    fh = tar.extractfile(m)
                    if fh is not None:
                        samples.append((fh.read(), cap))
        except Exception as e:
            print(f"  shard {si} read error: {e}", flush=True); continue
        if not samples:
            continue
        dl = DataLoader(ShardDS(samples), batch_size=BATCH, num_workers=WORKERS,
                        collate_fn=collate, pin_memory=True)
        with torch.no_grad():
            for px, caps, ok in dl:
                keep = ok.bool()
                if keep.sum() == 0:
                    continue
                px = px[keep].to(dev, dtype=dt, non_blocking=True)
                ie = pool(bb.vision_model(pixel_values=px)).float()
                kc = [c for c, k in zip(caps, ok.tolist()) if k]
                t = tok(kc, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                am = t.get("attention_mask")
                te = pool(bb.text_model(input_ids=t["input_ids"].to(dev),
                                        attention_mask=am.to(dev) if am is not None else None)).float()
                img_list.append(F.normalize(ie, dim=1).cpu().numpy().astype(np.float32))
                txt_list.append(F.normalize(te, dim=1).cpu().numpy().astype(np.float32))
                n += int(keep.sum())
        r = n / (time.perf_counter() - t0)
        print(f"shard {si+1}/{len(tars)} done | {n:,} pairs ({r:.1f}/s, ETA {(LIMIT-n)/max(r,1e-9)/60:.0f} min)", flush=True)
        if (si + 1) % CKPT_EVERY == 0:
            save(img_list, txt_list, n)
    save(img_list, txt_list, n)
    print(f"CC12M_DONE: {n:,} pairs -> {OUT} ({n/(time.perf_counter()-t0):.1f}/s avg)", flush=True)


if __name__ == "__main__":
    main()
