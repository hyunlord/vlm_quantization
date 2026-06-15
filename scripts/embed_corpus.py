"""대용량 이미지 폴더 → SigLIP2 1뷰 임베딩 + 256-튜닝 hash codes (데모 corpus 확장).

throughput 최적화: 1뷰(clean)만, 큰 배치, DataLoader 멀티워커(이미지 디코드 병렬),
bf16, pin_memory. GB10 sm_121 fallback에서 GPU forward는 직렬이지만 디코드/전처리를
워커로 병렬화해 GPU를 포화시킨다.

env:
  IMG_DIR     : 이미지 폴더(재귀 glob jpg/png/webp)
  HEADS       : demo_hashheads.pt (img_h 재사용; 학습 안 함)
  BATCH=192   WORKERS=16
  APPEND_TO   : 기존 demo_index.npz (있으면 병합)
  OUT         : 출력 npz (기본 /tmp/demo_index_big.npz)
  REL_ROOT    : paths를 이 루트 기준 상대경로로 (데모 /images 서빙용; 기본 IMG_DIR)
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

MODEL = "google/siglip2-so400m-patch14-384"
IMG_DIR = os.environ["IMG_DIR"]
HEADS = os.environ.get("HEADS", "/tmp/demo_hashheads.pt")
BATCH = int(os.environ.get("BATCH", "192"))
WORKERS = int(os.environ.get("WORKERS", "16"))
APPEND_TO = os.environ.get("APPEND_TO", "")
OUT = os.environ.get("OUT", "/tmp/demo_index_big.npz")
REL_ROOT = os.environ.get("REL_ROOT", IMG_DIR)
IMG = 384
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32


def load_processor():
    from transformers import AutoProcessor, GemmaTokenizer, SiglipImageProcessor, SiglipProcessor
    try:
        return AutoProcessor.from_pretrained(MODEL)
    except (AttributeError, ValueError):
        return SiglipProcessor(image_processor=SiglipImageProcessor.from_pretrained(MODEL),
                               tokenizer=GemmaTokenizer.from_pretrained(MODEL))


_proc = load_processor()


class ImgDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
            px = _proc(images=im, return_tensors="pt")["pixel_values"][0]
            return px, i, 1
        except Exception:
            return torch.zeros(3, IMG, IMG), i, 0


def main():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(IMG_DIR, "**", e), recursive=True)
    paths = sorted(paths)
    print(f"found {len(paths)} images in {IMG_DIR}", flush=True)

    hh = torch.load(HEADS, map_location="cpu"); BITS = hh["bits"]
    img_h = NestedHashLayer(hh["embed"], hh["hidden"], BITS, 0.1)
    img_h.load_state_dict(hh["img_h"]); img_h.to(dev).eval()

    from transformers import AutoModel
    bb = AutoModel.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    try:
        del bb.text_model; torch.cuda.empty_cache()
    except Exception:
        pass

    def pool(o):
        return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

    dl = DataLoader(ImgDS(paths), batch_size=BATCH, num_workers=WORKERS,
                    pin_memory=True, shuffle=False)
    chunks = {b: [] for b in BITS}; embs = []; kept_idx = []
    t0 = time.perf_counter(); n = 0
    with torch.no_grad():
        for px, idx, ok in dl:
            keep = ok.bool()
            if keep.sum() == 0:
                continue
            px = px[keep].to(dev, dtype=dt, non_blocking=True)
            e = pool(bb.vision_model(pixel_values=px)).float()
            outs = img_h(e.to(dev))
            for k, b in enumerate(BITS):
                chunks[b].append((outs[k]["binary"] > 0).cpu().numpy().astype(np.uint8))
            embs.append(F.normalize(e, dim=1).cpu().numpy().astype(np.float32))
            kept_idx += [int(i) for i, kp in zip(idx.tolist(), keep.tolist()) if kp]
            n += int(keep.sum())
            if n % (BATCH * 20) < BATCH:
                r = n / (time.perf_counter() - t0)
                print(f"  {n}/{len(paths)} ({r:.1f} img/s, ETA {(len(paths)-n)/max(r,1e-9)/60:.0f} min)", flush=True)

    kept_paths = [os.path.relpath(paths[i], REL_ROOT) for i in kept_idx]
    art = {f"packed_{b}": np.packbits(np.concatenate(chunks[b], axis=0), axis=1) for b in BITS}
    art["emb"] = np.concatenate(embs, axis=0)
    art["paths"] = np.array(kept_paths, dtype=object)
    art["captions"] = np.array([""] * len(kept_paths), dtype=object)
    art["item_ids"] = np.arange(len(kept_paths))

    if APPEND_TO and os.path.exists(APPEND_TO):
        old = np.load(APPEND_TO, allow_pickle=True)
        for b in BITS:
            art[f"packed_{b}"] = np.concatenate([old[f"packed_{b}"], art[f"packed_{b}"]], axis=0)
        art["emb"] = np.concatenate([old["emb"], art["emb"]], axis=0)
        art["paths"] = np.concatenate([old["paths"], art["paths"]])
        art["captions"] = np.concatenate([old["captions"], art["captions"]])
        print(f"appended to {APPEND_TO}: total {len(art['paths'])}", flush=True)
    art["ids"] = np.arange(len(art["paths"]))
    np.savez_compressed(OUT, **art)
    print(f"EMBED_DONE: {len(kept_paths)} new, total {len(art['paths'])} -> {OUT} "
          f"({n/(time.perf_counter()-t0):.1f} img/s avg)", flush=True)


if __name__ == "__main__":
    main()
