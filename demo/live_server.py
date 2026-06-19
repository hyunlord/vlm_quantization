"""Live inference demo — text→image (and image→image) retrieval over a COCO
corpus, using the 1024-bit trained options (k1024_*). Separate app/port from the
comparison HTML.

Pick an OPTION (trained hash head) + BIT length, type a query → SigLIP2 text
encoder → that head's txt_h → Hamming search (faiss IndexBinaryFlat) over corpus
codes computed from the SAME head. "float" option = raw embedding cosine (ceiling).

Corpus = demo_index.npz (113K COCO images, emb+paths+captions); images served
from data/coco. Codes are recomputed per head at startup.

Run:  uvicorn demo.live_server:app --host 0.0.0.0 --port 8200
"""
from __future__ import annotations
import os, sys, time, uuid
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

MODEL = "google/siglip2-so400m-patch14-384"
INDEX = os.environ.get("LIVE_INDEX", "/tmp/demo_index.npz")
IMAGE_ROOT = os.environ.get("LIVE_IMAGE_ROOT", f"{REPO}/data/coco")
OFFER_BITS = [64, 256, 1024]
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32

# label -> head file. demo_index emb is stored L2-normalized only, so every head is
# fed normalized input (corpus+query self-consistent). The head's first LayerNorm
# absorbs input scale, so the raw-trained heads (raw-input, bit-old demo) behave
# nearly as-trained in this normalized mode.
HEAD_FILES = {
    "한국어+영어 (ft113)": "/tmp/ft_ko_113.pt",   # best 1:1 + 한국어 fine-tune (KO R@10 71.1=float추월, EN 거의 무손실)
    "best (1:2 mix)": "/tmp/sweep_c226574_coco_oi_rkd_crovca.pt",   # sweet spot (R@1 42.5)
    "1:1 mix":        "/tmp/sweep_c113287_coco_oi_rkd_crovca.pt",
    "CC12M 403K":     "/tmp/k1024_coco_oi_rkd_crovca.pt",
    "COCO baseline":  "/tmp/k1024_coco.pt",
    "COCO + CroVCA":  "/tmp/k1024_coco_crovca.pt",
    "974K capped":    "/tmp/k1024_974c_coco_oi_rkd_crovca.pt",
    "974K uncapped":  "/tmp/k1024_974u_coco_oi_rkd_crovca.pt",
    "COCO raw-input": "/tmp/k1024_raw_coco.pt",
    "bit-old demo":   "/tmp/demo_hashheads.pt",
}

# ---- multi-source corpus: COCO (demo_index) + Open Images (oi_index) ----
# (url_prefix, index_path, image_root, captions_in_index). Add more (image_root must exist) to grow.
SOURCES = [
    ("coco",  os.environ.get("LIVE_INDEX", "/tmp/demo_index.npz"), f"{REPO}/data/coco",          False),
    ("oi",    "/tmp/oi_index_167k.npz",                            f"{REPO}/data/image_only/oi", True),
    ("cc12m", "/tmp/cc12m_index.npz",                              f"{REPO}/data/cc12m_imgs",    True),  # 806K (recovered paths)
]
import re, json as _json
_capmap = {}
_dco = f"{REPO}/data/coco/dataset_coco.json"
if os.path.exists(_dco):
    for im in _json.load(open(_dco))["images"]:
        _capmap[im["cocoid"]] = im["sentences"][0]["raw"] if im.get("sentences") else ""
def _cid(p):
    m = re.search(r"_0*(\d+)\.jpg$", p); return int(m.group(1)) if m else -1
# Open Images captions: narrative jsonls keyed by image_id (= filename stem)
_oicap = {}
for _f in ["/tmp/oi_nar_val.jsonl", "/tmp/oi_nar_test.jsonl"]:
    if os.path.exists(_f):
        for _line in open(_f):
            try: _o = _json.loads(_line)
            except Exception: continue
            _iid = _o.get("image_id")
            if _iid and _iid not in _oicap: _oicap[_iid] = (_o.get("caption", "") or "")
def _stem(p): return p.rsplit("/", 1)[-1].rsplit(".", 1)[0]

emb_parts, urls, captions, MOUNTS = [], [], [], {}
for prefix, ipath, root, has_cap in SOURCES:
    if not os.path.exists(ipath):
        print(f"[live] source skip (missing index): {ipath}", flush=True); continue
    ix = np.load(ipath, allow_pickle=True)
    e = np.ascontiguousarray(ix["emb"].astype(np.float32)); e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    emb_parts.append(e)
    ps = [str(p) for p in ix["paths"]]
    urls += [f"/img/{prefix}/{p}" for p in ps]
    icap = [str(c) for c in ix["captions"]] if ("captions" in ix and has_cap) else []
    if icap and any(icap[:50]):
        captions += icap
    elif prefix == "oi":
        captions += [_oicap.get(_stem(p), "") for p in ps]          # OI narrative captions
    else:
        captions += [_capmap.get(_cid(p), "") for p in ps]          # COCO captions from filename cocoid
    MOUNTS[prefix] = root
    print(f"[live] source '{prefix}': {len(ps):,} imgs <- {ipath}", flush=True)
emb = np.concatenate(emb_parts, 0); emb_t = torch.from_numpy(emb)
N = len(urls)
print(f"[live] TOTAL corpus N={N:,} ({len(MOUNTS)} sources) | captions {sum(1 for c in captions if c):,}/{N:,}", flush=True)

try:
    import faiss; _HAVE_FAISS = True
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
except Exception:
    _HAVE_FAISS = False

from transformers import AutoModel
backbone = AutoModel.from_pretrained(MODEL, dtype=dt).to(dev).eval()
try: del backbone.vision_model
except Exception: pass
try:
    from transformers import AutoProcessor; tok = AutoProcessor.from_pretrained(MODEL).tokenizer
except Exception:
    from transformers import GemmaTokenizer; tok = GemmaTokenizer.from_pretrained(MODEL)
def _pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)

# ---- per-head: load txt_h, compute corpus img codes, build faiss indices ----
OPTIONS = {}    # label -> {"txt_h":, "bits":[...], "packed":{bit:np.uint8}, "faiss":{bit:index}}
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
for label, pth in HEAD_FILES.items():
    if not os.path.exists(pth):
        print(f"[live] SKIP {label}: {pth} missing", flush=True); continue
    h = torch.load(pth, map_location="cpu")
    bits = h["bits"]; offer = [b for b in OFFER_BITS if b in bits]
    img_h = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); img_h.load_state_dict(h["img_h"]); img_h.to(dev).eval()
    txt_h = NestedHashLayer(h["embed"], h["hidden"], bits, 0.1); txt_h.load_state_dict(h["txt_h"]); txt_h.to(dev).eval()
    bidx = {b: bits.index(b) for b in offer}
    packed = {}
    with torch.no_grad():
        for s in range(0, N, 16384):
            chunk = emb_t[s:s+16384].to(dev)
            outs = img_h(chunk)
            for b in offer:
                c01 = (outs[bidx[b]]["binary"] > 0).cpu().numpy().astype(np.uint8)
                pk = np.packbits(c01, axis=1)
                packed.setdefault(b, []).append(pk)
    packed = {b: np.ascontiguousarray(np.concatenate(v, 0)) for b, v in packed.items()}
    faiss_idx = {}
    if _HAVE_FAISS:
        for b in offer:
            ix = faiss.IndexBinaryFlat(b); ix.add(packed[b]); faiss_idx[b] = ix
    OPTIONS[label] = {"txt_h": txt_h, "bits": offer, "bidx": bidx, "packed": packed, "faiss": faiss_idx}
    del img_h
    print(f"[live] ready: {label} (bits {offer})", flush=True)
print(f"[live] options: {list(OPTIONS)} + float | faiss={_HAVE_FAISS}", flush=True)


def _encode_text(q: str):
    t = tok([q], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    am = t.get("attention_mask")
    with torch.no_grad():
        e = _pool(backbone.text_model(input_ids=t["input_ids"].to(dev),
                  attention_mask=am.to(dev) if am is not None else None)).float()
    return F.normalize(e, dim=1)              # (1,1152) normalized


def _txt_code(label, bit, en):
    with torch.no_grad():
        o = OPTIONS[label]["txt_h"](en.to(dev))
    bi = OPTIONS[label]["bidx"][bit]   # index into the head's FULL bit_list
    c01 = (o[bi]["binary"][0] > 0).cpu().numpy().astype(np.uint8)
    return np.packbits(c01)


def _hamming(packed_db, q):
    x = np.bitwise_xor(packed_db, q); return LUT[x].sum(1)


def _rows_float(qn, k):
    s = emb @ qn; o = np.argpartition(-s, k)[:k]; return [int(i) for i in o[np.argsort(-s[o])]]


def _rows_bit(label, bit, qcode, k):
    if _HAVE_FAISS:
        _, ids = OPTIONS[label]["faiss"][bit].search(qcode.reshape(1, -1), k); return [int(i) for i in ids[0]]
    d = _hamming(OPTIONS[label]["packed"][bit], qcode); o = np.argpartition(d, k)[:k]; return [int(i) for i in o[np.argsort(d[o])]]


def _results(rows, qcode=None, label=None, bit=None, qn=None):
    out = []
    for r in rows:
        d = {"pos": r, "url": urls[r], "caption": captions[r][:120]}
        if qcode is not None:
            d["hamming"] = int(_hamming(OPTIONS[label]["packed"][bit][r:r+1], qcode)[0])
        elif qn is not None:
            d["cosine"] = round(float(emb[r] @ qn), 3)
        out.append(d)
    return out


# warm up faiss thread-pool + BLAS so the FIRST real query isn't a cold outlier
try:
    _wen = _encode_text("a warmup query")
    for _lbl in OPTIONS:
        for _b in OPTIONS[_lbl]["bits"]:
            _rows_bit(_lbl, _b, _txt_code(_lbl, _b, _wen), 24)
    _rows_float(_wen[0].cpu().numpy().astype(np.float32), 24)
    print("[live] warmup done", flush=True)
except Exception as _e:
    print(f"[live] warmup skipped: {_e}", flush=True)

app = FastAPI(title="Live Hash Inference Demo (1024-bit options)")
from fastapi.staticfiles import StaticFiles
for _pfx, _root in MOUNTS.items():
    if Path(_root).is_dir():
        app.mount(f"/img/{_pfx}", StaticFiles(directory=_root), name=f"img_{_pfx}")
        print(f"[live] mounted /img/{_pfx} -> {_root}", flush=True)
_HTML = Path(__file__).resolve().parent / "live.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML.read_text(encoding="utf-8") if _HTML.exists() else "<h1>live.html missing</h1>"


@app.get("/options")
def options():
    return {"options": [{"label": l, "bits": OPTIONS[l]["bits"]} for l in OPTIONS],
            "offer_bits": OFFER_BITS, "corpus": N, "faiss": _HAVE_FAISS,
            "float_bytes": emb.shape[1] * 4}


@app.get("/search")
def search(q: str, option: str = "CC12M 403K", bit: int = 256, k: int = 24):
    t0 = time.perf_counter(); en = _encode_text(q); enc_ms = (time.perf_counter() - t0) * 1e3
    if option == "float":
        qn = en[0].cpu().numpy().astype(np.float32)
        t1 = time.perf_counter(); rows = _rows_float(qn, k); s_ms = (time.perf_counter() - t1) * 1e3
        return {"option": "float", "bytes_per_item": emb.shape[1] * 4, "encode_ms": round(enc_ms, 2),
                "search_ms": round(s_ms, 3), "corpus": N, "results": _results(rows, qn=qn)}
    if option not in OPTIONS: raise HTTPException(400, f"unknown option {option}")
    if bit not in OPTIONS[option]["bits"]: bit = OPTIONS[option]["bits"][-1]
    qcode = _txt_code(option, bit, en)
    t1 = time.perf_counter(); rows = _rows_bit(option, bit, qcode, k); s_ms = (time.perf_counter() - t1) * 1e3
    return {"option": option, "bit": bit, "bytes_per_item": bit // 8, "encode_ms": round(enc_ms, 2),
            "search_ms": round(s_ms, 3), "corpus": N, "faiss": _HAVE_FAISS,
            "results": _results(rows, qcode=qcode, label=option, bit=bit)}


@app.get("/similar")
def similar(pos: int, option: str = "CC12M 403K", bit: int = 256, k: int = 24):
    if pos < 0 or pos >= N: raise HTTPException(400, "bad pos")
    if option == "float":
        qn = emb[pos]
        t1 = time.perf_counter(); rows = _rows_float(qn, k + 1); s_ms = (time.perf_counter() - t1) * 1e3
        return {"option": "float", "seed": pos, "bytes_per_item": emb.shape[1] * 4, "corpus": N,
                "search_ms": round(s_ms, 3), "results": _results([r for r in rows if r != pos][:k], qn=qn)}
    if option not in OPTIONS: raise HTTPException(400, f"unknown option {option}")
    if bit not in OPTIONS[option]["bits"]: bit = OPTIONS[option]["bits"][-1]
    qcode = OPTIONS[option]["packed"][bit][pos]
    t1 = time.perf_counter(); rows = _rows_bit(option, bit, qcode, k + 1); s_ms = (time.perf_counter() - t1) * 1e3
    return {"option": option, "bit": bit, "seed": pos, "bytes_per_item": bit // 8, "corpus": N,
            "search_ms": round(s_ms, 3), "faiss": _HAVE_FAISS,
            "results": _results([r for r in rows if r != pos][:k], qcode=qcode, label=option, bit=bit)}
