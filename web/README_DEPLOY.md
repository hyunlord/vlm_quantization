# README_DEPLOY — 1-bit cross-modal search PWA

Goal: **git clone → follow this → working PWA** (text search + on-device image indexing).
This is the consolidated deploy branch (`deploy-snapshot`, base `web-perf-panel` 7250632).
Research-only scripts (distill/head-adapt/encoder-sweep) live on their own branches and were
pruned here. **No functional change** vs `web-perf-panel`; structure-only cleanup + this doc.

## 0. What it is
Frozen **SigLIP2-So400m** backbone → **Matryoshka NestedHashLayer** → **1024-bit** codes; browser-side
**Hamming** search over a packed `index.bin` (50K COCO images, 128 bytes/image). Two query paths:
- **Server text path** (accurate): `POST /encode_query` runs the so400m text tower on DGX → 1024-bit code → browser searches.
- **Offline browser path**: transformers.js e5-small (q8) → `txt_h.onnx` (fp32) → code → browser searches. No server round-trip.
- **On-device image indexing**: MobileCLIP2 vision ONNX → `img_h_*` ONNX → code; index your own photos in-browser.

Search is pure JS (`web/static/search.js`, 256-LUT popcount), verified byte-identical to faiss/python.

## 1. Prerequisites (on the serving box, e.g. DGX)
- Python venv with the repo deps: `.venv/bin/python` (torch, transformers, fastapi, uvicorn, onnx, onnxruntime).
- Checkpoints in `/tmp` (regenerate via training branches if missing):
  - `/tmp/ft_ko_113.pt` — the deployed hash head (img_h + txt_h, bits [8..1024], embed 1152, hidden 384).
  - `/tmp/demo_index.npz` — cached SigLIP2 embeddings + image paths for the corpus.
  - `/tmp/imgh_mobileclip2-s2.pt`, `/tmp/imgh_mobileclip2-s0.pt` — image-side head-adapt checkpoints (only for the image-indexing ONNX).
- Images at `~/data/coco` (for thumbnails).
- For the public tunnel: `cloudflared` installed.

## 2. Build the gitignored artifacts
`.gitignore` excludes `static/data/`, `static/thumbs/`, `static/onnx/` — rebuild them:

```bash
cd ~/github/vlm_quantization
PY=.venv/bin/python

# (a) packed index + meta + thumbnails (50K)  -> web/static/data/{index.bin,meta.json,index_info.json}, web/static/thumbs/*.jpg
HEAD_PATH=/tmp/ft_ko_113.pt $PY web/build_index.py \
  --n 50000 --seed 42 --index /tmp/demo_index.npz --image-root ~/data/coco --out web/static
# (add --reuse-thumbs on rebuilds to skip thumbnail regeneration)

# (b) text head ONNX (offline browser path)  -> web/static/onnx/txt_h.onnx
HEAD_PATH=/tmp/ft_ko_113.pt $PY web/export_txth_onnx.py

# (c) image-side ONNX (on-device indexing). Needs /tmp/imgh_mobileclip2-s{0,2}.pt
$PY web/export_vis_s0.py      # -> vis_mobileclip2-s0.onnx (~45MB) + img_h_mobileclip2-s0.onnx  (lighter, phone default)
$PY web/export_vis_onnx.py    # -> vis_mobileclip2-s2.onnx (~144MB) + img_h_mobileclip2-s2.onnx (higher R@10)
$PY web/quant_vis.py          # -> vis_mobileclip2-s2.{int8,fp16}.onnx (smaller, browser)
```
Each export self-checks torch-vs-onnxruntime parity (max|Δ|~1e-7) and the full chain vis→img_h→packbits.

## 3. Run the server
```bash
# port 8300 (8200/8123 are older demo servers — avoid)
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python -m uvicorn web.query_server:app --host 0.0.0.0 --port 8300
# persistent: run inside tmux, e.g.  tmux new -s web ...  (log to /tmp/web_server.log)
```
Local check: `curl -sI http://127.0.0.1:8300/` → 200. Endpoints: `/`, `/app.js`, `/search.js`, `/sw.js`,
`/manifest.webmanifest`, `POST /encode_query`, `/data/index.bin`, `/data/meta.json`, `/thumbs/<id>.jpg`,
`/onnx/*.onnx`. Add `?perf=1` (text) or `?perf=1&img=1` (image) for the latency panels.

## 4. Public access — Cloudflare tunnel (recommended)
A PWA service worker needs a **secure context** (https or localhost). Cloudflare Quick Tunnel gives a
free https URL with no account and no inbound firewall changes — cleaner than ngrok (token/limits) or
Tailscale (per-device). Lessons learned: prefer this over ngrok for phone testing.

```bash
# one-time: install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
# then, with the server already on :8300:
cloudflared tunnel --url http://localhost:8300
# prints a https://<random>.trycloudflare.com URL — open it on your phone; install as PWA (Add to Home Screen)
```
For a stable named URL use a Cloudflare account: `cloudflared tunnel login` → `cloudflared tunnel create <name>`
→ route DNS → `cloudflared tunnel run <name>`. The Quick Tunnel above is sufficient for demos/phone tests.

Alternatives (kept for reference): SSH tunnel `ssh -L 8300:localhost:8300 dgx-spark` → `http://localhost:8300/`
(localhost = secure context, good for laptop); Tailscale `http://<ts-ip>:8300/` if the subnet is routed.

## 5. Smoke / verify
```bash
.venv/bin/python web/smoke_retrieval.py        # server retrieval sanity
node web/verify_js.mjs                          # JS search == python (byte-parity)
.venv/bin/python web/verify_parity.py           # head ONNX vs torch parity
```

## 6. File map (deploy essentials only)
- `web/query_server.py` — FastAPI: `/encode_query` + static serving.
- `web/common.py` — shared Encoder / pack_bits / hamming_topk (parity contract).
- `web/build_index.py` — builds `index.bin` + thumbnails from cached embeddings.
- `web/export_txth_onnx.py` / `export_vis_onnx.py` / `export_vis_s0.py` / `quant_vis.py` — ONNX exports.
- `web/static/{index.html,app.js,search.js,sw.js,manifest.webmanifest,perf.js,perf_img.js}` — the PWA.
- `web/{HANDOFF_v1,HANDOFF_v2,WEB_HYBRID,PERF_PANEL,PERF_IMG_PANEL}.md` — design/verification notes.
- See `../STATE_OF_PROJECT.md` and `../ASSETS.md` for the full project record + asset inventory.
