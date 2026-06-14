# Photo Search Demo — natural-language search over your own photos

A self-contained web demo of binary cross-modal hashing: point it at a folder of
photos, type a description (Korean or English), get the matching photos instantly.
The pitch is **efficiency** — each photo is stored as a tiny binary code
(32–128 bytes) and search is a bitwise XOR/Hamming scan, so it stays fast and
offline even over large libraries. See `../docs/ROADMAP.md` for the full vision.

## Two demos

- **`compare.html` + `server.py`** — the **side-by-side speed demo**: one query, float
  cosine on the left vs binary Hamming (selectable bit length) on the right, fired in
  parallel so the faster (binary) result visibly lands first, with per-search latency.
  Best for *showing off* the efficiency story. (See "Side-by-side demo" below.)
- **`index.html`** (via `src/serve/api.py`) — the **single-panel search** over your own
  photo folder. Best for *using* it on a real library.

## Side-by-side demo (float vs binary, parallel latency)

```bash
# needs: /tmp/demo_index.npz + /tmp/demo_hashheads.pt (built by build_demo_index.py
# from cached embeddings — trains opt-1024 hash heads, applies to the test corpus)
export DEMO_INDEX=/tmp/demo_index.npz DEMO_HEADS=/tmp/demo_hashheads.pt
export DEMO_IMAGE_ROOT=data/coco       # so result images render
export CORPUS_MULT=20                   # tile the corpus to make the search step non-trivial
uvicorn demo.server:app --host 0.0.0.0 --port 8100
# from your laptop:  ssh -N -L 8100:localhost:8100 dgx-spark   then open http://localhost:8100
```

Flow: the page calls `POST /encode` once (shared SigLIP2 text encode), then fires
`GET /search/float` and `GET /search/bit?bit=` in parallel and renders whichever returns
first. Encode cost is shown separately so the panels compare **pure search latency**.

## How it works

```
text query --(SigLIP2 text + hash head)--> binary code
                                              |  XOR + popcount (Hamming)
            image binary codes  <------------ |  over the packed index
                                              v
                                          top-K photos
```

Reuses the existing serving stack:
- `scripts/build_serving_index.py` — encodes a photo corpus into a packed `.npz` index.
- `src/serve/api.py` — FastAPI: `GET /` (this UI), `GET /stats`, `POST /search/text`,
  and `/images/*` static serving of the corpus.
- `demo/index.html` — the frontend (stat panel + gallery + timing).

## Run it (3 steps)

```bash
# 1. folder -> jsonl corpus
python scripts/folder_to_jsonl.py --folder ~/Photos --out data/photos.jsonl

# 2. build the packed index (needs a trained checkpoint; GPU optional, CPU works)
python scripts/build_serving_index.py \
    --checkpoint checkpoints/<run>/best.ckpt \
    --jsonl data/photos.jsonl --data-root ~/Photos \
    --bits 64,256 --out indexes/photos.npz

# 3. launch the demo
export RETRIEVAL_CHECKPOINT=checkpoints/<run>/best.ckpt
export RETRIEVAL_INDEX=indexes/photos.npz
export RETRIEVAL_IMAGE_ROOT=~/Photos        # so result images render in the browser
uvicorn src.serve.api:app --host 0.0.0.0 --port 8100
# open http://localhost:8100
```

> No personal photos handy? Build the index from COCO test images
> (`--jsonl` a COCO jsonl, `--data-root data/coco`) to demo the capability, or convert
> an existing dashboard index with `build_serving_index.py --from-pt <index.pt>`
> (no checkpoint/GPU needed).

## Notes / honest caveats

- **Multilingual.** SigLIP2 is multilingual, so Korean queries ("졸려하는 강아지")
  work without translation — a real edge over most photo apps.
- **Quality vs efficiency.** Binary codes trade some retrieval accuracy for huge
  storage/speed wins (see `docs/DATA_AND_BENCHMARKS.md`). For small libraries you can
  keep the fp32 baseline (`--save-emb`) for top quality and use hashing for scale.
- **Named people/pets** ("우리집 태기") need an *enrollment* step (few-shot
  personalization), not zero-shot retrieval — a planned follow-on feature.
