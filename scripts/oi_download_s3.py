"""Parallel Open Images downloader via unsigned S3 (no external repo dep).

Reads an id list (lines like "train/<image_id>") and downloads each
s3://open-images-dataset/<line>.jpg into <out>/<image_id>.jpg.
Resume-safe (skips files already on disk), graceful on per-image errors.

Replaces the CVDF downloader.py whose raw URL now 404s.

Env:
  IDS    : id list file (default data/image_only/train_ids.txt)
  OUT    : output dir   (default data/image_only/oi/train)
  WORKERS: thread count (default 64)
  LIMIT  : cap number of ids (0 = all)
"""
from __future__ import annotations
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "open-images-dataset"
IDS = os.environ.get("IDS", "data/image_only/train_ids.txt")
OUT = os.environ.get("OUT", "data/image_only/oi/train")
WORKERS = int(os.environ.get("WORKERS", "64"))
LIMIT = int(os.environ.get("LIMIT", "0"))

os.makedirs(OUT, exist_ok=True)
_local = threading.local()


def client():
    c = getattr(_local, "c", None)
    if c is None:
        c = _local.c = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=4))
    return c


done = 0
skip = 0
fail = 0
lock = threading.Lock()
t0 = time.perf_counter()


def fetch(line: str):
    global done, skip, fail
    line = line.strip()
    if not line:
        return
    key = line + ".jpg"                       # train/<id>.jpg
    img_id = line.split("/")[-1]
    dst = os.path.join(OUT, img_id + ".jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        with lock:
            skip += 1
        return
    try:
        client().download_file(BUCKET, key, dst)
        with lock:
            done += 1
            n = done
        if n % 5000 == 0:
            rate = (done) / (time.perf_counter() - t0)
            with lock:
                rem = total - done - skip
            print(f"  {done} dl, {skip} skip, {fail} fail "
                  f"({rate:.0f}/s, ETA {rem/max(rate,1e-9)/60:.0f} min)", flush=True)
    except Exception:
        # remove partial, count failure (image may be absent / access error)
        try:
            if os.path.exists(dst):
                os.remove(dst)
        except OSError:
            pass
        with lock:
            fail += 1


with open(IDS) as f:
    ids = [ln for ln in f.read().splitlines() if ln.strip()]
if LIMIT:
    ids = ids[:LIMIT]
total = len(ids)
print(f"downloading {total} ids -> {OUT} with {WORKERS} workers", flush=True)

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(fetch, ids))

print(f"OI_DL_DONE: {done} downloaded, {skip} skipped, {fail} failed "
      f"({(done)/(time.perf_counter()-t0):.0f}/s) -> {OUT}", flush=True)
