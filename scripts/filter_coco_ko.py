"""Filter a Korean-COCO JSONL to remove Karpathy val/test images (leakage fix).

`coco_ko.jsonl` pairs Korean captions with the FULL COCO 2014 image set, which
includes the Karpathy `val` and `test` images used for evaluation. Training on it
leaks the eval images. This script keeps only `train`+`restval` images.

Usage:
    python scripts/filter_coco_ko.py \
        --coco-ko data/coco_ko/coco_ko.jsonl \
        --karpathy-json data/coco/dataset_coco.json \
        --out data/coco_ko/coco_ko_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ko", required=True, help="input Korean-COCO JSONL")
    ap.add_argument("--karpathy-json", required=True, help="dataset_coco.json")
    ap.add_argument("--out", required=True, help="filtered output JSONL")
    ap.add_argument("--image-key", default="image_path")
    a = ap.parse_args()

    data = json.load(open(a.karpathy_json))
    keep = {e["filename"] for e in data["images"] if e["split"] in ("train", "restval")}
    heldout = {e["filename"] for e in data["images"] if e["split"] in ("test", "val")}

    n = kept = 0
    with open(a.coco_ko) as f, open(a.out, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if os.path.basename(obj.get(a.image_key, "")) in keep:
                g.write(line + "\n")
                kept += 1

    leak = sum(
        1 for line in open(a.out)
        if os.path.basename(json.loads(line)[a.image_key]) in heldout
    )
    print(f"{a.coco_ko}: {n} -> {a.out}: {kept} (train+restval only)")
    print(f"held-out (test/val) leakage in output: {leak}  (must be 0)")
    if leak:
        raise SystemExit("ERROR: leakage detected in filtered output")


if __name__ == "__main__":
    main()
