"""Scan a photo folder (recursively) into a JSONL corpus for build_serving_index.py.

Each line: {"image_path": "<path relative to --relative-to>"}.  Pair the same
--relative-to value with build_serving_index.py's --data-root and the demo's
RETRIEVAL_IMAGE_ROOT so paths resolve everywhere.

    python scripts/folder_to_jsonl.py --folder ~/Photos --out data/photos.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, help="photo folder to scan (recursively)")
    ap.add_argument("--out", default="data/photos.jsonl")
    ap.add_argument("--relative-to", default=None,
                    help="paths are written relative to this root (default: --folder)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    root = Path(args.relative_to).expanduser().resolve() if args.relative_to else folder
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w") as f:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                f.write(json.dumps({"image_path": str(p.resolve().relative_to(root))}) + "\n")
                n += 1
    print(f"wrote {out} ({n:,} images, relative to {root})")
    print(f"next: build_serving_index.py --jsonl {out} --data-root {root} ...")


if __name__ == "__main__":
    main()
