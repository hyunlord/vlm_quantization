"""Script-free fetch + export of extra image-text datasets.

`datasets >= 4` dropped loading-script support, so HF repos like nlphuji/flickr30k
and google/docci can no longer be read via load_dataset(). We pull the raw files
(HF direct download / GCS tarball) and assemble them ourselves into the
GenericImageTextDataset format so they plug into both scripts/zeroshot_benchmark.py
and the training `extra_datasets` config hook.

Output per dataset:
    <data_root>/<name>/images/*            (image files)
    <data_root>/<name>/<name>_<split>.jsonl   ({"image_path","captions"|"caption","image_id"})

Usage:
    python scripts/fetch_extra_datasets.py --data-root data --datasets flickr30k docci
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import tarfile
import traceback
import zipfile
from pathlib import Path


def save_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  -> {path} ({len(items)} rows)", flush=True)


def export_flickr(data_root: Path) -> None:
    """Flickr30K from nlphuji/flickr30k raw files (images zip + annotations csv).

    Karpathy split lives in the csv `split` column (train 29K / val 1K / test 1K).
    """
    print("== Flickr30K (nlphuji/flickr30k direct files) ==", flush=True)
    from huggingface_hub import hf_hub_download

    out = data_root / "flickr30k"
    (out / "images").mkdir(parents=True, exist_ok=True)
    csv_path = hf_hub_download("nlphuji/flickr30k", "flickr_annotations_30k.csv", repo_type="dataset")
    zip_path = hf_hub_download("nlphuji/flickr30k", "flickr30k-images.zip", repo_type="dataset")

    have = {p.name for p in (out / "images").glob("*.jpg")}
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".jpg")]
        for i, m in enumerate(members):
            name = Path(m).name
            if name not in have:
                (out / "images" / name).write_bytes(z.read(m))
            if i % 5000 == 0:
                print(f"  flickr imgs {i}/{len(members)}", flush=True)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    cap_col = "raw" if "raw" in cols else ("caption" if "caption" in cols else "captions")
    buckets: dict[str, list] = {}
    for i, r in enumerate(rows):
        cap = r[cap_col]
        try:
            caps = ast.literal_eval(cap) if isinstance(cap, str) and cap.strip().startswith("[") else [cap]
        except (ValueError, SyntaxError):
            caps = [cap]
        iid = r.get("img_id", i)
        buckets.setdefault(r.get("split", "test"), []).append({
            "image_path": f"images/{r['filename']}",
            "captions": list(caps),
            "image_id": int(iid) if str(iid).isdigit() else i,
        })
    for sp, items in buckets.items():
        save_jsonl(out / f"flickr30k_{sp}.jsonl", items)
    print("  NOTE: extra_datasets data_root for flickr30k = <data_root>/flickr30k", flush=True)


def export_docci(data_root: Path) -> None:
    """DOCCI from the authoritative GCS tarball (no HF). ~136-word dense captions."""
    print("== DOCCI (GCS tarball) ==", flush=True)
    out = data_root / "docci"
    out.mkdir(parents=True, exist_ok=True)
    desc = out / "docci_descriptions.jsonlines"
    tarp = out / "docci_images.tar.gz"
    base = "https://storage.googleapis.com/docci/data"
    if not desc.exists() or desc.stat().st_size == 0:
        subprocess.run(["wget", "-q", "-O", str(desc), f"{base}/docci_descriptions.jsonlines"], check=True)
    if not tarp.exists() or tarp.stat().st_size < 7_000_000_000:
        print("  downloading 7.6GB image tarball ...", flush=True)
        subprocess.run(["wget", "-q", "-c", "-O", str(tarp), f"{base}/docci_images.tar.gz"], check=True)
    if not (out / "images").exists():
        print("  extracting tarball ...", flush=True)
        with tarfile.open(tarp) as t:
            t.extractall(out)
    rows = [json.loads(line) for line in open(desc) if line.strip()]
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r.get("split", "train"), []).append({
            "image_path": r["image_file"],   # bare filename -> data_root must be <data_root>/docci/images
            "caption": r["description"],
            "image_id": r.get("example_id"),
        })
    for sp, items in buckets.items():
        save_jsonl(out / f"docci_{sp}.jsonl", items)
    print("  NOTE: extra_datasets data_root for docci = <data_root>/docci/images (bare filenames)", flush=True)


EXPORTERS = {"flickr30k": export_flickr, "docci": export_docci}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--datasets", nargs="+", default=["flickr30k", "docci"], choices=list(EXPORTERS))
    args = ap.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        try:
            EXPORTERS[name](args.data_root)
        except Exception:
            traceback.print_exc()
    print("FETCH_EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
