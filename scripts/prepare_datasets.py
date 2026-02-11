"""Convert various image-text datasets to unified JSONL format.

Output JSONL format (one entry per line):
    {"image_path": "relative/path.jpg", "caption": "text description"}

For multi-caption datasets, produces one entry per caption.

Usage:
    python scripts/prepare_datasets.py aihub --input /path/to/aihub --output /path/to/output
    python scripts/prepare_datasets.py cc3m --input /path/to/cc3m_tsv --output /path/to/output
    python scripts/prepare_datasets.py cc3m-ko --input /path/to/cc3m --ko-input /path/to/ko_cc3m --output /path/to/output
    python scripts/prepare_datasets.py coco-ko --input /path/to/MSCOCO_train_val_Korean.json --output /path/to/output
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def prepare_aihub(input_dir: Path, output_dir: Path) -> None:
    """Convert AI Hub #71454 (Korean-English image description) to JSONL.

    Expected input structure (adjust based on actual AI Hub download):
        input_dir/
            images/
                000001.jpg
                000002.jpg
                ...
            annotations.json   (or multiple JSON/CSV files)

    The annotation format varies by AI Hub dataset version.
    This script handles common patterns — adjust as needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "aihub_71454.jsonl"

    # Try JSON annotation file
    ann_candidates = list(input_dir.glob("*.json")) + list(
        input_dir.glob("**/*annotation*.json")
    )

    if not ann_candidates:
        print("Error: No JSON annotation files found in", input_dir)
        print("Please adjust the script for your AI Hub download format.")
        sys.exit(1)

    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for ann_file in ann_candidates:
            print(f"  Processing: {ann_file}")
            with open(ann_file, encoding="utf-8") as f:
                data = json.load(f)

            # Common AI Hub formats:
            # Format 1: {"images": [{"file_name": ..., "captions": [{"ko": ..., "en": ...}]}]}
            # Format 2: list of {"image_path": ..., "description_ko": ..., "description_en": ...}
            images = data if isinstance(data, list) else data.get("images", [])

            for item in images:
                # Determine image path
                img_path = item.get("file_name") or item.get("image_path", "")
                if not img_path:
                    continue

                # Extract captions (both Korean and English)
                captions = []
                if "captions" in item:
                    for cap in item["captions"]:
                        if isinstance(cap, dict):
                            for lang in ("ko", "en", "korean", "english"):
                                if lang in cap and cap[lang]:
                                    captions.append(cap[lang])
                        elif isinstance(cap, str):
                            captions.append(cap)

                # Fallback: direct description fields
                for key in ("description_ko", "description_en", "caption_ko",
                            "caption_en", "description", "caption"):
                    val = item.get(key)
                    if val and val not in captions:
                        captions.append(val)

                # Write one JSONL entry per caption
                for caption in captions:
                    entry = {"image_path": img_path, "caption": caption}
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

    print(f"  Wrote {count:,} entries to {jsonl_path}")


def prepare_coco_ko(input_path: Path, output_dir: Path) -> None:
    """Convert AIHub #261 COCO Korean captions JSON to JSONL.

    Input format (list of dicts):
        [{"file_path": "val2014/COCO_val2014_000000391895.jpg",
          "captions": ["English cap 1", ...],
          "id": 391895,
          "caption_ko": ["한국어 캡션 1", ...]}, ...]

    Output: one JSONL entry per Korean caption.
    data_root should point to existing COCO directory (images reused).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "coco_ko.jsonl"

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    skipped = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for item in data:
            img_path = item.get("file_path", "")
            if not img_path:
                skipped += 1
                continue

            ko_captions = item.get("caption_ko", [])
            if isinstance(ko_captions, str):
                ko_captions = [ko_captions]

            for caption in ko_captions:
                caption = caption.strip()
                if caption:
                    entry = {"image_path": img_path, "caption": caption}
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

    print(f"  Wrote {count:,} entries ({count // max(len(data), 1)} captions/image avg) to {jsonl_path}")
    if skipped:
        print(f"  Skipped {skipped} items with no file_path")


def prepare_cc3m(input_path: Path, output_dir: Path) -> None:
    """Convert CC3M TSV (image_url \\t caption) to JSONL.

    Assumes images have been pre-downloaded (e.g., via img2dataset) to:
        output_dir/images/{index:09d}.jpg

    Input TSV format: image_url<TAB>caption
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cc3m.jsonl"

    count = 0
    with open(input_path, encoding="utf-8") as f, \
         open(jsonl_path, "w", encoding="utf-8") as out:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            # Image path matches img2dataset output convention
            img_path = f"images/{i:09d}.jpg"
            caption = row[1].strip()
            if caption:
                entry = {"image_path": img_path, "caption": caption}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Wrote {count:,} entries to {jsonl_path}")


def prepare_cc3m_ko(
    input_path: Path, ko_input_path: Path, output_dir: Path
) -> None:
    """Merge CC3M English + Ko-CC3M Korean translations into bilingual JSONL.

    Each image gets two entries (one English, one Korean caption).
    Assumes images pre-downloaded to output_dir/images/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cc3m_ko.jsonl"

    # Read English captions
    en_captions: dict[int, str] = {}
    with open(input_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if len(row) >= 2 and row[1].strip():
                en_captions[i] = row[1].strip()

    # Read Korean captions (same line order as English)
    ko_captions: dict[int, str] = {}
    with open(ko_input_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if len(row) >= 2 and row[1].strip():
                ko_captions[i] = row[1].strip()
            elif len(row) >= 1 and row[0].strip():
                ko_captions[i] = row[0].strip()

    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        all_indices = sorted(set(en_captions.keys()) | set(ko_captions.keys()))
        for i in all_indices:
            img_path = f"images/{i:09d}.jpg"

            if i in en_captions:
                entry = {"image_path": img_path, "caption": en_captions[i]}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

            if i in ko_captions:
                entry = {"image_path": img_path, "caption": ko_captions[i]}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Wrote {count:,} entries to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image-text datasets to unified JSONL format"
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    # AI Hub
    p_aihub = subparsers.add_parser("aihub", help="AI Hub #71454")
    p_aihub.add_argument("--input", type=Path, required=True, help="AI Hub download dir")
    p_aihub.add_argument("--output", type=Path, required=True, help="Output dir")

    # CC3M (English only)
    p_cc3m = subparsers.add_parser("cc3m", help="CC3M English")
    p_cc3m.add_argument("--input", type=Path, required=True, help="CC3M TSV file")
    p_cc3m.add_argument("--output", type=Path, required=True, help="Output dir")

    # COCO Korean (AIHub #261)
    p_coco_ko = subparsers.add_parser("coco-ko", help="AIHub #261 COCO Korean captions")
    p_coco_ko.add_argument("--input", type=Path, required=True, help="MSCOCO_train_val_Korean.json path")
    p_coco_ko.add_argument("--output", type=Path, required=True, help="Output dir")

    # CC3M + Ko-CC3M (bilingual)
    p_cc3m_ko = subparsers.add_parser("cc3m-ko", help="CC3M + Ko-CC3M bilingual")
    p_cc3m_ko.add_argument("--input", type=Path, required=True, help="CC3M English TSV")
    p_cc3m_ko.add_argument(
        "--ko-input", type=Path, required=True, help="Ko-CC3M Korean TSV"
    )
    p_cc3m_ko.add_argument("--output", type=Path, required=True, help="Output dir")

    args = parser.parse_args()

    print(f"Preparing dataset: {args.dataset}")

    if args.dataset == "aihub":
        prepare_aihub(args.input, args.output)
    elif args.dataset == "coco-ko":
        prepare_coco_ko(args.input, args.output)
    elif args.dataset == "cc3m":
        prepare_cc3m(args.input, args.output)
    elif args.dataset == "cc3m-ko":
        prepare_cc3m_ko(args.input, args.ko_input, args.output)


if __name__ == "__main__":
    main()
