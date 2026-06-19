#!/usr/bin/env bash
# Fetch multilingual benchmarks for web/paper_multiling_run.py. Re-run safe (idempotent).
#   XM3600 (primary): captions.jsonl + images/{key}.jpg  (direct download; no HF datasets dep)
#   XTD10  (fallback): caption .txt files; images reuse local COCO (data/coco/{train,val}2014)
# Usage:  bash web/paper_multiling_fetch.sh [DATA_ROOT=data]
set -euo pipefail
ROOT="${1:-data}"

echo "== XM3600 =="
mkdir -p "$ROOT/xm3600/images"
if [ ! -f "$ROOT/xm3600/captions.jsonl" ]; then
  wget -qO /tmp/xm_caps.zip https://google.github.io/crossmodal-3600/web-data/captions.zip
  unzip -o /tmp/xm_caps.zip -d "$ROOT/xm3600" >/dev/null
fi
if [ "$(find "$ROOT/xm3600/images" -name '*.jpg' | head -1)" = "" ]; then
  wget -qO /tmp/xm_imgs.tgz https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz
  tar -xzf /tmp/xm_imgs.tgz -C "$ROOT/xm3600/images"
fi
echo "XM3600: $(find "$ROOT/xm3600/images" -name '*.jpg' | wc -l) images | captions: $(wc -l < "$ROOT/xm3600/captions.jsonl") lines"

echo "== XTD10 (fallback) =="
if [ ! -d "$ROOT/xtd10_src/XTD10" ]; then
  git clone --depth 1 https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10.git "$ROOT/xtd10_src"
fi
echo "XTD10: $(wc -l < "$ROOT/xtd10_src/XTD10/test_image_names.txt") images | langs: $(ls "$ROOT/xtd10_src/XTD10"/test_1kcaptions_*.txt 2>/dev/null | wc -l)"
