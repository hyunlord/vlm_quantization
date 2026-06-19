#!/usr/bin/env bash
# Autonomous watcher: once the 1M train download AND the 167K embed are both
# done, embed the train images and APPEND them to the 167K index -> ~1.1M corpus.
set -u
cd "$HOME/github/vlm_quantization" || exit 1
DL_LOG=/tmp/oi_train_dl.log
IDX_167K=/tmp/oi_index_167k.npz
OUT=/tmp/oi_index_1m.npz

echo "[train-embed-watch] waiting for train download (OI_DL_DONE)..."
while ! grep -q OI_DL_DONE "$DL_LOG" 2>/dev/null; do sleep 120; done
echo "[train-embed-watch] train download done $(date +%H:%M)"

echo "[train-embed-watch] waiting for 167K index ($IDX_167K)..."
while [ ! -f "$IDX_167K" ]; do sleep 120; done
echo "[train-embed-watch] 167K index ready $(date +%H:%M); embedding train (append)..."

IMG_DIR=data/image_only/oi/train \
HEADS=/tmp/demo_hashheads.pt \
REL_ROOT=data/image_only/oi \
APPEND_TO="$IDX_167K" \
OUT="$OUT" \
BATCH=384 WORKERS=32 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python /tmp/embed_corpus.py
echo "OI_1M_DONE $(date +%H:%M) -> $OUT"
