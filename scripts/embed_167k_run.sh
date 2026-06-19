#!/usr/bin/env bash
# Embed the OI test set and append to the val index -> 167K corpus.
# Writes OI_167K_DONE ONLY on success (the npz is written only when embed
# finishes, so the flag and the artifact agree). Fresh log each run.
set -u
cd "$HOME/github/vlm_quantization" || exit 1
IMG_DIR=data/image_only/oi/test \
HEADS=/tmp/demo_hashheads.pt \
REL_ROOT=data/image_only/oi \
APPEND_TO=/tmp/oi_index.npz \
OUT=/tmp/oi_index_167k.npz \
BATCH="${BATCH:-256}" WORKERS="${WORKERS:-16}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python /tmp/embed_corpus.py
rc=$?
if [ $rc -eq 0 ] && [ -f /tmp/oi_index_167k.npz ]; then
  echo "OI_167K_DONE $(date +%H:%M)"
else
  echo "OI_167K_FAIL rc=$rc $(date +%H:%M)"
fi
