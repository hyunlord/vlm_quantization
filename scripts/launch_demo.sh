#!/usr/bin/env bash
# Launch the float-vs-binary demo server, detached. Env overrides supported.
set -u
cd "${REPO:-$HOME/github/vlm_quantization}" || exit 1
PORT="${PORT:-8123}"
LOG="${LOG:-/tmp/smoke_demo.log}"
pkill -f "uvicorn demo.server" 2>/dev/null || true
sleep 1
rm -f "$LOG"
export DEMO_INDEX="${DEMO_INDEX:-/tmp/oi_index.npz}"
export DEMO_HEADS="${DEMO_HEADS:-/tmp/demo_hashheads.pt}"
export DEMO_IMAGE_ROOT="${DEMO_IMAGE_ROOT:-data/image_only/oi}"
export CORPUS_MULT="${CORPUS_MULT:-1}"
export REPO="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup .venv/bin/uvicorn demo.server:app --host 127.0.0.1 --port "$PORT" \
  >"$LOG" 2>&1 </dev/null &
disown 2>/dev/null || true
echo "launched uvicorn on :$PORT (log: $LOG)"
exit 0
