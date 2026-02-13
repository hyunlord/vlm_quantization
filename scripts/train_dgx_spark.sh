#!/usr/bin/env bash
# ==============================================================================
# DGX Spark Training Launcher
# ==============================================================================
# Starts the monitoring dashboard + training in one command.
#
# Usage:
#   bash scripts/train_dgx_spark.sh                     # default config
#   bash scripts/train_dgx_spark.sh --config configs/dgx_spark.yaml
#   bash scripts/train_dgx_spark.sh --no-monitor        # skip dashboard
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
CONFIG="configs/dgx_spark.yaml"
MONITOR=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --no-monitor) MONITOR=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================="
echo "DGX Spark Training"
echo "  Config: $CONFIG"
echo "  Monitor: $MONITOR"
echo "=============================="

# GPU info
uv run python -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name} ({p.total_memory / 1024**3:.1f} GB)')
else:
    print('  WARNING: No CUDA GPU detected')
"

# Start monitoring server in background
MONITOR_PID=""
if [ "$MONITOR" = true ]; then
    echo ""
    echo "Starting monitoring dashboard on http://localhost:8000 ..."
    uv run python -m uvicorn monitor.server.app:app \
        --host 0.0.0.0 --port 8000 --log-level warning &
    MONITOR_PID=$!
    sleep 2

    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "  Dashboard: http://localhost:8000"
    else
        echo "  WARNING: Monitor server failed to start"
        MONITOR_PID=""
    fi
fi

# Detect Google Drive remote for post-training sync
GDRIVE_REMOTE=""
if command -v rclone &>/dev/null; then
    for remote in $(rclone listremotes 2>/dev/null); do
        rtype=$(rclone config show "${remote%:}" 2>/dev/null | grep "^type" | awk '{print $3}')
        if [ "$rtype" = "drive" ]; then
            GDRIVE_REMOTE="${remote%:}"
            break
        fi
    done
fi

GDRIVE_PROJECT_PATH="vlm_quantization"

sync_to_drive() {
    if [ -z "$GDRIVE_REMOTE" ]; then
        return
    fi
    echo ""
    echo "Syncing results to Google Drive..."

    # Metrics DB
    if [ -f "$PROJECT_DIR/monitor/metrics.db" ]; then
        rclone copy "$PROJECT_DIR/monitor/metrics.db" \
            "${GDRIVE_REMOTE}:${GDRIVE_PROJECT_PATH}/monitor/" 2>/dev/null \
            && echo "  Metrics DB — synced" || echo "  Metrics DB — sync failed"
    fi

    # Checkpoints
    if [ -d "$PROJECT_DIR/checkpoints" ]; then
        rclone sync "$PROJECT_DIR/checkpoints" \
            "${GDRIVE_REMOTE}:${GDRIVE_PROJECT_PATH}/checkpoints/" \
            --progress --transfers=4 2>/dev/null \
            && echo "  Checkpoints — synced" || echo "  Checkpoints — sync failed"
    fi
}

# Cleanup on exit
cleanup() {
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo ""
        echo "Stopping monitoring server..."
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    sync_to_drive
}
trap cleanup EXIT

# Run training
echo ""
echo "Starting training..."
echo "=============================="
PYTHONPATH="$PROJECT_DIR" uv run python train.py --config "$CONFIG"
