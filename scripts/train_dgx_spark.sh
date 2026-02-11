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

# Activate venv
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_DIR"

echo "=============================="
echo "DGX Spark Training"
echo "  Config: $CONFIG"
echo "  Monitor: $MONITOR"
echo "=============================="

# GPU info
python -c "
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
    python -m uvicorn monitor.server.app:app \
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

# Cleanup on exit
cleanup() {
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo ""
        echo "Stopping monitoring server..."
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Run training
echo ""
echo "Starting training..."
echo "=============================="
python train.py --config "$CONFIG"
