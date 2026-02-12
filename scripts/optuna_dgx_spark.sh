#!/usr/bin/env bash
# ==============================================================================
# DGX Spark Optuna Hyperparameter Search Launcher
# ==============================================================================
# Starts the monitoring dashboard + Optuna search in one command.
#
# Usage:
#   bash scripts/optuna_dgx_spark.sh
#   bash scripts/optuna_dgx_spark.sh --config configs/dgx_spark.yaml --n-trials 100
#   bash scripts/optuna_dgx_spark.sh --no-monitor
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Defaults
CONFIG="configs/dgx_spark.yaml"
N_TRIALS=50
STUDY_NAME="cross_modal_hash_opt"
STORAGE="sqlite:///optuna_results.db"
SEARCH_EPOCHS=5
MONITOR=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --n-trials) N_TRIALS="$2"; shift 2 ;;
        --study-name) STUDY_NAME="$2"; shift 2 ;;
        --storage) STORAGE="$2"; shift 2 ;;
        --search-epochs) SEARCH_EPOCHS="$2"; shift 2 ;;
        --no-monitor) MONITOR=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate venv
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_DIR"
export OPTUNA_STORAGE="$STORAGE"

echo "=============================="
echo "DGX Spark Optuna Search"
echo "  Config:        $CONFIG"
echo "  Trials:        $N_TRIALS"
echo "  Study:         $STUDY_NAME"
echo "  Storage:       $STORAGE"
echo "  Search epochs: $SEARCH_EPOCHS"
echo "  Monitor:       $MONITOR"
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
    echo "  Optuna dashboard: http://localhost:8000/optuna"
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

# Run Optuna search
echo ""
echo "Starting Optuna hyperparameter search..."
echo "=============================="
python optuna_search.py \
    --config "$CONFIG" \
    --n-trials "$N_TRIALS" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --search-epochs "$SEARCH_EPOCHS"

echo ""
echo "=============================="
echo "Search complete!"
echo "  Results: $STORAGE"
echo "  Dashboard: http://localhost:8000/optuna"
echo "=============================="
