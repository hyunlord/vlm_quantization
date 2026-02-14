#!/usr/bin/env bash
# ==============================================================================
# Monitoring Dashboard Server
# ==============================================================================
# Start/stop the monitoring dashboard independently from training.
#
# Usage:
#   bash scripts/monitor.sh              # start (foreground)
#   bash scripts/monitor.sh --background # start (background, writes PID file)
#   bash scripts/monitor.sh --stop       # stop background server
#   bash scripts/monitor.sh --status     # check if running
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${MONITOR_HOST:-0.0.0.0}"
PORT="${MONITOR_PORT:-8001}"
PID_FILE="$PROJECT_DIR/.monitor.pid"

start_server() {
    local background="${1:-false}"

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Monitor already running (PID $pid) at http://${HOST}:${PORT}"
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting monitoring dashboard on http://${HOST}:${PORT} ..."

    if [ "$background" = true ]; then
        PYTHONPATH="$PROJECT_DIR" uv run python -m uvicorn monitor.server.app:app \
            --host "$HOST" --port "$PORT" --log-level warning &
        local pid=$!
        sleep 2

        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid" > "$PID_FILE"
            echo "  Dashboard running in background (PID $pid)"
            echo "  Stop with: bash scripts/monitor.sh --stop"
        else
            echo "  ERROR: Monitor server failed to start"
            return 1
        fi
    else
        echo "  Press Ctrl+C to stop"
        echo ""
        PYTHONPATH="$PROJECT_DIR" uv run python -m uvicorn monitor.server.app:app \
            --host "$HOST" --port "$PORT"
    fi
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping monitor server (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "  Stopped."
        else
            echo "Monitor server not running (stale PID file removed)."
            rm -f "$PID_FILE"
        fi
    else
        echo "No monitor server PID file found."
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Monitor running (PID $pid) at http://${HOST}:${PORT}"
            return 0
        else
            echo "Monitor not running (stale PID file)."
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "Monitor not running."
        return 1
    fi
}

# Returns 0 if monitor is running (for use by other scripts)
is_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        kill -0 "$pid" 2>/dev/null && return 0
    fi
    return 1
}

# Parse arguments
case "${1:-}" in
    --stop)
        stop_server
        ;;
    --status)
        check_status
        ;;
    --background|-b)
        start_server true
        ;;
    --is-running)
        is_running
        ;;
    ""|--foreground)
        start_server false
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: bash scripts/monitor.sh [--background|--stop|--status]"
        exit 1
        ;;
esac
