#!/usr/bin/env bash
# ==============================================================================
# DGX Spark One-Time Setup
# ==============================================================================
# Run this once after cloning the repo on your DGX Spark.
#
# Usage:
#   cd /path/to/vlm_quantization
#   bash scripts/setup_dgx_spark.sh
#
# What it does:
#   1. Installs uv + creates venv + installs deps (PyTorch CUDA 12.4)
#   2. Downloads COCO dataset + Karpathy split
#   3. Syncs extra datasets from Google Drive (AIHub, CC3M) + auto-prepares JSONL
#   4. Builds the monitoring dashboard frontend
#
# Prerequisites for Google Drive sync:
#   rclone config  (add a remote of type "drive", e.g. named "gdrive")
#   Data expected at: <drive>/data/aihub/ and <drive>/data/cc3m_ko/
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=============================="
echo "DGX Spark Setup"
echo "Project: $PROJECT_DIR"
echo "=============================="

# ---------- 1. Python environment (uv) ----------
echo ""
echo "[1/4] Setting up Python environment with uv..."

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "  uv: $(uv --version)"

# Clear cached resolution to avoid stale PyTorch-index transitive deps
uv cache clean markupsafe 2>/dev/null || true

# uv sync creates .venv and installs all deps from pyproject.toml
# PyTorch CUDA index is configured in pyproject.toml [tool.uv]
uv sync

echo "  Python: $(uv run python --version)"
uv run python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ---------- 2. Download COCO dataset ----------
DATA_DIR="$PROJECT_DIR/data/coco"
echo ""
echo "[2/4] Setting up COCO dataset..."

mkdir -p "$DATA_DIR"

download_and_extract() {
    local name="$1"
    local url="$2"
    local target_dir="$3"

    if [ -d "$target_dir" ]; then
        echo "  $name — already exists, skipping"
        return
    fi

    echo "  $name — downloading..."
    wget -q --show-progress "$url" -O "/tmp/$name.zip"
    echo "  $name — extracting..."
    unzip -q "/tmp/$name.zip" -d "$DATA_DIR/"
    rm "/tmp/$name.zip"
}

download_and_extract "train2014" \
    "http://images.cocodataset.org/zips/train2014.zip" \
    "$DATA_DIR/train2014"

download_and_extract "val2014" \
    "http://images.cocodataset.org/zips/val2014.zip" \
    "$DATA_DIR/val2014"

download_and_extract "annotations" \
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
    "$DATA_DIR/annotations"

# Karpathy split
KARPATHY_JSON="$DATA_DIR/dataset_coco.json"
if [ ! -f "$KARPATHY_JSON" ]; then
    echo "  Karpathy split — downloading..."
    wget -q --show-progress \
        "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" \
        -O "/tmp/caption_datasets.zip"
    unzip -q -j "/tmp/caption_datasets.zip" "dataset_coco.json" -d "$DATA_DIR/"
    rm "/tmp/caption_datasets.zip"
else
    echo "  Karpathy split — already exists, skipping"
fi

# Verify
echo ""
echo "  COCO verification:"
echo "    train2014: $(ls "$DATA_DIR/train2014/" | wc -l) images"
echo "    val2014:   $(ls "$DATA_DIR/val2014/" | wc -l) images"
echo "    Karpathy:  $([ -f "$KARPATHY_JSON" ] && echo 'OK' || echo 'MISSING')"

# ---------- 3. Extra datasets (auto-sync from Google Drive) ----------
echo ""
echo "[3/4] Extra datasets..."

# Install rclone if not present
if ! command -v rclone &>/dev/null; then
    echo "  Installing rclone for Google Drive sync..."
    curl -fsSL https://rclone.org/install.sh | sudo bash -s beta >/dev/null 2>&1
fi

# Check if rclone has a Google Drive remote configured
GDRIVE_REMOTE=""
if command -v rclone &>/dev/null; then
    # Look for any remote of type "drive"
    for remote in $(rclone listremotes 2>/dev/null); do
        rtype=$(rclone config show "${remote%:}" 2>/dev/null | grep "^type" | awk '{print $3}')
        if [ "$rtype" = "drive" ]; then
            GDRIVE_REMOTE="${remote%:}"
            break
        fi
    done

    if [ -z "$GDRIVE_REMOTE" ]; then
        echo "  No Google Drive remote found in rclone."
        echo "  Run 'rclone config' to add one (type: drive), then re-run this script."
        echo "  Or manually place datasets in data/aihub/ and data/cc3m_ko/"
    fi
fi

# Sync extra datasets from Google Drive
GDRIVE_DATA_PATH="data"  # path inside Google Drive (relative to root)

sync_from_drive() {
    local name="$1"
    local drive_path="$2"
    local local_path="$3"

    if [ -z "$GDRIVE_REMOTE" ]; then
        return
    fi

    # Check if remote path exists
    if rclone lsd "${GDRIVE_REMOTE}:${drive_path}" &>/dev/null; then
        if [ -d "$local_path" ]; then
            echo "  $name — already local, syncing updates..."
        else
            echo "  $name — downloading from Google Drive..."
        fi
        mkdir -p "$local_path"
        rclone sync "${GDRIVE_REMOTE}:${drive_path}" "$local_path" \
            --progress --transfers=8 --checkers=16
        echo "  $name — sync complete"
    else
        echo "  $name — not found on Drive (${GDRIVE_REMOTE}:${drive_path}), skipping"
    fi
}

sync_from_drive "COCO-Ko" "${GDRIVE_DATA_PATH}/coco_ko" "$PROJECT_DIR/data/coco_ko"
sync_from_drive "AIHub" "${GDRIVE_DATA_PATH}/aihub" "$PROJECT_DIR/data/aihub"
sync_from_drive "CC3M-Ko" "${GDRIVE_DATA_PATH}/cc3m_ko" "$PROJECT_DIR/data/cc3m_ko"

# Sync monitor DB + checkpoints from Google Drive (shared with Colab)
GDRIVE_PROJECT_PATH="vlm_quantization"

sync_file_from_drive() {
    local name="$1"
    local drive_file="$2"
    local local_file="$3"

    if [ -z "$GDRIVE_REMOTE" ]; then
        return
    fi

    if rclone ls "${GDRIVE_REMOTE}:${drive_file}" &>/dev/null; then
        echo "  $name — syncing from Google Drive..."
        mkdir -p "$(dirname "$local_file")"
        rclone copy "${GDRIVE_REMOTE}:${drive_file}" "$(dirname "$local_file")/"
        echo "  $name — done"
    else
        echo "  $name — not found on Drive, skipping"
    fi
}

sync_file_from_drive "Metrics DB" \
    "${GDRIVE_PROJECT_PATH}/monitor/metrics.db" \
    "$PROJECT_DIR/monitor/metrics.db"

sync_from_drive "Checkpoints" \
    "${GDRIVE_PROJECT_PATH}/checkpoints" \
    "$PROJECT_DIR/checkpoints"

# Auto-prepare JSONL for any datasets that have raw data but no JSONL

# COCO Korean (AIHub #261) — reuses existing COCO images
COCO_KO_DIR="$PROJECT_DIR/data/coco_ko"
COCO_KO_JSONL="$COCO_KO_DIR/coco_ko.jsonl"
if [ -d "$COCO_KO_DIR" ] && [ ! -f "$COCO_KO_JSONL" ]; then
    COCO_KO_JSON="$COCO_KO_DIR/MSCOCO_train_val_Korean.json"
    if [ -f "$COCO_KO_JSON" ]; then
        echo "  COCO-Ko: Preparing JSONL from Korean captions..."
        uv run python scripts/prepare_datasets.py coco-ko \
            --input "$COCO_KO_JSON" --output "$COCO_KO_DIR" || true
    fi
fi
if [ -f "$COCO_KO_JSONL" ]; then
    echo "  COCO-Ko: $(wc -l < "$COCO_KO_JSONL") entries ready"
fi

AIHUB_DIR="$PROJECT_DIR/data/aihub"
AIHUB_JSONL="$AIHUB_DIR/aihub_71454.jsonl"
if [ -d "$AIHUB_DIR" ] && [ ! -f "$AIHUB_JSONL" ]; then
    # Check if there are JSON annotation files
    if ls "$AIHUB_DIR"/*.json &>/dev/null || ls "$AIHUB_DIR"/**/*.json &>/dev/null 2>&1; then
        echo "  AIHub: Preparing JSONL from raw annotations..."
        uv run python scripts/prepare_datasets.py aihub \
            --input "$AIHUB_DIR" --output "$AIHUB_DIR" || true
    fi
fi
if [ -f "$AIHUB_JSONL" ]; then
    echo "  AIHub: $(wc -l < "$AIHUB_JSONL") entries ready"
fi

CC3M_DIR="$PROJECT_DIR/data/cc3m_ko"
CC3M_JSONL="$CC3M_DIR/cc3m_ko.jsonl"
if [ -d "$CC3M_DIR" ] && [ ! -f "$CC3M_JSONL" ]; then
    # Find TSV files
    EN_TSV="" KO_TSV=""
    for f in "$CC3M_DIR"/*.tsv; do
        [ -f "$f" ] || continue
        fname=$(basename "$f" | tr '[:upper:]' '[:lower:]')
        if echo "$fname" | grep -qE "ko|korean"; then
            KO_TSV="$f"
        else
            EN_TSV="$f"
        fi
    done

    if [ -n "$EN_TSV" ] && [ -n "$KO_TSV" ]; then
        echo "  CC3M-Ko: Preparing bilingual JSONL..."
        uv run python scripts/prepare_datasets.py cc3m-ko \
            --input "$EN_TSV" --ko-input "$KO_TSV" --output "$CC3M_DIR" || true
    elif [ -n "$EN_TSV" ]; then
        echo "  CC3M: Preparing English-only JSONL..."
        uv run python scripts/prepare_datasets.py cc3m \
            --input "$EN_TSV" --output "$CC3M_DIR" || true
        # Rename to match config expectation
        if [ -f "$CC3M_DIR/cc3m.jsonl" ] && [ ! -f "$CC3M_JSONL" ]; then
            mv "$CC3M_DIR/cc3m.jsonl" "$CC3M_JSONL"
        fi
    fi
fi
if [ -f "$CC3M_JSONL" ]; then
    echo "  CC3M-Ko: $(wc -l < "$CC3M_JSONL") entries ready"
fi

# ---------- 4. Build monitoring frontend ----------
echo ""
echo "[4/4] Building monitoring dashboard..."

if ! command -v node &>/dev/null; then
    echo "  Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -n bash - 2>/dev/null \
        && sudo -n apt-get install -y nodejs 2>/dev/null
fi

if command -v node &>/dev/null; then
    echo "  Node.js: $(node --version)"
    cd "$PROJECT_DIR/monitor/frontend"
    npm install --silent
    npm run build
    if [ -d "out" ] && [ -f "out/index.html" ]; then
        echo "  Frontend built -> monitor/frontend/out/"
    else
        echo "  WARNING: Frontend build did not produce out/index.html"
    fi
    cd "$PROJECT_DIR"
else
    echo "  WARNING: Could not install Node.js (needs sudo). Skipping dashboard build."
    echo "  Install manually: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -"
fi

# ---------- Done ----------
echo ""
echo "=============================="
echo "Setup complete!"
echo ""
echo "To start training:"
echo "  bash scripts/train_dgx_spark.sh"
echo ""
echo "Or manually:"
echo "  uv run python train.py --config configs/dgx_spark.yaml"
echo "=============================="
