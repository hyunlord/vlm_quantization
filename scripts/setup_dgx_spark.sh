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
#   1. Creates a Python virtual environment
#   2. Installs PyTorch + dependencies
#   3. Downloads COCO dataset + Karpathy split
#   4. Builds the monitoring dashboard frontend
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=============================="
echo "DGX Spark Setup"
echo "Project: $PROJECT_DIR"
echo "=============================="

# ---------- 1. Python environment ----------
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo ""
    echo "[1/4] Virtual environment already exists, skipping..."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

echo "  Installing PyTorch (CUDA)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

echo "  Installing project dependencies..."
pip install -r requirements.txt -q

echo "  Python: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

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

# ---------- 3. Extra datasets (instructions only) ----------
echo ""
echo "[3/4] Extra datasets (optional):"
echo "  To add AIHub #71454:"
echo "    1. Place data in: $PROJECT_DIR/data/aihub/"
echo "    2. Run: python scripts/prepare_datasets.py aihub --input data/aihub --output data/aihub"
echo ""
echo "  To add CC3M-Ko:"
echo "    1. Place data in: $PROJECT_DIR/data/cc3m_ko/"
echo "    2. Run: python scripts/prepare_datasets.py cc3m-ko --input <en.tsv> --ko-input <ko.tsv> --output data/cc3m_ko"

# ---------- 4. Build monitoring frontend ----------
echo ""
echo "[4/4] Building monitoring dashboard..."

if command -v node &>/dev/null; then
    echo "  Node.js: $(node --version)"
    cd "$PROJECT_DIR/monitor/frontend"
    npm install --silent 2>/dev/null
    npm run build 2>/dev/null
    echo "  Frontend built -> monitor/frontend/out/"
    cd "$PROJECT_DIR"
else
    echo "  WARNING: Node.js not found. Install Node.js 20+ to build the dashboard."
    echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -"
    echo "    sudo apt-get install -y nodejs"
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
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=. python train.py --config configs/dgx_spark.yaml"
echo "=============================="
