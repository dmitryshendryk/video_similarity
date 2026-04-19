#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.12"

echo "=== S2VS Video Deduplication — Setup ==="

# --- Python environment ---
echo ""
echo "[1/4] Checking Python $PYTHON_VERSION..."

if command -v python$PYTHON_VERSION &>/dev/null; then
    PYTHON_BIN="python$PYTHON_VERSION"
elif command -v python3 &>/dev/null && python3 --version | grep -q "3\.12"; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Python $PYTHON_VERSION not found."
    echo "Install with: pyenv install 3.12.11 && pyenv local 3.12.11"
    exit 1
fi

echo "Using: $($PYTHON_BIN --version)"

echo ""
echo "[2/4] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, skipping"
fi

source .venv/bin/activate

echo ""
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
pip install -r requirements.txt -q
pip install qdrant-client optimum-quanto -q
echo "Python packages installed"

# --- Frontend ---
echo ""
echo "[4/4] Installing frontend dependencies..."
if ! command -v bun &>/dev/null; then
    echo "WARNING: bun not found. Install from https://bun.sh"
    echo "Falling back to npm..."
    cd web-ui && npm install && cd ..
else
    cd web-ui && bun install && cd ..
fi

echo ""
echo "=== Setup complete ==="
echo "Run ./run.sh to start the application"
