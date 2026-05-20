#!/bin/bash
# One-shot environment setup with uv for Pix2Pix colorization.
# Usage: bash setup_env.sh

set -e

# 1. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 2. Create venv
echo "=== Creating virtual environment ==="
uv venv --python 3.11

# 3. Install all deps with compatible versions
echo "=== Installing dependencies ==="
uv pip install \
    'numpy>=1.26.4,<2' \
    'scipy>=1.15' \
    'torch>=2.0' \
    'torchvision>=0.15' \
    'scikit-image>=0.24' \
    'tqdm' \
    'Pillow' \
    'dominate' \
    'wandb'

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Then run:       bash train_coco.sh stage1"
