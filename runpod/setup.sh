#!/usr/bin/env bash
# setup.sh — One-time setup on a fresh RunPod A100 instance.
#
# Usage:
#   1. SSH into your RunPod pod or use the web terminal
#   2. Clone/upload this repo to /workspace/CloudModelTrain
#   3. Run: bash runpod/setup.sh
#
# Assumes: RunPod PyTorch template (CUDA + Python pre-installed)

set -euo pipefail

echo "=========================================="
echo "  CloudModelTrain — RunPod Setup"
echo "=========================================="

WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/CloudModelTrain"

cd "$PROJECT_DIR"

# --- System packages (if needed) ---
echo "[1/5] Checking system dependencies..."
apt-get update -qq && apt-get install -y -qq git-lfs > /dev/null 2>&1 || true

# --- Python dependencies ---
echo "[2/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# --- Optional: Flash Attention 2 (significant speedup on A100) ---
echo "[3/5] Installing Flash Attention 2 (optional, may take a few minutes)..."
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || {
    echo "  [WARN] flash-attn failed to install. Training will work without it."
    echo "  To enable later: pip install flash-attn --no-build-isolation"
}

# --- Create scratch directories on NVMe ---
echo "[4/6] Setting up NVMe scratch directories..."
# RunPod mounts fast storage at /workspace
mkdir -p "${PROJECT_DIR}/data_cache"
mkdir -p "${PROJECT_DIR}/datasets"
mkdir -p "${PROJECT_DIR}/outputs"

# --- Build datasets from HuggingFace ---
echo "[5/6] Building training datasets from HuggingFace..."
python scripts/build_datasets.py
echo "  Datasets built:"
for f in "${PROJECT_DIR}"/datasets/*/train.jsonl; do
    agent=$(basename "$(dirname "$f")")
    count=$(wc -l < "$f")
    echo "    $agent: $count records"
done

# --- Verify GPU ---
echo "[6/6] Verifying GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'BF16 support:    {torch.cuda.is_bf16_supported()}')
    print(f'TF32 support:    True (A100)')
else:
    echo 'WARNING: No GPU detected!'
    exit 1
"

echo ""
echo "=========================================="
echo "  Setup complete! Datasets ready."
echo ""
echo "  Train a single agent:"
echo "    bash runpod/train_agent.sh code_writer"
echo ""
echo "  Train all agents:"
echo "    bash runpod/train_all.sh --flash_attn"
echo ""
echo "  Rebuild datasets (to customize size):"
echo "    python scripts/build_datasets.py --max_samples 5000"
echo "=========================================="
