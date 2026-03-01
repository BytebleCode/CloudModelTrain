#!/usr/bin/env bash
# go.sh — One-command RunPod launcher: setup + build datasets + train.
#
# Usage (from RunPod web terminal or SSH):
#   cd /workspace/CloudModelTrain
#   bash runpod/go.sh                              # train all agents
#   bash runpod/go.sh code_writer                   # train one agent
#   bash runpod/go.sh code_writer --flash_attn      # with flash attention
#   bash runpod/go.sh --all --flash_attn            # all agents + flash attn
#
# This is the ONLY script you need to run on a fresh RunPod pod.
# It handles: deps, flash-attn, dataset download, GPU check, and training.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- Parse args ----
AGENT=""
TRAIN_ALL=false
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            TRAIN_ALL=true
            shift
            ;;
        --flash_attn|--lr|--epochs|--batch_size|--max_seq_length|--resume|--model)
            EXTRA_FLAGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                EXTRA_FLAGS+=("$2")
                shift
            fi
            shift
            ;;
        *)
            if [[ -z "$AGENT" ]]; then
                AGENT="$1"
            else
                EXTRA_FLAGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Default: train all
if [[ -z "$AGENT" && "$TRAIN_ALL" == "false" ]]; then
    TRAIN_ALL=true
fi

echo ""
echo "============================================"
echo "  CloudModelTrain — RunPod One-Click Launch"
echo "============================================"
echo ""

# ---- Step 1: Install deps (skip if already done) ----
SETUP_MARKER="${PROJECT_DIR}/.setup_done"
if [[ ! -f "$SETUP_MARKER" ]]; then
    echo "[1/4] Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt

    echo "[2/4] Installing Flash Attention 2..."
    pip install --quiet flash-attn --no-build-isolation 2>/dev/null || {
        echo "  [WARN] flash-attn install failed (training still works without it)"
    }

    touch "$SETUP_MARKER"
    echo "  Dependencies cached (won't reinstall on next run)"
else
    echo "[1/4] Dependencies already installed (cached)"
    echo "[2/4] Skipping flash-attn (cached)"
fi

# ---- Step 2: Build datasets (skip if already built) ----
DATASETS_MARKER="${PROJECT_DIR}/.datasets_done"
if [[ ! -f "$DATASETS_MARKER" ]]; then
    echo "[3/4] Downloading + building datasets from HuggingFace..."
    mkdir -p datasets
    python scripts/build_datasets.py
    touch "$DATASETS_MARKER"
else
    echo "[3/4] Datasets already built (cached)"
fi

# Show dataset stats
echo ""
echo "  Datasets:"
shopt -s nullglob
for f in "${PROJECT_DIR}"/datasets/*/train.jsonl; do
    agent=$(basename "$(dirname "$f")")
    count=$(wc -l < "$f")
    printf "    %-25s %s records\n" "$agent" "$count"
done
shopt -u nullglob
echo ""

# ---- Step 3: Verify GPU ----
echo "[4/4] Verifying GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU detected!'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {name} ({vram:.0f} GB)')
print(f'  BF16: {torch.cuda.is_bf16_supported()}')
"
echo ""

# ---- Step 4: Train ----
echo "============================================"
if [[ "$TRAIN_ALL" == "true" ]]; then
    echo "  Training ALL agents"
    [[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && echo "  Extra flags: ${EXTRA_FLAGS[*]}"
    echo "============================================"
    echo ""
    bash runpod/train_all.sh "${EXTRA_FLAGS[@]}"
else
    echo "  Training: $AGENT"
    [[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && echo "  Extra flags: ${EXTRA_FLAGS[*]}"
    echo "============================================"
    echo ""
    bash runpod/train_agent.sh "$AGENT" "" "${EXTRA_FLAGS[@]}"
fi
