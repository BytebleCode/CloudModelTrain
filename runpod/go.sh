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

# ---- Redirect HF cache to volume (not container disk) ----
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
mkdir -p "$HF_HOME"

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
echo "  Project dir: $PROJECT_DIR"
echo "  HF cache:    $HF_HOME"
echo "  Agent:       ${AGENT:-ALL}"
echo ""

# ---- Step 1: Install deps (skip if already done) ----
SETUP_MARKER="${PROJECT_DIR}/.setup_done_v6"
if [[ ! -f "$SETUP_MARKER" ]]; then
    echo "[1/4] Installing dependencies..."

    echo "  >> Upgrading pip..."
    pip install --upgrade pip 2>&1 | tail -1

    echo "  >> Installing torch..."
    pip install --upgrade torch torchvision torchaudio 2>&1 | tail -3

    echo "  >> Installing requirements.txt..."
    pip install -r requirements.txt 2>&1 | tail -5

    echo "  >> Upgrading transformers, accelerate, peft, bitsandbytes..."
    pip install --upgrade transformers accelerate peft bitsandbytes 2>&1 | tail -5

    echo ""
    echo "  Installed versions:"
    python -c "
import torch, transformers, peft, bitsandbytes, accelerate
print(f'    torch:          {torch.__version__}')
print(f'    transformers:   {transformers.__version__}')
print(f'    accelerate:     {accelerate.__version__}')
print(f'    peft:           {peft.__version__}')
print(f'    bitsandbytes:   {bitsandbytes.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
print(f'    CUDA version:   {torch.version.cuda}')
"
    echo ""

    echo "[2/4] Skipping Flash Attention (not pre-installed)"

    touch "$SETUP_MARKER"
    echo ""
    echo "  Dependencies cached (won't reinstall on next run)"
    echo "  Delete $SETUP_MARKER to force reinstall"
else
    echo "[1/4] Dependencies already installed (cached)"
    echo "[2/4] Skipping flash-attn (cached)"
    echo ""
    echo "  Cached versions:"
    python -c "
import torch, transformers, peft, bitsandbytes, accelerate
print(f'    torch:          {torch.__version__}')
print(f'    transformers:   {transformers.__version__}')
print(f'    accelerate:     {accelerate.__version__}')
print(f'    peft:           {peft.__version__}')
print(f'    bitsandbytes:   {bitsandbytes.__version__}')
" 2>/dev/null || true
    python -c "import flash_attn; print(f'    flash-attn:     {flash_attn.__version__}')" 2>/dev/null || echo "    flash-attn:     NOT installed"
fi
echo ""

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
DATASET_COUNT=0
for f in "${PROJECT_DIR}"/datasets/*/train.jsonl; do
    agent=$(basename "$(dirname "$f")")
    count=$(wc -l < "$f")
    size=$(du -h "$f" | cut -f1)
    printf "    %-25s %6s records  (%s)\n" "$agent" "$count" "$size"
    DATASET_COUNT=$((DATASET_COUNT + 1))
done
shopt -u nullglob
if [[ $DATASET_COUNT -eq 0 ]]; then
    echo "    [WARN] No datasets found in ${PROJECT_DIR}/datasets/"
fi
echo ""

# ---- Step 3: Verify GPU ----
echo "[4/4] Verifying GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU detected!'
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    vram = props.total_memory / 1e9
    print(f'  GPU {i}: {name} ({vram:.1f} GB VRAM)')
print(f'  BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'  TF32 supported: {torch.backends.cuda.matmul.allow_tf32}')
"

# Show disk space
echo ""
echo "  Disk space:"
df -h /workspace 2>/dev/null | tail -1 | awk '{printf "    /workspace: %s used / %s total (%s free)\n", $3, $2, $4}'
df -h / 2>/dev/null | tail -1 | awk '{printf "    /root:      %s used / %s total (%s free)\n", $3, $2, $4}'
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
