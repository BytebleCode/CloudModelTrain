#!/usr/bin/env bash
# train_agent.sh — Train a specific agent on RunPod.
#
# Usage:
#   bash runpod/train_agent.sh code_writer
#   bash runpod/train_agent.sh security_auditor my_run_v2
#   bash runpod/train_agent.sh code_writer "" --flash_attn --lr 1e-4
#   bash runpod/train_agent.sh code_writer "" --gpu h200_sxm
#
# This script:
#   1. Verifies GPU is available
#   2. Runs training with accelerate (bf16 mixed precision)
#   3. Logs output to file + stdout
#
# Arguments:
#   $1 = agent name (required)
#   $2 = run name (optional, pass "" to skip)
#   $3+ = extra flags forwarded to train.py

set -euo pipefail

AGENT="${1:?Usage: train_agent.sh <agent_name> [run_name] [extra_flags...]}"
RUN_NAME="${2:-}"
shift 2 2>/dev/null || shift 1
EXTRA_FLAGS=("$@")

PROJECT_DIR="/workspace/CloudModelTrain"
cd "$PROJECT_DIR"

# --- Verify GPU ---
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'" || {
    echo "ERROR: No GPU detected. Are you on a GPU pod?"
    exit 1
}

# --- Build timestamp for log file ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/outputs/${AGENT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# --- Build command ---
CMD=(
    accelerate launch
    --mixed_precision bf16
    --num_processes 1
    train.py
    --agent "$AGENT"
)

if [[ -n "$RUN_NAME" ]]; then
    CMD+=(--run_name "$RUN_NAME")
fi

CMD+=("${EXTRA_FLAGS[@]}")

echo "============================================"
echo "  Agent:     $AGENT"
echo "  Run:       ${RUN_NAME:-<auto>}"
echo "  Log:       $LOG_FILE"
echo "  Command:   ${CMD[*]}"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# --- Run training (tee to log + stdout) ---
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "  Training finished."
echo "  Log: $LOG_FILE"
echo "  Outputs: outputs/$AGENT/"
echo "============================================"
