#!/usr/bin/env bash
# launch.sh — Wrapper for single-GPU QLoRA training with accelerate.
#
# Usage:
#   bash scripts/launch.sh code_writer
#   bash scripts/launch.sh security_auditor sec_audit_v2
#   bash scripts/launch.sh test_generator "" --flash_attn
#
# Arguments:
#   $1 = agent name (required)
#   $2 = run name (optional, pass "" to skip)
#   $3+ = extra flags forwarded to train.py

set -euo pipefail

AGENT="${1:?Usage: launch.sh <agent_name> [run_name] [extra_flags...]}"
RUN_NAME="${2:-}"
shift 2 2>/dev/null || shift 1

EXTRA_FLAGS=("$@")

# Build command
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
echo "Agent:    $AGENT"
echo "Run:      ${RUN_NAME:-<auto-timestamp>}"
echo "Command:  ${CMD[*]}"
echo "============================================"

exec "${CMD[@]}"
