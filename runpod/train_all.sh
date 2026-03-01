#!/usr/bin/env bash
# train_all.sh — Train all agents sequentially on RunPod.
#
# Usage:
#   bash runpod/train_all.sh
#   bash runpod/train_all.sh --flash_attn
#
# Trains each agent one after another on the same GPU.
# Pass extra flags that apply to ALL agents.

set -euo pipefail

EXTRA_FLAGS=("$@")

AGENTS=(
    code_writer
    test_generator
    static_reviewer
    security_auditor
    performance_optimizer
    docs_generator
)

echo "============================================"
echo "  Training all ${#AGENTS[@]} agents"
echo "  Extra flags: ${EXTRA_FLAGS[*]:-none}"
echo "============================================"
echo ""

FAILED=()

for agent in "${AGENTS[@]}"; do
    echo ">>> Starting: $agent"
    if bash runpod/train_agent.sh "$agent" "" "${EXTRA_FLAGS[@]}"; then
        echo ">>> Completed: $agent"
    else
        echo ">>> FAILED: $agent"
        FAILED+=("$agent")
    fi
    echo ""
done

echo "============================================"
echo "  All done."
echo "  Succeeded: $(( ${#AGENTS[@]} - ${#FAILED[@]} )) / ${#AGENTS[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo "============================================"
