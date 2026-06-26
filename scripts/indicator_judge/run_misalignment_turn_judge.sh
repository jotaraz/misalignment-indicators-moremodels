#!/bin/bash
#
# Run misalignment turn judge across behaviors.
#
# Uses Opus 4.6 to judge whether each assistant turn contains misaligned thinking.
# No indicator taxonomy — open-ended per-turn misalignment assessment.
#
# Output structure:
#   bloom/indicator_results/misalignment_turns/<behavior>/rollout_misalignment_turns.json
#
# Usage:
#   bash scripts/indicator_judge/run_misalignment_turn_judge.sh

set -e

# Load environment variables from .env file
ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please set ANTHROPIC_API_KEY environment variable"
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    exit 1
fi

# Resolve data_generation directory
DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Use bloom venv python (has anthropic)
PYTHON="${DATA_GEN_DIR}/bloom/.venv/bin/python"

# ============================================================
# Configuration
# ============================================================

BEHAVIORS=(
    # "sabotage_backdoor|bloom/bloom-results/instructed-long-horizon-sabotage-backdoor/rollout.json"
    # "sabotage_bug|bloom/bloom-results/instructed-long-horizon-sabotage-bug/rollout.json"
    # "sycophancy|bloom/bloom-results/sycophancy/rollout.json"
    # "strategic_deception|bloom/bloom-results/strategic-deception_glmflash/rollout.json"
    "self_preferential_bias|bloom/bloom-results/self-preferential-bias_glm_4_7_flash/rollout.json"
    "self_promotion|bloom/bloom-results/self-promotion_glm_4_7_flash/rollout.json"
)

MODEL="claude-opus-4-6"
THINKING_BUDGET=20000
MAX_CONCURRENT=5

# Output goes next to each rollout.json file

# ============================================================
# Main loop
# ============================================================

TOTAL=${#BEHAVIORS[@]}
CURRENT=0

echo "============================================================"
echo "Misalignment Turn Judge"
echo "============================================================"
echo "Behaviors:       $TOTAL"
echo "Model:           $MODEL"
echo "Thinking budget: $THINKING_BUDGET"
echo "Concurrency:     $MAX_CONCURRENT"
echo "Output base:     $OUTPUT_BASE"
echo "============================================================"
echo ""

for behavior_config in "${BEHAVIORS[@]}"; do
    BEHAVIOR_NAME="${behavior_config%%|*}"
    ROLLOUT_FILE="${behavior_config##*|}"
    CURRENT=$((CURRENT + 1))

    if [ ! -f "$ROLLOUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] WARNING: Rollout file not found: $ROLLOUT_FILE — skipping"
        continue
    fi

    OUTPUT_DIR="$(dirname "$ROLLOUT_FILE")"
    BASENAME=$(basename "$ROLLOUT_FILE" .json)
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_misalignment_turns.json"

    # Skip if output already exists (resume support)
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (already done): $BEHAVIOR_NAME"
        continue
    fi

    echo "************************************************************"
    echo "[$CURRENT/$TOTAL] $BEHAVIOR_NAME"
    echo "************************************************************"
    echo "  Rollout: $ROLLOUT_FILE"
    echo "  Output:  $OUTPUT_FILE"
    echo ""

    PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" $PYTHON -m black_box_ind_judge "$ROLLOUT_FILE" \
        --misalignment-turns \
        --model "$MODEL" \
        --thinking-budget "$THINKING_BUDGET" \
        -c "$MAX_CONCURRENT" \
        -o "$OUTPUT_FILE"

    echo ""
    echo "[$CURRENT/$TOTAL] Done: $BEHAVIOR_NAME"
    echo ""
done

echo "============================================================"
echo "All $TOTAL behaviors completed!"
echo "============================================================"
echo ""
echo "Results:"
for behavior_config in "${BEHAVIORS[@]}"; do
    ROLLOUT_FILE="${behavior_config##*|}"
    BASENAME=$(basename "$ROLLOUT_FILE" .json)
    echo "  $(dirname "$ROLLOUT_FILE")/${BASENAME}_misalignment_turns.json"
done
