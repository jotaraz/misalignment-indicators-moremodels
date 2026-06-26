#!/bin/bash
# Run MASK evaluation via OpenRouter with reasoning capture.
#
# Uses ood_misalignment_eval/mask/run_inference.py to generate responses,
# capturing both content and reasoning (thinking) from the model. Outputs
# JSONL + rollout.json in bloom-compatible format.
#
# Output:
#   ood_misalignment_eval/mask/rollouts/<model>/mask_<split>_rollout.json
#   ood_misalignment_eval/mask/rollouts/<model>/mask_<split>.jsonl
#
# Usage:
#   bash scripts/misalign_behaviors/mask_eval.sh
#   bash scripts/misalign_behaviors/mask_eval.sh --test
#   bash scripts/misalign_behaviors/mask_eval.sh --model z-ai/glm-4.7-flash --splits known_facts provided_facts
#   bash scripts/misalign_behaviors/mask_eval.sh --max-prompts 5

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="z-ai/glm-4.7-flash"
SPLITS="known_facts provided_facts"
MAX_CONCURRENT=50
MAX_TOKENS=4096
TEMPERATURE=1.0
TEST_FLAG=""
MAX_PROMPTS_FLAG=""

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MASK_DIR="$PROJECT_DIR/ood_misalignment_eval/mask"
PYTHON="$PROJECT_DIR/bloom/.venv/bin/python"

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --splits)         SPLITS="$2";         shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2";     shift 2 ;;
        --temperature)    TEMPERATURE="$2";    shift 2 ;;
        --max-prompts)    MAX_PROMPTS_FLAG="--max-prompts $2"; shift 2 ;;
        --test)           TEST_FLAG="--test";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "MASK Evaluation"
echo "=========================================="
echo "Model:          $MODEL"
echo "Splits:         $SPLITS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max tokens:     $MAX_TOKENS"
echo "Temperature:    $TEMPERATURE"
echo "Test mode:      ${TEST_FLAG:-no}"
echo "=========================================="
echo ""

cd "$MASK_DIR"

$PYTHON run_inference.py \
    --model "$MODEL" \
    --splits $SPLITS \
    --max-concurrent "$MAX_CONCURRENT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    $TEST_FLAG \
    $MAX_PROMPTS_FLAG

echo ""
echo "Done!"
