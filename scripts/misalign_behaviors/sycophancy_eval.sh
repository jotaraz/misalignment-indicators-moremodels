#!/bin/bash
# Generate sycophancy-eval rollouts and convert to bloom rollout format.
#
# Steps:
#   1. Run inference via OpenRouter (glm-4.7-flash)
#   2. Convert JSONL outputs to rollout.json format
#
# Output:
#   sycophancy-eval/rollouts/<MODEL>/{answer,are_you_sure,feedback}_rollout.json
#
# Usage:
#   bash scripts/misalign_behaviors/sycophancy_eval.sh
#   bash scripts/misalign_behaviors/sycophancy_eval.sh --dataset answer --max-prompts 5

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="glm-4.7-flash"
DATASET="all"                # Options: answer, are_you_sure, feedback, all
MAX_CONCURRENT=50
MAX_PROMPTS="50"             # Leave empty for all, or set to a number for testing

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SYCOPHANCY_DIR="$PROJECT_DIR/ood_misalignment_eval/sycophancy-eval"

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment variables from $PROJECT_DIR/.env"
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
else
    echo "Warning: .env file not found at $PROJECT_DIR/.env"
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY is not set"
    exit 1
fi

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2";        shift 2 ;;
        --max-prompts) MAX_PROMPTS="$2";    shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --model)       MODEL="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build inference command ──────────────────────────────────────────────────
CMD="python $SYCOPHANCY_DIR/run_inference.py --dataset $DATASET --max-concurrent $MAX_CONCURRENT"
if [ -n "$MAX_PROMPTS" ]; then
    CMD="$CMD --max-prompts $MAX_PROMPTS"
fi

echo "=========================================="
echo "Sycophancy-Eval Rollout Generation"
echo "=========================================="
echo "Model:          $MODEL"
echo "Dataset:        $DATASET"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max prompts:    ${MAX_PROMPTS:-all}"
echo "=========================================="
echo ""

# ── Step 1: Run inference ────────────────────────────────────────────────────
echo "[Step 1/2] Running inference..."
echo "Command: $CMD"
eval $CMD

# ── Step 2: Convert to rollout format ────────────────────────────────────────
echo ""
echo "[Step 2/2] Converting to rollout format..."
python "$SYCOPHANCY_DIR/convert_to_rollout.py" --model "$MODEL" --dataset "$DATASET"

# ── Report outputs ───────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Done! Output rollouts:"
echo "=========================================="
if [ "$DATASET" = "all" ]; then
    for ds in answer are_you_sure feedback; do
        ROLLOUT="$SYCOPHANCY_DIR/rollouts/$MODEL/${ds}_rollout.json"
        if [ -f "$ROLLOUT" ]; then
            echo "  $ROLLOUT"
        fi
    done
else
    ROLLOUT="$SYCOPHANCY_DIR/rollouts/$MODEL/${DATASET}_rollout.json"
    if [ -f "$ROLLOUT" ]; then
        echo "  $ROLLOUT"
    fi
fi
