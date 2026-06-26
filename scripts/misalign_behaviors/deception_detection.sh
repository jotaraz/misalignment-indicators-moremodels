#!/bin/bash
# Generate deception-detection rollouts and convert to bloom format.
#
# Steps:
#   1. Generate rollouts using deception-detection framework
#   2. Convert to bloom format (rollout.json + judgment.json)
#
# Output:
#   deception-detection/data/rollouts/<DATASET>__<MODEL>_bloom/rollout.json
#   deception-detection/data/rollouts/<DATASET>__<MODEL>_bloom/judgment.json
#
# Usage:
#   bash scripts/misalign_behaviors/deception_detection.sh
#   bash scripts/misalign_behaviors/deception_detection.sh --dataset roleplaying__plain --model glm-9b-flash

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ID="roleplaying__plain"   # Options: insider_trading__upscale, insider_trading__onpolicy,
                                  #   insider_trading__prewritten, insider_trading_doubledown__upscale,
                                  #   insider_trading_doubledown__onpolicy, roleplaying__plain,
                                  #   roleplaying__plain_short, sandbagging_v2__wmdp_mmlu
MODEL_NAME="glm-9b-flash"        # Options: llama-1b, llama-3b, llama-8b, llama-70b, llama-70b-3.3,
                                  #   gemma-2b, gemma-7b, gemma-9b, mistral-7b, qwen-14b, qwen-32b, glm-9b-flash
NUM_ROLLOUTS=1
USE_API=False                     # True for Together API, False for local GPU (capitalized for Python Fire)
GRADE_ROLLOUTS=False              # Capitalized for Python Fire
GRADER_API="anthropic"            # Options: openai, anthropic
MAX_NEW_TOKENS=4096
MAX_PROMPTS="100"                 # Leave empty for all, or set to a number for testing

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DECEPTION_DIR="$PROJECT_DIR/ood_misalignment_eval/deception-detection"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET_ID="$2";     shift 2 ;;
        --model)       MODEL_NAME="$2";     shift 2 ;;
        --num-rollouts) NUM_ROLLOUTS="$2";  shift 2 ;;
        --use-api)     USE_API="$2";        shift 2 ;;
        --grade)       GRADE_ROLLOUTS="$2"; shift 2 ;;
        --max-prompts) MAX_PROMPTS="$2";    shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment setup ────────────────────────────────────────────────────────
export HF_HOME=${HF_HOME:-/workspace-vast/pretrained_ckpts}
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ── Find Python ──────────────────────────────────────────────────────────────
VENV_DIR="$DECEPTION_DIR/.venv"
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
    echo "Using venv Python: $PYTHON"
else
    PYTHON="python"
    echo "Warning: Virtual environment not found at $VENV_DIR, using system python"
fi

echo "=========================================="
echo "Deception-Detection Rollout Generation"
echo "=========================================="
echo "Dataset:        $DATASET_ID"
echo "Model:          $MODEL_NAME"
echo "Num rollouts:   $NUM_ROLLOUTS"
echo "Use API:        $USE_API"
echo "Grade rollouts: $GRADE_ROLLOUTS"
echo "Max prompts:    ${MAX_PROMPTS:-all}"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "=========================================="
echo ""

# ── Step 1: Generate rollouts ────────────────────────────────────────────────
echo "[Step 1/2] Generating rollouts..."
cd "$DECEPTION_DIR"

CMD="$PYTHON deception_detection/scripts/generate_rollouts.py \
    --dataset_partial_id=\"$DATASET_ID\" \
    --num=$NUM_ROLLOUTS \
    --use_api=$USE_API \
    --model_name=\"$MODEL_NAME\" \
    --grade_rollouts=$GRADE_ROLLOUTS \
    --grader_api=\"$GRADER_API\" \
    --max_new_tokens=$MAX_NEW_TOKENS"

if [ -n "$MAX_PROMPTS" ]; then
    CMD="$CMD --max_prompts=$MAX_PROMPTS"
fi

echo "Command: $CMD"
eval $CMD
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Rollout generation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# ── Step 2: Convert to bloom format ─────────────────────────────────────────
ROLLOUT_FILE="$DECEPTION_DIR/data/rollouts/${DATASET_ID}__${MODEL_NAME}.json"

if [ ! -f "$ROLLOUT_FILE" ]; then
    echo "Error: Expected rollout file not found: $ROLLOUT_FILE"
    exit 1
fi

echo ""
echo "[Step 2/2] Converting to bloom format..."
cd "$PROJECT_DIR"
python "$DECEPTION_DIR/convert_to_bloom.py" "$ROLLOUT_FILE"

# ── Report outputs ───────────────────────────────────────────────────────────
OUTPUT_DIR="$DECEPTION_DIR/data/rollouts/${DATASET_ID}__${MODEL_NAME}_bloom"
echo ""
echo "=========================================="
echo "Done! Output files:"
echo "=========================================="
echo "  $OUTPUT_DIR/rollout.json"
echo "  $OUTPUT_DIR/judgment.json"
