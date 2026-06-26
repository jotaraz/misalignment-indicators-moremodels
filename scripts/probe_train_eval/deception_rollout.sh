#!/bin/bash
# Script to generate rollouts from the deception-detection roleplaying dataset
# Usage: ./generate_rollouts.sh

# Configuration - EDIT THESE
DATASET_ID="roleplaying__plain"  # Options: insider_trading__upscale, insider_trading__onpolicy, insider_trading__prewritten, insider_trading_doubledown__upscale, insider_trading_doubledown__onpolicy, roleplaying__plain, roleplaying__plain_short, sandbagging_v2__wmdp_mmlu
NUM_ROLLOUTS=1  # Number of rollouts to generate
USE_API=False  # Set to True for Together API, False for local GPU (must be capitalized for Python Fire)
MODEL_NAME="glm-9b-flash"  # Options: llama-1b, llama-3b, llama-8b, llama-70b, llama-70b-3.3, gemma-2b, gemma-7b, gemma-9b, mistral-7b, qwen-14b, qwen-32b, glm-9b-flash
GRADE_ROLLOUTS=False #True  # Must be capitalized for Python Fire
GRADER_API="anthropic"  # Options: openai, anthropic
MAX_NEW_TOKENS=4096  # Maximum tokens to generate per rollout (local generation only)
MAX_PROMPTS="10"  # Leave empty for all, or set to a number like 10 for testing

# Setup paths
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
DECEPTION_DIR="$PROJECT_DIR/deception-detection"
LOG_DIR="$PROJECT_DIR/logs"
timestamp=$(date +%Y%m%d_%H%M%S)

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Use venv Python directly (more reliable than source activate in piped subshells)
VENV_DIR="$DECEPTION_DIR/.venv"
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
    echo "Using venv Python: $PYTHON"
else
    PYTHON="python"
    echo "Warning: Virtual environment not found at $VENV_DIR, using system python"
fi

# Set environment variables
export HF_HOME=/workspace-vast/pretrained_ckpts
export $(grep -v '^#' "$PROJECT_DIR/.env" 2>/dev/null | xargs) 2>/dev/null || true

# Change to deception-detection directory
cd "$DECEPTION_DIR" || exit 1

echo "=========================================="
echo "Generating Rollouts"
echo "=========================================="
echo "Dataset: $DATASET_ID"
echo "Model: $MODEL_NAME"
echo "Number of rollouts: $NUM_ROLLOUTS"
echo "Use API: $USE_API"
echo "Grade rollouts: $GRADE_ROLLOUTS"
echo "Grader API: $GRADER_API"
echo "Max new tokens: $MAX_NEW_TOKENS"
if [ -n "$MAX_PROMPTS" ]; then
    echo "Max prompts: $MAX_PROMPTS"
fi
echo "=========================================="

# Build command
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

# Setup logging
LOG_FILE="$LOG_DIR/rollout_${MODEL_NAME}_${timestamp}.log"
echo "Log file: $LOG_FILE"
echo ""

# Run the command with logging
echo "Running command:"
echo "$CMD"
echo ""

# Execute command and capture output to both console and log file
{
    echo "=========================================="
    echo "Rollout Generation Log"
    echo "Started at: $(date)"
    echo "Model: $MODEL_NAME"
    echo "Dataset: $DATASET_ID"
    echo "=========================================="
    echo ""

    eval $CMD 2>&1

    EXIT_CODE=$?

    echo ""
    echo "=========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        ROLLOUT_FILE="$DECEPTION_DIR/data/rollouts/${DATASET_ID}__${MODEL_NAME}.json"
        echo "Rollout generation complete!"
        echo "Output saved to: $ROLLOUT_FILE"

        # Convert to sycophancy-eval format for black_box_ind_judge
        CONVERTED_FILE="$DECEPTION_DIR/rollouts/${MODEL_NAME}/${DATASET_ID}_rollout.json"
        echo ""
        echo "Converting to sycophancy-eval format..."
        $PYTHON "$PROJECT_DIR/scripts/convert_deception_to_rollout.py" \
            --input "$ROLLOUT_FILE" \
            --output "$CONVERTED_FILE"
        echo "Converted rollout saved to: $CONVERTED_FILE"
    else
        echo "Rollout generation failed with exit code: $EXIT_CODE"
    fi
    echo "Finished at: $(date)"
    echo "=========================================="

    exit $EXIT_CODE
} 2>&1 | tee "$LOG_FILE"
