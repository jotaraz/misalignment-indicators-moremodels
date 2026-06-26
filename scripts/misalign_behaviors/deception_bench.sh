#!/bin/bash
# Run DeceptionBench inference and convert to bloom rollout format.
#
# Steps:
#   1. Run inference via OpenRouter (with optional deception judging)
#   2. Convert JSONL outputs to bloom format (rollout.json + judgment.json)
#
# Output:
#   <OUTPUT_DIR>/rollout.json + judgment.json
#
# Usage:
#   bash scripts/misalign_behaviors/deception_bench.sh
#   bash scripts/misalign_behaviors/deception_bench.sh --max-prompts 5
#   bash scripts/misalign_behaviors/deception_bench.sh --type Sycophantic_Misrepresentation --judge
#   bash scripts/misalign_behaviors/deception_bench.sh --skip-inference --model glm-4.7-flash

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="z-ai/glm-4.7-flash"
MODEL_NAME=""                 # Display name (auto-derived from MODEL if empty)
DECEPTION_TYPE=""             # Options: Sycophantic_Misrepresentation, Alignment_Faking, Sandbagging (empty = all)
MAX_CONCURRENT=50
MAX_PROMPTS=""                # Leave empty for all, or set to a number for testing
JUDGE=false                   # Run deception judge on results
JUDGE_MODEL="claude-sonnet-4-5-20250929"
SKIP_INFERENCE=false          # Set to true to skip inference, just convert existing JSONL
OUTPUT_DIR=""                 # Auto-set if empty

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DB_DIR="$PROJECT_DIR/ood_misalignment_eval/deception-bench"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --model-name)     MODEL_NAME="$2";     shift 2 ;;
        --type)           DECEPTION_TYPE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --max-prompts)    MAX_PROMPTS="$2";    shift 2 ;;
        --judge)          JUDGE=true;          shift ;;
        --judge-model)    JUDGE_MODEL="$2";    shift 2 ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Derive display name ──────────────────────────────────────────────────────
if [ -z "$MODEL_NAME" ]; then
    # "z-ai/glm-4.7-flash" -> "glm-4.7-flash"
    MODEL_NAME="${MODEL##*/}"
fi

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment variables from $PROJECT_DIR/.env"
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
else
    echo "Warning: .env file not found at $PROJECT_DIR/.env"
fi

echo "=========================================="
echo "DeceptionBench Rollout Generation"
echo "=========================================="
echo "Model:          $MODEL_NAME ($MODEL)"
echo "Deception type: ${DECEPTION_TYPE:-all}"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max prompts:    ${MAX_PROMPTS:-all}"
echo "Judge:          $JUDGE"
echo "Skip inference: $SKIP_INFERENCE"
echo "=========================================="
echo ""

# ── Step 1: Run inference ────────────────────────────────────────────────────
if [ "$SKIP_INFERENCE" = false ]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "Error: OPENROUTER_API_KEY is not set"
        exit 1
    fi

    echo "[Step 1/2] Running inference..."
    CMD="python $DB_DIR/run_inference.py --model $MODEL --max-concurrent $MAX_CONCURRENT"
    if [ -n "$MODEL_NAME" ]; then
        CMD="$CMD --model-name $MODEL_NAME"
    fi
    if [ -n "$DECEPTION_TYPE" ]; then
        CMD="$CMD --type $DECEPTION_TYPE"
    fi
    if [ -n "$MAX_PROMPTS" ]; then
        CMD="$CMD --max-prompts $MAX_PROMPTS"
    fi
    if [ "$JUDGE" = true ]; then
        CMD="$CMD --judge --judge-model $JUDGE_MODEL"
    fi

    echo "Command: $CMD"
    eval $CMD
    echo ""
else
    echo "[Step 1/2] Skipping inference (using existing JSONL)..."
    echo ""
fi

# ── Step 2: Convert to bloom format ─────────────────────────────────────────
TYPE_SUFFIX=""
if [ -n "$DECEPTION_TYPE" ]; then
    TYPE_SUFFIX="_${DECEPTION_TYPE}"
fi

JSONL_PATH="$DB_DIR/rollouts/$MODEL_NAME/deception_bench${TYPE_SUFFIX}.jsonl"

if [ ! -f "$JSONL_PATH" ]; then
    echo "Error: Expected JSONL file not found: $JSONL_PATH"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DB_DIR/rollouts/$MODEL_NAME/bloom${TYPE_SUFFIX}"
fi

echo "[Step 2/2] Converting to bloom format..."
python "$SCRIPT_DIR/convert_deception_bench.py" "$JSONL_PATH" --output_dir "$OUTPUT_DIR"

# ── Report outputs ───────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Done! Output files:"
echo "=========================================="
echo "  $OUTPUT_DIR/rollout.json"
echo "  $OUTPUT_DIR/judgment.json"
