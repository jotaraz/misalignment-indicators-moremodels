#!/bin/bash
# Run ImpossibleBench evaluation and convert logs to bloom rollout format.
#
# Steps:
#   1. Setup uv venv if needed
#   2. Run inspect_ai evaluation via demo.py
#   3. Convert eval logs to bloom format (rollout.json + judgment.json)
#
# Output:
#   <OUTPUT_DIR>/rollout.json + judgment.json
#
# Usage:
#   bash scripts/misalign_behaviors/impossiblebench.sh
#   bash scripts/misalign_behaviors/impossiblebench.sh --task replicate_lcb --split conflicting
#   bash scripts/misalign_behaviors/impossiblebench.sh --log-dir /path/to/logs --skip-generation

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
TASK="replicate_lcb"          # Options: replicate_lcb, replicate_swe
SPLIT="conflicting"           # Options: original, oneoff, conflicting
LOG_DIR=""                    # Auto-detected if empty
OUTPUT_DIR=""                 # Auto-set if empty
SKIP_GENERATION=false         # Set to true to skip evaluation, just convert
LIMIT="50"                    # Limit samples per task (per split)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
IB_DIR="$PROJECT_DIR/ood_misalignment_eval/impossiblebench"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2";       shift 2 ;;
        --split)           SPLIT="$2";      shift 2 ;;
        --log-dir)         LOG_DIR="$2";    shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --skip-generation) SKIP_GENERATION=true; shift ;;
        --limit)           LIMIT="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

echo "=========================================="
echo "ImpossibleBench Pipeline"
echo "=========================================="
echo "Task:            $TASK"
echo "Split:           $SPLIT"
echo "Skip generation: $SKIP_GENERATION"
echo "=========================================="
echo ""

# ── Step 1: Setup uv venv ───────────────────────────────────────────────────
if [ ! -d "$IB_DIR/.venv" ]; then
    echo "[Setup] Creating uv virtual environment in impossiblebench/..."
    cd "$IB_DIR"
    uv venv
    uv pip install -e .
    echo "  Done."
    echo ""
fi

if [ "$SKIP_GENERATION" = false ]; then
    # ── Step 2: Run evaluation ────────────────────────────────────────────────
    echo "[Step 1/2] Running ImpossibleBench evaluation..."
    EVAL_CMD="uv run --directory $IB_DIR python demo.py $TASK"
    if [ -n "$LIMIT" ]; then
        EVAL_CMD="$EVAL_CMD --limit $LIMIT"
    fi
    eval $EVAL_CMD
    echo ""
fi

# ── Determine log directory ──────────────────────────────────────────────────
if [ -z "$LOG_DIR" ]; then
    # inspect_ai stores logs in ~/.inspect/logs by default
    LOG_DIR="$HOME/.inspect/logs"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$IB_DIR/bloom_rollout/${TASK}_${SPLIT}"
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory not found: $LOG_DIR"
    echo "If logs are elsewhere, use --log-dir to specify the path."
    exit 1
fi

# ── Step 3: Convert to bloom format ──────────────────────────────────────────
STEP_NUM="2"
if [ "$SKIP_GENERATION" = true ]; then
    STEP_NUM="1"
    TOTAL="1"
else
    TOTAL="2"
fi

echo "[Step $STEP_NUM/$TOTAL] Converting eval logs to bloom format..."
# Run converter via impossiblebench venv (needs inspect_ai to read .eval logs)
uv run --directory "$IB_DIR" python "$SCRIPT_DIR/convert_impossiblebench.py" \
    "$LOG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT"

echo ""
echo "=========================================="
echo "Done! Output files:"
echo "=========================================="
echo "  $OUTPUT_DIR/rollout.json"
echo "  $OUTPUT_DIR/judgment.json"
