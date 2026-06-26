#!/bin/bash
# Run agentic-misalignment experiments and convert to bloom rollout format.
#
# Steps:
#   1. Setup uv venv if needed
#   2. Generate prompts from config
#   3. Run model experiments
#   4. Convert results to bloom format (rollout.json + judgment.json)
#
# Output:
#   <OUTPUT_DIR>/rollout.json + judgment.json
#
# Usage:
#   bash scripts/misalign_behaviors/agentic_misalignment.sh
#   bash scripts/misalign_behaviors/agentic_misalignment.sh --config configs/example_experiment_config.yaml
#   bash scripts/misalign_behaviors/agentic_misalignment.sh --results-dir results/my_experiment --skip-generation

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG="configs/example_experiment_config.yaml"
RESULTS_DIR=""               # Auto-detected from config if empty
OUTPUT_DIR=""                # Auto-set based on experiment_id if empty
SKIP_GENERATION=false        # Set to true to skip prompt generation + experiment, just convert
SAMPLES_PER_CONDITION="10"   # Samples per condition (overrides config value)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AGENTIC_DIR="$PROJECT_DIR/ood_misalignment_eval/agentic-misalignment"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="$2";       shift 2 ;;
        --results-dir)     RESULTS_DIR="$2";  shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";   shift 2 ;;
        --skip-generation) SKIP_GENERATION=true; shift ;;
        --samples-per-condition) SAMPLES_PER_CONDITION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

echo "=========================================="
echo "Agentic Misalignment Pipeline"
echo "=========================================="
echo "Config:          $CONFIG"
echo "Skip generation: $SKIP_GENERATION"
echo "=========================================="
echo ""

# ── Step 1: Setup uv venv ───────────────────────────────────────────────────
if [ ! -d "$AGENTIC_DIR/.venv" ]; then
    echo "[Setup] Creating uv virtual environment in agentic-misalignment/..."
    cd "$AGENTIC_DIR"
    uv venv
    uv pip install -r requirements.txt
    echo "  Done."
    echo ""
fi

if [ "$SKIP_GENERATION" = false ]; then
    # ── Step 2: Generate prompts ──────────────────────────────────────────────
    echo "[Step 1/3] Generating prompts..."
    uv run --directory "$AGENTIC_DIR" python scripts/generate_prompts.py --config "$CONFIG"
    echo ""

    # ── Step 3: Run experiments ───────────────────────────────────────────────
    echo "[Step 2/3] Running experiments..."
    uv run --directory "$AGENTIC_DIR" python scripts/run_experiments.py --config "$CONFIG" --samples "$SAMPLES_PER_CONDITION"
    echo ""
fi

# ── Determine results directory ──────────────────────────────────────────────
if [ -z "$RESULTS_DIR" ]; then
    # Extract experiment_id from config
    EXPERIMENT_ID=$(python -c "
import yaml
with open('$AGENTIC_DIR/$CONFIG') as f:
    config = yaml.safe_load(f)
print(config.get('experiment_id', 'unknown'))
" 2>/dev/null || echo "unknown")
    RESULTS_DIR="$AGENTIC_DIR/results/$EXPERIMENT_ID"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$RESULTS_DIR/bloom_rollout"
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# ── Step 4: Convert to bloom format ──────────────────────────────────────────
STEP_NUM="3"
if [ "$SKIP_GENERATION" = true ]; then
    STEP_NUM="1"
    TOTAL="1"
else
    TOTAL="3"
fi

echo "[Step $STEP_NUM/$TOTAL] Converting to bloom format..."
python "$SCRIPT_DIR/convert_agentic_misalignment.py" "$RESULTS_DIR" --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Done! Output files:"
echo "=========================================="
echo "  $OUTPUT_DIR/rollout.json"
echo "  $OUTPUT_DIR/judgment.json"
