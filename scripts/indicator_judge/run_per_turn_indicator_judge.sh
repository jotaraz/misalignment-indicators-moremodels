#!/bin/bash
#
# Run per-turn indicator detection (single run, no aggregation or filtering)
#
# This version evaluates each turn separately (with the full conversation as context),
# guiding the model to first list candidate sentences then evaluate each independently.
# Results from all turns are assembled into a single output.
#
# Usage:
#   ./run_per_turn_indicator_judge.sh <rollout_file> [output_dir]
#
# Example:
#   ./run_per_turn_indicator_judge.sh /path/to/rollout.json indicator_results/sabotage/

set -e

# Load environment variables from .env file
ENV_FILE="/workspace-vast/kaiwen/persona_vectors/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please set ANTHROPIC_API_KEY environment variable"
    exit 1
fi

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    exit 1
fi

# Parse arguments
ROLLOUT_FILE="$1"
OUTPUT_DIR="${2:-indicator_results}"

# Model configuration
PROVIDER="anthropic"
MODEL="claude-opus-4-5-20251101"  # claude-haiku-4-5-20251001

if [ -z "$ROLLOUT_FILE" ]; then
    echo "Usage: $0 <rollout_file> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  rollout_file    Path to the rollout JSON file"
    echo "  output_dir      Output directory (default: indicator_results)"
    exit 1
fi

if [ ! -f "$ROLLOUT_FILE" ]; then
    echo "Error: Rollout file not found: $ROLLOUT_FILE"
    exit 1
fi

# Resolve data_generation directory (parent of scripts/)
DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Extract base name from rollout file
BASENAME=$(basename "$ROLLOUT_FILE" .json)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running Per-Turn Indicator Detection"
echo "=========================================="
echo "Rollout file: $ROLLOUT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Mode: per-turn (evaluate each turn separately, single run)"
echo "=========================================="

# Step 1: Run per-turn indicator detection (single run)
echo ""
echo "[Step 1/2] Running per-turn indicator detection..."
PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" python -m black_box_ind_judge "$ROLLOUT_FILE" \
    --rollout \
    --general \
    --per-turn \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    -o "$OUTPUT_DIR/${BASENAME}_per_turn.json"

# Step 2: Generate HTML visualization
echo ""
echo "[Step 2/2] Generating HTML visualization..."
python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
    --indicators "$OUTPUT_DIR/${BASENAME}_per_turn.json" \
    -o "$OUTPUT_DIR/${BASENAME}_per_turn.html"

echo ""
echo "=========================================="
echo "Completed! Generated files:"
echo "=========================================="
echo ""
echo "  JSON: $OUTPUT_DIR/${BASENAME}_per_turn.json"
echo "  HTML: $OUTPUT_DIR/${BASENAME}_per_turn.html"
