#!/bin/bash

# Script to run indicator detection on rollout file using Claude Opus 4.5

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

# Resolve data_generation directory (parent of scripts/)
DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SUBDIR="sandbagging"
ROLLOUT_FILE="$DATA_GEN_DIR/bloom/bloom-results/$SUBDIR/rollout.json"
MODEL="claude-opus-4-5-20251101" # claude-haiku-4-5-20251001

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    exit 1
fi

echo "Starting indicator detection..."
echo "Input: $ROLLOUT_FILE"
echo "Model: $MODEL"
echo "---"

# Run the judge with anthropic/claude-opus-4-5-20251101
PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" python -m black_box_ind_judge "$ROLLOUT_FILE" \
    --rollout \
    --general \
    -p anthropic \
    -m "$MODEL" \
    -c 5 \
    -o "$DATA_GEN_DIR/indicator_results/${MODEL}_rollout_evaluated_${SUBDIR}_general_rep1.json"

echo ""
echo "Indicator detection complete!"
echo "Results saved to: $DATA_GEN_DIR/indicator_results/${MODEL}_rollout_evaluated_${SUBDIR}_general_rep1.json"

# Visualize transcripts with indicator highlighting
INDICATOR_FILE="$DATA_GEN_DIR/indicator_results/${MODEL}_rollout_evaluated_${SUBDIR}_general_rep1.json"
OUTPUT_HTML="$DATA_GEN_DIR/bloom/bloom-results/${SUBDIR}/visualization_${MODEL}_${SUBDIR}.html"

echo ""
echo "Generating visualization..."
echo "Rollout file: $ROLLOUT_FILE"
echo "Indicator file: $INDICATOR_FILE"
echo "Output: $OUTPUT_HTML"

python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
    --indicators "$INDICATOR_FILE" \
    --output "$OUTPUT_HTML"

echo ""
echo "Visualization complete!"
echo "HTML output: $OUTPUT_HTML"
