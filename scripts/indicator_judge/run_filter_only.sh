#!/bin/bash
#
# Run filtering on existing aggregated indicators and generate HTML
# This skips the k-run detection step and only runs the validation/filtering pass
#
# Usage:
#   ./run_filter_only.sh <aggregated_json_file> <rollout_file> [output_dir]
#
# Example:
#   scripts/run_filter_only.sh bloom/indicator_results/sandbag/rollout_aggregated.json bloom/bloom-results/sandbagging/rollout.json bloom/indicator_results/sandbag/

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
AGGREGATED_FILE="$1"
ROLLOUT_FILE="$2"
OUTPUT_DIR="${3:-$(dirname "$AGGREGATED_FILE")}"

# Model configuration
PROVIDER="anthropic"
MODEL="claude-opus-4-5-20251101"  # claude-haiku-4-5-20251001

if [ -z "$AGGREGATED_FILE" ] || [ -z "$ROLLOUT_FILE" ]; then
    echo "Usage: $0 <aggregated_json_file> <rollout_file> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  aggregated_json_file  Path to the aggregated JSON file (e.g., rollout_aggregated.json)"
    echo "  rollout_file          Path to the rollout JSON file"
    echo "  output_dir            Output directory (default: same as aggregated file)"
    exit 1
fi

if [ ! -f "$AGGREGATED_FILE" ]; then
    echo "Error: Aggregated file not found: $AGGREGATED_FILE"
    exit 1
fi

if [ ! -f "$ROLLOUT_FILE" ]; then
    echo "Error: Rollout file not found: $ROLLOUT_FILE"
    exit 1
fi

# Resolve data_generation directory (parent of scripts/)
DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Extract base name from aggregated file (remove _aggregated suffix)
BASENAME=$(basename "$AGGREGATED_FILE" .json | sed 's/_aggregated$//')

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running Filtering on Existing Aggregated Results"
echo "=========================================="
echo "Aggregated file: $AGGREGATED_FILE"
echo "Rollout file: $ROLLOUT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Basename: $BASENAME"
echo "=========================================="

echo ""
echo "[Step 1/3] Running filtering pass on aggregated indicators..."

# Run the filtering using black_box_ind_judge with --filter-only
PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" python -m black_box_ind_judge "$AGGREGATED_FILE" \
    --filter-only "$ROLLOUT_FILE" \
    --general \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    -o "$OUTPUT_DIR/${BASENAME}_filtered.json"

echo ""
echo "[Step 2/3] Generating HTML visualizations..."

# Generate HTML for filtered indicators
python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
    --indicators "$OUTPUT_DIR/${BASENAME}_filtered.json" \
    -o "$OUTPUT_DIR/${BASENAME}_filtered.html"

echo ""
echo "[Step 3/3] Generating combined HTML viewer..."
python "$DATA_GEN_DIR/generate_combined_indicator_html.py" \
    --input-dir "$OUTPUT_DIR" \
    --basename "$BASENAME" \
    -o "$OUTPUT_DIR/${BASENAME}_combined.html"

echo ""
echo "=========================================="
echo "Completed! Generated files:"
echo "=========================================="
echo ""
echo "JSON Results:"
echo "  - Filtered:    $OUTPUT_DIR/${BASENAME}_filtered.json"
echo ""
echo "HTML Visualizations:"
echo "  - Filtered:    $OUTPUT_DIR/${BASENAME}_filtered.html"
echo "  - Combined:    $OUTPUT_DIR/${BASENAME}_combined.html"
