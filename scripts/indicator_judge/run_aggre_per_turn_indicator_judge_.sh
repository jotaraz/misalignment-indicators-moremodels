#!/bin/bash
#
# Run per-turn aggregated indicator detection with k=3 runs and generate visualization
#
# This version evaluates each turn separately (with the full conversation as context),
# guiding the model to first list candidate sentences then evaluate each independently.
# Results are aggregated across k runs and filtered.
#
# Usage:
#   ./run_aggre_per_turn_indicator_judge_.sh <rollout_file> [output_dir]
#
# Example:
#   ./scripts/run_aggre_per_turn_indicator_judge_.sh bloom/bloom-results/instructed-long-horizon-sabotage-backdoor/rollout.json bloom/indicator_results/instructed-long-horizon-sabotage-backdoor/

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
K=3
TEMP=1

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
echo "Running Per-Turn Aggregated Indicator Detection"
echo "=========================================="
echo "Rollout file: $ROLLOUT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Mode: per-turn (evaluate each turn separately)"
echo "Aggregation: k=$K runs, temp=$TEMP"
echo "=========================================="

# Step 1: Run per-turn aggregated indicator detection
echo ""
echo "[Step 1/5] Running per-turn indicator detection with k=$K aggregation..."
PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" python -m black_box_ind_judge "$ROLLOUT_FILE" \
    --rollout \
    --per-turn \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --aggregate-k "$K" \
    --aggregate-temp "$TEMP" \
    -o "$OUTPUT_DIR/${BASENAME}_aggregated.json"

# Define output file names
AGGREGATED_FILE="$OUTPUT_DIR/${BASENAME}_aggregated.json"

echo ""
echo "[Step 2/5] Generating HTML visualizations for individual runs..."

# Generate HTML for each individual run
for i in $(seq 1 "$K"); do
    RUN_FILE="$OUTPUT_DIR/${BASENAME}_run${i}.json"
    if [ -f "$RUN_FILE" ]; then
        echo "  - Generating visualization for run $i..."
        python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
            --indicators "$RUN_FILE" \
            -o "$OUTPUT_DIR/${BASENAME}_run${i}.html"
    else
        echo "  - Warning: Run $i file not found: $RUN_FILE"
    fi
done

echo ""
echo "[Step 3/5] Generating visualization for aggregated (unfiltered) indicators..."
# Convert unfiltered_indicators format to detected_indicators format for visualization
python -c "
import json

with open('$AGGREGATED_FILE') as f:
    data = json.load(f)

converted = {
    'metadata': data.get('metadata', {}),
    'evaluated_rollouts': []
}

for rollout in data.get('evaluated_rollouts', []):
    detected = []
    for ind in rollout.get('unfiltered_indicators', []):
        best_detection = max(ind.get('detections', [{'confidence': 0, 'evidence': ''}]),
                             key=lambda x: x.get('confidence', 0))
        detected.append({
            'indicator_name': ind.get('indicator_name'),
            'turn_number': ind.get('turn_number'),
            'sentence': ind.get('sentence', ''),
            'evidence': best_detection.get('evidence', ''),
            'confidence': best_detection.get('confidence', 0),
        })

    converted['evaluated_rollouts'].append({
        'variation_number': rollout.get('variation_number'),
        'variation_description': rollout.get('variation_description'),
        'repetition_number': rollout.get('repetition_number'),
        'detected_indicators': detected,
    })

with open('$OUTPUT_DIR/${BASENAME}_unfiltered.json', 'w') as f:
    json.dump(converted, f, indent=2)
"

python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
    --indicators "$OUTPUT_DIR/${BASENAME}_unfiltered.json" \
    -o "$OUTPUT_DIR/${BASENAME}_unfiltered.html"

echo ""
echo "[Step 4/5] Generating visualization for filtered indicators..."
# Convert filtered_indicators format to detected_indicators format for visualization
python -c "
import json

with open('$AGGREGATED_FILE') as f:
    data = json.load(f)

converted = {
    'metadata': data.get('metadata', {}),
    'evaluated_rollouts': []
}

for rollout in data.get('evaluated_rollouts', []):
    detected = []
    for ind in rollout.get('filtered_indicators', []):
        detected.append({
            'indicator_name': ind.get('indicator_name'),
            'turn_number': ind.get('turn_number'),
            'sentence': ind.get('sentence', ''),
            'evidence': ind.get('evidence', ''),
            'confidence': ind.get('confidence', 0),
        })

    converted['evaluated_rollouts'].append({
        'variation_number': rollout.get('variation_number'),
        'variation_description': rollout.get('variation_description'),
        'repetition_number': rollout.get('repetition_number'),
        'detected_indicators': detected,
    })

with open('$OUTPUT_DIR/${BASENAME}_filtered.json', 'w') as f:
    json.dump(converted, f, indent=2)
"

python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
    --indicators "$OUTPUT_DIR/${BASENAME}_filtered.json" \
    -o "$OUTPUT_DIR/${BASENAME}_filtered.html"

echo ""
echo "[Step 5/5] Generating combined HTML viewer..."
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
echo "  - Aggregated (main):  $AGGREGATED_FILE"
for i in $(seq 1 "$K"); do
    echo "  - Individual run $i:   $OUTPUT_DIR/${BASENAME}_run${i}.json"
done
echo "  - Unfiltered format:  $OUTPUT_DIR/${BASENAME}_unfiltered.json"
echo "  - Filtered format:    $OUTPUT_DIR/${BASENAME}_filtered.json"
echo ""
echo "HTML Visualizations:"
for i in $(seq 1 "$K"); do
    echo "  - Run $i:       $OUTPUT_DIR/${BASENAME}_run${i}.html"
done
echo "  - Unfiltered:  $OUTPUT_DIR/${BASENAME}_unfiltered.html"
echo "  - Filtered:    $OUTPUT_DIR/${BASENAME}_filtered.html"
echo "  - Combined:    $OUTPUT_DIR/${BASENAME}_combined.html  <-- Open this for full comparison view"
