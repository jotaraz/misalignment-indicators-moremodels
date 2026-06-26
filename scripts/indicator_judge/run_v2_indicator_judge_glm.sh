#!/bin/bash
#
# Run v2 indicator detection across all behaviors, indicator granularities, and models.
#
# Settings: --rollout --per-turn --neutral --no-future-context --spans
# Aggregation: k=3, temp=1 (same as existing scripts)
# Models: haiku 4.5 and opus 4.5
# Indicator sets: v2.2-finegrain, v2.2-mid, v2.2-per-behavior, v2.2-cross-behavior
#
# Output structure:
#   bloom/indicator_results/v2/<behavior>/<indicator_set>_<model_short>/
#
# Usage:
#   ./scripts/run_v2_indicator_judge.sh
#
# To run a subset, edit the BEHAVIORS, INDICATOR_SETS, and MODELS arrays below.

set -e

# Load environment variables from .env file
ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please set ANTHROPIC_API_KEY environment variable"
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    exit 1
fi

# Resolve data_generation directory (parent of scripts/)
DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# Configuration
# ============================================================

# Behavior folders and their rollout files
# Format: "<short_name>|<rollout_path>"
BEHAVIORS=(
    "sabotage_backdoor|bloom/bloom-results/instructed-long-horizon-sabotage-backdoor_/rollout.json"
    "sabotage_bug|bloom/bloom-results/instructed-long-horizon-sabotage-bug/rollout.json"
    # "sandbagging|bloom/bloom-results/sandbagging/rollout.json"
    "sycophancy|bloom/bloom-results/sycophancy/rollout.json"
    # "undermining_oversight|bloom/bloom-results/undermining_oversight/rollout.json"
)

INDICATOR_SETS=(
    "v2.2-finegrain"
    "v2.2-mid"
    "v2.2-per-behavior"
    "v2.2-cross-behavior"
)

# Format: "<model_id>|<short_name>"
MODELS=(
    "z-ai/glm-4.7-flash|glm_flash"
)

# Aggregation settings
PROVIDER="openrouter"
K=3
TEMP=1

# Output base directory
OUTPUT_BASE="bloom/indicator_results/v2.2"

# ============================================================
# Compute total jobs
# ============================================================
TOTAL_JOBS=$(( ${#BEHAVIORS[@]} * ${#INDICATOR_SETS[@]} * ${#MODELS[@]} ))
CURRENT_JOB=0

echo "============================================================"
echo "V2 Indicator Detection — Full Matrix"
echo "============================================================"
echo "Behaviors:      ${#BEHAVIORS[@]}"
echo "Indicator sets: ${#INDICATOR_SETS[@]}"
echo "Models:         ${#MODELS[@]}"
echo "Total jobs:     $TOTAL_JOBS"
echo "Aggregation:    k=$K, temp=$TEMP"
echo "Flags:          --rollout --per-turn --neutral --no-future-context --spans"
echo "Output base:    $OUTPUT_BASE"
echo "============================================================"
echo ""

# ============================================================
# Helper: run one detection + visualization job
# ============================================================
run_one_job() {
    local BEHAVIOR_NAME="$1"
    local ROLLOUT_FILE="$2"
    local IND_SET="$3"
    local MODEL_ID="$4"
    local MODEL_SHORT="$5"

    local OUTPUT_DIR="${OUTPUT_BASE}/${BEHAVIOR_NAME}/${IND_SET}_${MODEL_SHORT}"
    local BASENAME=$(basename "$ROLLOUT_FILE" .json)

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "  [Detection] Running per-turn neutral indicator detection..."
    echo "    Rollout:    $ROLLOUT_FILE"
    echo "    Indicators: $IND_SET"
    echo "    Model:      $MODEL_ID ($MODEL_SHORT)"
    echo "    Output:     $OUTPUT_DIR"
    echo ""

    # Step 1: Run per-turn aggregated indicator detection
    PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" python -m black_box_ind_judge "$ROLLOUT_FILE" \
        --rollout \
        --per-turn \
        --neutral \
        --no-future-context \
        --spans \
        --indicators "$IND_SET" \
        --provider "$PROVIDER" \
        --model "$MODEL_ID" \
        --aggregate-k "$K" \
        --aggregate-temp "$TEMP" \
        -c 10 \
        -o "$OUTPUT_DIR/${BASENAME}_aggregated.json"

    local AGGREGATED_FILE="$OUTPUT_DIR/${BASENAME}_aggregated.json"

    # Step 2: Generate HTML visualizations for individual runs
    for i in $(seq 1 "$K"); do
        RUN_FILE="$OUTPUT_DIR/${BASENAME}_run${i}.json"
        if [ -f "$RUN_FILE" ]; then
            python "$DATA_GEN_DIR/visualize_transcripts.py" "$ROLLOUT_FILE" \
                --indicators "$RUN_FILE" \
                -o "$OUTPUT_DIR/${BASENAME}_run${i}.html"
        fi
    done

    # Step 3: Generate visualization for aggregated (unfiltered) indicators
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

    # Step 4: Generate visualization for filtered indicators
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

    # Step 5: Generate combined HTML viewer
    python "$DATA_GEN_DIR/generate_combined_indicator_html.py" \
        --input-dir "$OUTPUT_DIR" \
        --basename "$BASENAME" \
        -o "$OUTPUT_DIR/${BASENAME}_combined.html"
}

# ============================================================
# Main loop
# ============================================================
for behavior_config in "${BEHAVIORS[@]}"; do
    BEHAVIOR_NAME="${behavior_config%%|*}"
    ROLLOUT_FILE="${behavior_config##*|}"

    if [ ! -f "$ROLLOUT_FILE" ]; then
        echo "WARNING: Rollout file not found: $ROLLOUT_FILE — skipping"
        continue
    fi

    for IND_SET in "${INDICATOR_SETS[@]}"; do
        for model_config in "${MODELS[@]}"; do
            MODEL_ID="${model_config%%|*}"
            MODEL_SHORT="${model_config##*|}"

            CURRENT_JOB=$((CURRENT_JOB + 1))

            # Skip if aggregated output already exists (resume support)
            ROLLOUT_BASENAME=$(basename "$ROLLOUT_FILE" .json)
            EXPECTED_OUTPUT="${OUTPUT_BASE}/${BEHAVIOR_NAME}/${IND_SET}_${MODEL_SHORT}/${ROLLOUT_BASENAME}_aggregated.json"
            if [ -f "$EXPECTED_OUTPUT" ]; then
                echo ""
                echo "[$CURRENT_JOB/$TOTAL_JOBS] SKIP (already done): $BEHAVIOR_NAME | $IND_SET | $MODEL_SHORT"
                continue
            fi

            echo ""
            echo "************************************************************"
            echo "[$CURRENT_JOB/$TOTAL_JOBS] $BEHAVIOR_NAME | $IND_SET | $MODEL_SHORT"
            echo "************************************************************"

            run_one_job "$BEHAVIOR_NAME" "$ROLLOUT_FILE" "$IND_SET" "$MODEL_ID" "$MODEL_SHORT"

            echo ""
            echo "[$CURRENT_JOB/$TOTAL_JOBS] Done: $BEHAVIOR_NAME | $IND_SET | $MODEL_SHORT"
        done
    done
done

echo ""
echo "============================================================"
echo "All $TOTAL_JOBS jobs completed!"
echo "============================================================"
echo ""
echo "Results saved under: $OUTPUT_BASE/"
echo ""
echo "Directory structure:"
for behavior_config in "${BEHAVIORS[@]}"; do
    BEHAVIOR_NAME="${behavior_config%%|*}"
    for IND_SET in "${INDICATOR_SETS[@]}"; do
        for model_config in "${MODELS[@]}"; do
            MODEL_SHORT="${model_config##*|}"
            echo "  ${OUTPUT_BASE}/${BEHAVIOR_NAME}/${IND_SET}_${MODEL_SHORT}/"
        done
    done
done
