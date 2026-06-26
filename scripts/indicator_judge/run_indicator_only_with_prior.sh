#!/bin/bash
#
# Rerun the INDICATOR turn judge only (stage 2) on all 10 dirs, with the new
# --test-behavior soft prior enabled. Assumes the misalignment judge has
# already written its output to rollout_misalignment_turns.json — the indicator
# judge merges into that file, overwriting only `indicators_turns` and
# metadata.indicator_judge. Misalignment fields (summary, misaligned_turns) are
# preserved.
#
# Before invoking, back up the previous indicator-judge output for comparison:
#   cp rollout_misalignment_turns.json rollout_misalignment_turns_no_prior.json
#
# Usage:
#   bash scripts/indicator_judge/run_indicator_only_with_prior.sh

set -e

ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading env from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

DATA_GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${DATA_GEN_DIR}/bloom/.venv/bin/python"

MODEL="claude-opus-4-6"
THINKING_BUDGET=20000
MAX_CONCURRENT=10
PER_TURN_CONCURRENCY=5

# Format: "<behavior_label>|<rollout_path>"
ROLLOUTS=(
    "sabotage|${DATA_GEN_DIR}/bloom/bloom-results/instructed-covert-code-sabotage_glm_4_7_flash/rollout.json"
    "sandbagging|${DATA_GEN_DIR}/bloom/bloom-results/instructed-strategic-sandbagging_glm_4_7_flash/rollout.json"
    "sycophancy|${DATA_GEN_DIR}/bloom/bloom-results/sycophancy/rollout.json"
    "self-preservation|${DATA_GEN_DIR}/bloom/bloom-results/self-preservation_glm_4_7_flash/rollout.json"
    "strategic-deception|${DATA_GEN_DIR}/bloom/bloom-results/strategic-deception_glm_4_7_flash/rollout.json"
    "sabotage|${DATA_GEN_DIR}/bloom/bloom-results-test/test_instructed-covert-code-sabotage_glm_4_7_flash/rollout.json"
    "sandbagging|${DATA_GEN_DIR}/bloom/bloom-results-test/test_instructed-strategic-sandbagging_glm_4_7_flash/rollout.json"
    "sycophancy|${DATA_GEN_DIR}/bloom/bloom-results-test/test_sycophancy_glm_4_7_flash/rollout.json"
    "self-preservation|${DATA_GEN_DIR}/bloom/bloom-results-test/test_self-preservation_glm_4_7_flash/rollout.json"
    "strategic-deception|${DATA_GEN_DIR}/bloom/bloom-results-test/test_strategic-deception_glm_4_7_flash/rollout.json"
)

TOTAL=${#ROLLOUTS[@]}
CURRENT=0

echo "============================================================"
echo "Indicator Turn Judge — rerun with test_behavior prior"
echo "============================================================"
echo "Dirs:            $TOTAL"
echo "Model:           $MODEL (adaptive thinking, medium effort)"
echo "Concurrency:     rollouts=${MAX_CONCURRENT}, turns=${PER_TURN_CONCURRENCY}"
echo "============================================================"

for ENTRY in "${ROLLOUTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    TEST_BEHAVIOR="${ENTRY%%|*}"
    ROLLOUT="${ENTRY##*|}"
    DIR="$(dirname "$ROLLOUT")"
    BASENAME="$(basename "$ROLLOUT" .json)"
    OUTPUT_FILE="${DIR}/${BASENAME}_misalignment_turns.json"

    if [ ! -f "$ROLLOUT" ]; then
        echo "[$CURRENT/$TOTAL] MISSING rollout: $ROLLOUT — skipping"
        continue
    fi
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] MISSING misalign output: $OUTPUT_FILE — skipping (run misalignment judge first)"
        continue
    fi

    echo ""
    echo "[$CURRENT/$TOTAL] $(basename "$DIR")  (behavior: ${TEST_BEHAVIOR})"
    PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" $PYTHON -m black_box_ind_judge "$ROLLOUT" \
        --indicator-turns \
        --model "$MODEL" \
        --thinking-budget "$THINKING_BUDGET" \
        -c "$MAX_CONCURRENT" \
        --per-turn-concurrency "$PER_TURN_CONCURRENCY" \
        --indicator-set v2_6 \
        --test-behavior "$TEST_BEHAVIOR" \
        -o "$OUTPUT_FILE"
done

echo ""
echo "============================================================"
echo "All $TOTAL dirs completed"
echo "============================================================"
