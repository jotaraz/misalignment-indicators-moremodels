#!/bin/bash
#
# Run BOTH turn judges (misalignment + indicator) on 5 dev + 5 test bloom dirs.
#
# Flow per dir:
#   1. Misalignment judge (--per-turn) writes rollout_misalignment_turns.json
#      with keys `summary` + `misaligned_turns`.
#   2. Indicator judge (--indicator-turns) merges `indicators_turns` into the
#      same file, preserving the misalignment fields.
#
# Assumes `rollout_misalignment_turns.json` has already been archived to
# `rollout_misalignment_turns_v3.json` before invoking.
#
# Usage:
#   bash scripts/indicator_judge/run_both_turn_judges_v3.sh

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

# Format: "<behavior_label>|<rollout_path>". behavior_label is the value used
# by the indicator judge's `--test-behavior` soft prior (must match one of the
# `relevant_behaviors` values in indicators/misalignment_indicators_v2_6.py).
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
echo "Both Turn Judges (misalignment + indicator) — v3"
echo "============================================================"
echo "Dirs:            $TOTAL"
echo "Model:           $MODEL"
echo "Thinking budget: $THINKING_BUDGET"
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
        echo "[$CURRENT/$TOTAL] MISSING: $ROLLOUT — skipping"
        continue
    fi

    # Skip if both stages already completed (indicator judge writes
    # metadata.indicator_judge on success).
    if [ -f "$OUTPUT_FILE" ] && $PYTHON -c "import json,sys; sys.exit(0 if json.load(open('$OUTPUT_FILE')).get('metadata',{}).get('indicator_judge') else 1)" 2>/dev/null; then
        echo "[$CURRENT/$TOTAL] SKIP (complete): $(basename "$DIR")"
        continue
    fi

    echo ""
    echo "************************************************************"
    echo "[$CURRENT/$TOTAL] $(basename "$DIR")  (behavior: ${TEST_BEHAVIOR})"
    echo "************************************************************"
    echo "  Rollout: $ROLLOUT"
    echo "  Output:  $OUTPUT_FILE"

    echo ""
    echo "  --- Stage 1: misalignment turn judge ---"
    PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" $PYTHON -m black_box_ind_judge "$ROLLOUT" \
        --misalignment-turns \
        --per-turn \
        --model "$MODEL" \
        --thinking-budget "$THINKING_BUDGET" \
        -c "$MAX_CONCURRENT" \
        --per-turn-concurrency "$PER_TURN_CONCURRENCY" \
        -o "$OUTPUT_FILE"

    echo ""
    echo "  --- Stage 2: indicator turn judge (merge, behavior=${TEST_BEHAVIOR}) ---"
    PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" $PYTHON -m black_box_ind_judge "$ROLLOUT" \
        --indicator-turns \
        --model "$MODEL" \
        --thinking-budget "$THINKING_BUDGET" \
        -c "$MAX_CONCURRENT" \
        --per-turn-concurrency "$PER_TURN_CONCURRENCY" \
        --indicator-set v2_6 \
        --test-behavior "$TEST_BEHAVIOR" \
        -o "$OUTPUT_FILE"

    echo "[$CURRENT/$TOTAL] Done: $(basename "$DIR")"
done

echo ""
echo "============================================================"
echo "All $TOTAL dirs completed"
echo "============================================================"
