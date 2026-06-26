#!/bin/bash
set -e  # Stop on first error

BEHAVIORS=("sandbagging" "sycophancy") # "undermining_oversight"
INDICATOR_BASE="bloom/indicator_results"
JUDGMENT_BASE="bloom/bloom-results"
EVAL_DIR="bloom"

for behavior in "${BEHAVIORS[@]}"; do
    echo "=================================================="
    echo "Scoring: ${behavior}"
    echo "=================================================="

    # Consistency evaluation (across runs within the indicator results)
    echo "[${behavior}] Running indicator consistency evaluation..."
    # python ${EVAL_DIR}/evaluate_indicator_consistency.py \
    #     ${INDICATOR_BASE}/${behavior}_finegrain/
    # python ${EVAL_DIR}/evaluate_indicator_consistency.py \
    #     ${INDICATOR_BASE}/${behavior}/
    # python ${EVAL_DIR}/evaluate_indicator_consistency.py \
    #     ${INDICATOR_BASE}/${behavior}-multiturn/
    python ${EVAL_DIR}/evaluate_indicator_consistency.py \
        ${INDICATOR_BASE}/${behavior}_neutral_coarse/
    python ${EVAL_DIR}/evaluate_indicator_consistency.py \
        ${INDICATOR_BASE}/${behavior}_neutral_finegrain/
    # Indicator vs judgment evaluation
    echo "[${behavior}] Running indicator vs judgment evaluation..."
    # python ${EVAL_DIR}/evaluate_indicator_vs_judgment.py \
    #     --indicator-dir ${INDICATOR_BASE}/${behavior}_finegrain/ \
    #     --judgment-file ${JUDGMENT_BASE}/${behavior}/judgment.json
    # python ${EVAL_DIR}/evaluate_indicator_vs_judgment.py \
    #     --indicator-dir ${INDICATOR_BASE}/${behavior}/ \
    #     --judgment-file ${JUDGMENT_BASE}/${behavior}/judgment.json
    # python ${EVAL_DIR}/evaluate_indicator_vs_judgment.py \
    #     --indicator-dir ${INDICATOR_BASE}/${behavior}-multiturn/ \
    #     --judgment-file ${JUDGMENT_BASE}/${behavior}/judgment.json
    python ${EVAL_DIR}/evaluate_indicator_vs_judgment.py \
        --indicator-dir ${INDICATOR_BASE}/${behavior}_neutral_coarse/ \
        --judgment-file ${JUDGMENT_BASE}/${behavior}/judgment.json
    python ${EVAL_DIR}/evaluate_indicator_vs_judgment.py \
        --indicator-dir ${INDICATOR_BASE}/${behavior}_neutral_finegrain/ \
        --judgment-file ${JUDGMENT_BASE}/${behavior}/judgment.json
    echo ""
done

echo "All scoring complete."
