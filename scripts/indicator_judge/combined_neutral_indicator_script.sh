#!/bin/bash
#
# Combined neutral indicator detection script
#
# Runs both fine-grained and coarse-grained (general) neutral indicator detection
# across multiple rollout folders.
#
# Usage:
#   ./scripts/combined_neutral_indicator_script.sh
#
# Edit the ROLLOUT_CONFIGS array below to specify which rollout folders to process.
# Each entry is: "<rollout_file>|<output_base_dir>"

set -e

# ============================================================
# Configuration: Add rollout files and their output directories
# ============================================================
# Format: "<rollout_file>|<output_base_dir>"
# The script will create two subdirectories under each output_base_dir:
#   <output_base_dir>_neutral_finegrain/  (fine-grained indicators)
#   <output_base_dir>_neutral_coarse/     (coarse-grained/general indicators)

ROLLOUT_CONFIGS=(
    # "bloom/bloom-results/sandbagging/rollout.json|bloom/indicator_results/sandbagging"
    # "bloom/bloom-results/sycophancy/rollout.json|bloom/indicator_results/sycophancy"
    "bloom/bloom-results/undermining_oversight/rollout.json|bloom/indicator_results/undermining_oversight"
    # "bloom/bloom-results/instructed-long-horizon-sabotage-backdoor/rollout.json|bloom/indicator_results/sabotage_backdoor"
    # "bloom/bloom-results/sandbagging_benign/rollout.json|bloom/indicator_results/sandbagging_benign"
    # "bloom/bloom-results/sycophancy_benign/rollout.json|bloom/indicator_results/sycophancy_benign"
    # "bloom/bloom-results/undermining_oversight_benign/rollout.json|bloom/indicator_results/undermining_oversight_benign"
)

# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL=${#ROLLOUT_CONFIGS[@]}
CURRENT=0

echo "============================================================"
echo "Combined Neutral Indicator Detection"
echo "============================================================"
echo "Total rollout folders: $TOTAL"
echo "Each folder will get both fine-grained and coarse-grained detection"
echo "============================================================"
echo ""

for config in "${ROLLOUT_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Parse config
    ROLLOUT_FILE="${config%%|*}"
    OUTPUT_BASE="${config##*|}"

    FINEGRAIN_DIR="${OUTPUT_BASE}_neutral_finegrain"
    COARSE_DIR="${OUTPUT_BASE}_neutral_coarse"

    echo ""
    echo "************************************************************"
    echo "[$CURRENT/$TOTAL] Processing: $ROLLOUT_FILE"
    echo "************************************************************"

    # --- Fine-grained (no --general) ---
    echo ""
    echo "--- [$CURRENT/$TOTAL] Fine-grained neutral indicators ---"
    echo "Output: $FINEGRAIN_DIR"
    bash "$SCRIPT_DIR/run_neutral_per_turn_indicator_judge.sh" \
        "$ROLLOUT_FILE" "$FINEGRAIN_DIR"

    # --- Coarse-grained (--general) ---
    echo ""
    echo "--- [$CURRENT/$TOTAL] Coarse-grained neutral indicators ---"
    echo "Output: $COARSE_DIR"
    bash "$SCRIPT_DIR/run_neutral_per_turn_indicator_judge.sh" \
        "$ROLLOUT_FILE" "$COARSE_DIR" --general

    echo ""
    echo "[$CURRENT/$TOTAL] Done: $ROLLOUT_FILE"
done

echo ""
echo "============================================================"
echo "All $TOTAL rollout folders processed!"
echo "============================================================"
echo ""
echo "Results:"
for config in "${ROLLOUT_CONFIGS[@]}"; do
    OUTPUT_BASE="${config##*|}"
    echo "  Fine-grained: ${OUTPUT_BASE}_neutral_finegrain/"
    echo "  Coarse:       ${OUTPUT_BASE}_neutral_coarse/"
done
