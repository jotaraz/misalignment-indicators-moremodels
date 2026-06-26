#!/bin/bash
# Generate synthetic transcripts and train probes for selected indicators.
#
# Generates and trains both:
#   - Behavior-level probes (sabotage, reward_hacking, deception)
#   - Finegrain probes (6 misalignment indicators, no precursors)
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_generate_and_train.sh [label_mode]
#
# Examples:
#   bash scripts/probe_train_eval/run_probe_generate_and_train.sh        # default: turn mode
#   bash scripts/probe_train_eval/run_probe_generate_and_train.sh span    # span mode

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=${1:-turn}

# Generation params
K=200               # transcripts per label (positive; negative = k//2)
MAX_CONCURRENT=15   # parallel generation processes

# Training params
REG_COEFF=10.0

# --- Behavior-level indicators ---
BEHAVIOR_INDICATORS=(
    "Sabotage"
    "Reward Hacking"
    "Deception"
)

# --- Finegrain indicators (no precursors) ---
FINEGRAIN_INDICATORS=(
    "Observer Suspicion Modeling"
    "Recognized-Concern Suppression"
    "Manufactured Agreement Shift"
    "Dishonest Positive Evaluation"
    "Concealment Planning"
    "Unauthorized Consequential Action Planning"
)

echo "========================================"
echo "Label mode:  ${LABEL_MODE}"
echo "K:           ${K}"
echo "========================================"
echo "Behavior indicators (${#BEHAVIOR_INDICATORS[@]}):"
for ind in "${BEHAVIOR_INDICATORS[@]}"; do
    echo "  - ${ind}"
done
echo "Finegrain indicators (${#FINEGRAIN_INDICATORS[@]}):"
for ind in "${FINEGRAIN_INDICATORS[@]}"; do
    echo "  - ${ind}"
done
echo ""

cd "${BASE_DIR}"

# --- Step 1: Generate behavior transcripts ---
echo "[Step 1/4] Generating behavior-level transcripts (k=${K})..."
${PYTHON} -m probe.generate \
    --indicator-set behavior \
    --indicator "${BEHAVIOR_INDICATORS[@]}" \
    --k ${K} \
    --max-concurrent ${MAX_CONCURRENT}

# --- Step 2: Generate finegrain transcripts ---
echo ""
echo "[Step 2/4] Generating finegrain transcripts (k=${K})..."
${PYTHON} -m probe.generate \
    --indicator-set finegrain \
    --indicator "${FINEGRAIN_INDICATORS[@]}" \
    --k ${K} \
    --max-concurrent ${MAX_CONCURRENT} \
    --overwrite

# --- Step 3: Train behavior probes ---
echo ""
echo "[Step 3/4] Training behavior probes (label_mode=${LABEL_MODE}, reg_coeff=${REG_COEFF})..."
${PYTHON} -m probe.train \
    --indicator-set behavior \
    --indicator "${BEHAVIOR_INDICATORS[@]}" \
    --label-mode "${LABEL_MODE}" \
    --reg-coeff ${REG_COEFF}

# --- Step 4: Train finegrain probes ---
echo ""
echo "[Step 4/4] Training finegrain probes (label_mode=${LABEL_MODE}, reg_coeff=${REG_COEFF})..."
${PYTHON} -m probe.train \
    --indicator-set finegrain \
    --indicator "${FINEGRAIN_INDICATORS[@]}" \
    --label-mode "${LABEL_MODE}" \
    --reg-coeff ${REG_COEFF}

echo ""
echo "Done."
echo "  Behavior probes: ${BASE_DIR}/probe/probes/behavior/"
echo "  Finegrain probes: ${BASE_DIR}/probe/probes/finegrain/"
