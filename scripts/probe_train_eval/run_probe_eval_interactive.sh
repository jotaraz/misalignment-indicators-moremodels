#!/bin/bash
# Interactive probe eval: run on a node with 1 GPU.
# Evaluates 2 new positive datasets, then runs indicator GT, misalignment GT, and visualization.
#
# Usage (from an interactive GPU node):
#   bash scripts/probe_train_eval/run_probe_eval_interactive.sh

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python
PROBES_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2
RESULTS_SUBDIR=v2_3_gen_prompt_v2_span_v2

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

cd "${BASE_DIR}"

PROBE_FOLDERS=$(find "${PROBES_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')

# New positive datasets
NEW_POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_glm_4_7_flash"
)

echo "=========================================="
echo "Probe Eval - add new datasets + GT metrics"
echo "Started: $(date)"
echo "=========================================="

# ---- Step 1: Evaluate on 2 new positive datasets ----
echo ""
echo "========== Step 1: Evaluate on new positive datasets =========="
# ${PYTHON} -m probe_eval.evaluate \
#     --experiment_folder ${PROBE_FOLDERS} \
#     --rollout_dir ${NEW_POSITIVE_ROLLOUT_DIRS[@]} \
#     --behavior_threshold 5

# ---- Step 2: Misalignment ground-truth metrics (with all-negative) ----
echo ""
echo "========== Step 2: Misalignment ground truth =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \
    --results-subdir ${RESULTS_SUBDIR} \
    --include-all-negative

# ---- Step 3: Generate misalignment dashboard visualization ----
echo ""
echo "========== Step 3: Misalignment dashboard visualization =========="
${PYTHON} probe_eval/visualize_misalignment.py \
    --results-subdir ${RESULTS_SUBDIR}

echo ""
echo "=========================================="
echo "Complete at $(date)"
echo "=========================================="
echo "Results: probe_eval/results/${RESULTS_SUBDIR}/"
