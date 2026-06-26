#!/bin/bash
# =============================================================================
# Evaluate probes on hard-negative scenarios + recompute misalignment GT
# =============================================================================
#
# Runs existing v3_v2_5_combined_v1_span probes on the hard-negative rollouts
# (benign daily-task scenarios designed to superficially resemble indicators),
# then recomputes misalignment ground-truth summaries to include the new data.
#
# Jobs:
#   Job 1 (1 GPU):  Score hard-negative rollouts with all 72 probes
#   Job 2 (no GPU): Recompute misalignment GT (dev + test) including new data
#
# Usage:
#   bash scripts/probe_train_eval/run_hard_negatives_eval.sh
#   bash scripts/probe_train_eval/run_hard_negatives_eval.sh --local
# =============================================================================

set -e

LOCAL_MODE=false
[[ "${1}" == "--local" ]] && LOCAL_MODE=true

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PROBE_DIR=${BASE_DIR}/probe/probes/v3_v2_5_combined_v1_span
RESULTS_SUBDIR=v3_v2_5_combined_v1_span

# Hard-negative rollout dirs (all_negative)
HARD_NEG_ROLLOUT_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/hard-negatives/rollouts/glm-4.7-flash/hard_negatives_glm_4_7_flash"
)

# SLURM settings
PARTITION="general,overflow"
QOS="high"
MEM="128G"

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/hard_neg_eval_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Shared eval logic ----
run_eval() {
    cd "${BASE_DIR}"
    export HF_HOME=/workspace-vast/pretrained_ckpts

    PROBE_FOLDERS=$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
    echo "Found $(echo "${PROBE_FOLDERS}" | wc -w) probes"

    echo ""
    echo "========== Scoring hard-negative rollouts (all_negative) =========="
    ${PYTHON} -m probe_eval.evaluate \
        --experiment_folder ${PROBE_FOLDERS} \
        --rollout_dir ${HARD_NEG_ROLLOUT_DIRS[@]} \
        --all_negative \
        --behavior_threshold 5 \
        --skip_existing

    echo ""
    echo "========== Misalignment ground truth (dev) =========="
    TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"
    ${PYTHON} -m probe_eval.misalignment_ground_truth \
        --results-subdir ${RESULTS_SUBDIR} \
        --include-all-negative \
        --exclude-behaviors "${TEST_BEHAVIOR_PATTERNS}" \
        --output-suffix _dev

    echo ""
    echo "========== Misalignment ground truth (test) =========="
    ${PYTHON} -m probe_eval.misalignment_ground_truth \
        --results-subdir ${RESULTS_SUBDIR} \
        --include-all-negative \
        --include-behaviors "${TEST_BEHAVIOR_PATTERNS}" \
        --output-suffix _test

    echo ""
    echo "Complete at $(date)"
    echo "Results: probe_eval/results/${RESULTS_SUBDIR}/"
}

# ---- Local mode ----
if ${LOCAL_MODE}; then
    run_eval
    exit 0
fi

# ---- SLURM mode ----
# Job 1: Score rollouts (needs GPU)
EVAL_SCRIPT="${WORK_DIR}/eval_hard_neg.qsh"
cat <<EOF > "${EVAL_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_hard_neg
#SBATCH --output=${WORK_DIR}/eval_%j.out
#SBATCH --error=${WORK_DIR}/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=========================================="
echo "Hard-Negative Probe Evaluation"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

echo ""
echo "========== Scoring hard-negative rollouts =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${HARD_NEG_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --behavior_threshold 5 \\
    --skip_existing

echo ""
echo "Eval complete at \$(date)"
EOF

EVAL_JOB=$(sbatch --parsable "${EVAL_SCRIPT}")
echo "[1/2] Eval job ${EVAL_JOB} submitted (1 GPU)"

# Job 2: Recompute GT (no GPU, after eval)
GT_SCRIPT="${WORK_DIR}/gt_recompute.qsh"
cat <<EOF > "${GT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_gt_hard_neg
#SBATCH --output=${WORK_DIR}/gt_%j.out
#SBATCH --error=${WORK_DIR}/gt_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Recompute Misalignment Ground Truth"
echo "Started: \$(date)"
echo "=========================================="

TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"

echo ""
echo "========== Misalignment ground truth (dev) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

echo ""
echo "========== Misalignment ground truth (test) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test

echo ""
echo "GT recompute complete at \$(date)"
echo "Results: probe_eval/results/${RESULTS_SUBDIR}/"
EOF

GT_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} "${GT_SCRIPT}")
echo "[2/2] GT job ${GT_JOB} submitted (no GPU, after ${EVAL_JOB})"

echo ""
echo "========================================"
echo "Jobs Submitted"
echo "========================================"
echo "  Eval:    ${EVAL_JOB} (1 GPU)"
echo "  GT:      ${GT_JOB} (no GPU, after eval)"
echo "  Results: probe_eval/results/${RESULTS_SUBDIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
