#!/bin/bash
# Evaluate ALL trained probes (finegrain + behavior-level) on all rollout dirs,
# then compute indicator ground-truth metrics.
# Supports both SLURM submission and local execution.
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_eval_all_v2_slurm.sh [--local]

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LOCAL_MODE=false
[[ "${1}" == "--local" ]] && LOCAL_MODE=true

# SLURM settings
JOB_NAME="probe_eval_all_v2"
PARTITION="general"
MEMORY="128G"

# Rollout directories to evaluate on
ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
)

# Probe directories
FINEGRAIN_PROBES_DIR=${BASE_DIR}/probe/probes/finegrain
BEHAVIOR_PROBES_DIR=${BASE_DIR}/probe/probes/behavior

# Collect probe folders
FINEGRAIN_FOLDERS=$(find "${FINEGRAIN_PROBES_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
BEHAVIOR_FOLDERS=$(find "${BEHAVIOR_PROBES_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
NUM_FINEGRAIN=$(echo "${FINEGRAIN_FOLDERS}" | wc -w)
NUM_BEHAVIOR=$(echo "${BEHAVIOR_FOLDERS}" | wc -w)

echo "Found ${NUM_FINEGRAIN} finegrain probes, ${NUM_BEHAVIOR} behavior probes"
echo "Rollout dirs: ${#ROLLOUT_DIRS[@]}"

# ---- Shared eval logic ----
run_eval() {
    echo ""
    echo "========== Step 1: Finegrain probes =========="
    ${PYTHON} -m probe_eval.evaluate \
        --experiment_folder ${FINEGRAIN_FOLDERS} \
        --rollout_dir ${ROLLOUT_DIRS[@]} \
        --behavior_threshold 5

    echo ""
    echo "========== Step 2: Behavior-level probes =========="
    ${PYTHON} -m probe_eval.evaluate \
        --experiment_folder ${BEHAVIOR_FOLDERS} \
        --rollout_dir ${ROLLOUT_DIRS[@]} \
        --behavior_threshold 5

    echo ""
    echo "========== Step 3: Indicator ground truth (finegrain) =========="
    ${PYTHON} -m probe_eval.indicator_ground_truth --results-subdir finegrain

    echo ""
    echo "========== Step 4: Indicator ground truth (behavior) =========="
    ${PYTHON} -m probe_eval.indicator_ground_truth --results-subdir behavior

    echo ""
    echo "Complete at $(date)"
}

# ---- Local mode ----
if ${LOCAL_MODE}; then
    cd "${BASE_DIR}"
    export HF_HOME=/workspace-vast/pretrained_ckpts
    run_eval
    exit 0
fi

# ---- SLURM mode ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_eval_all_v2_${timestamp}"
mkdir -p "${WORK_DIR}"

SLURM_SCRIPT="${WORK_DIR}/probe_eval.qsh"

cat <<EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${WORK_DIR}/eval_%j.out
#SBATCH --error=${WORK_DIR}/eval_%j.err
#SBATCH --gres=gpu:2
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=========================================="
echo "Probe Evaluation - All Probes x All Rollouts"
echo "Finegrain probes: ${NUM_FINEGRAIN}"
echo "Behavior probes: ${NUM_BEHAVIOR}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Started: \$(date)"
echo "=========================================="

# ---- Step 1: Evaluate finegrain probes ----
echo ""
echo "========== Step 1: Finegrain probes =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder ${FINEGRAIN_FOLDERS} \\
    --rollout_dir ${ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5

# ---- Step 2: Evaluate behavior-level probes ----
echo ""
echo "========== Step 2: Behavior-level probes =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder ${BEHAVIOR_FOLDERS} \\
    --rollout_dir ${ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5

# ---- Step 3: Indicator ground truth metrics ----
echo ""
echo "========== Step 3: Indicator ground truth (finegrain) =========="
${PYTHON} -m probe_eval.indicator_ground_truth --results-subdir finegrain

echo ""
echo "========== Step 4: Indicator ground truth (behavior) =========="
${PYTHON} -m probe_eval.indicator_ground_truth --results-subdir behavior

echo ""
echo "Complete at \$(date)"
EOF

sbatch "${SLURM_SCRIPT}"

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  Finegrain probes: ${NUM_FINEGRAIN}"
echo "  Behavior probes:  ${NUM_BEHAVIOR}"
echo "  Rollouts:         ${#ROLLOUT_DIRS[@]}"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/eval_*.out"
echo "=========================================="
