#!/bin/bash
# Evaluate a trained deception probe on multiple rollout datasets.
# Supports both SLURM submission and local execution.
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_eval_slurm.sh <experiment_folder> [--local]
#
# Examples:
#   # Submit via SLURM:
#   bash scripts/probe_train_eval/run_probe_eval_slurm.sh deception-detection/results/my_probe
#
#   # Run locally (no SLURM):
#   bash scripts/probe_train_eval/run_probe_eval_slurm.sh deception-detection/results/my_probe --local

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

EXPERIMENT_FOLDER=${1:?Usage: bash scripts/probe_train_eval/run_probe_eval_slurm.sh <experiment_folder> [--local]}
LOCAL_MODE=false
[[ "${2}" == "--local" ]] && LOCAL_MODE=true

# SLURM settings
JOB_NAME="probe_eval"
NUM_GPUS=1
PARTITION="general"
MEMORY="128G"

# Rollout directories to evaluate
ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/sandbagging"
    "${BASE_DIR}/bloom/bloom-results/undermining_oversight"
    "${BASE_DIR}/deception-detection/data/rollouts/insider_trading__onpolicy__glm-9b-flash_bloom"
    "${BASE_DIR}/deception-detection/data/rollouts/roleplaying__plain__glm-9b-flash_bloom"
)

# ---- Local mode: run directly ----
if ${LOCAL_MODE}; then
    cd "${BASE_DIR}"
    export HF_HOME=/workspace-vast/pretrained_ckpts
    for ROLLOUT_DIR in "${ROLLOUT_DIRS[@]}"; do
        NAME=$(basename "${ROLLOUT_DIR}")
        echo "========================================"
        echo "Evaluating on: ${NAME}"
        echo "========================================"
        ${PYTHON} -m probe_eval.evaluate \
            --experiment_folder "${EXPERIMENT_FOLDER}" \
            --rollout_dir "${ROLLOUT_DIR}" \
            --behavior_threshold 5
        echo ""
    done
    exit 0
fi

# ---- SLURM mode ----
PROBE_NAME=$(basename "${EXPERIMENT_FOLDER}")
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_eval_${PROBE_NAME}_${timestamp}"
mkdir -p "${WORK_DIR}"

SLURM_SCRIPT="${WORK_DIR}/probe_eval.qsh"

cat <<EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_${PROBE_NAME}
#SBATCH --output=${WORK_DIR}/probe_eval_%j.out
#SBATCH --error=${WORK_DIR}/probe_eval_%j.err
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================="
echo "Probe Evaluation"
echo "=========================================="
echo "Job started at: \$(date)"
echo "Experiment: ${EXPERIMENT_FOLDER}"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="
echo ""

EOF

for ROLLOUT_DIR in "${ROLLOUT_DIRS[@]}"; do
    NAME=$(basename "${ROLLOUT_DIR}")
    cat <<EOF >> "${SLURM_SCRIPT}"
echo "========================================"
echo "Evaluating on: ${NAME}"
echo "========================================"
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder "${EXPERIMENT_FOLDER}" \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --behavior_threshold 5
echo ""

EOF
done

cat <<'EOF' >> "${SLURM_SCRIPT}"
echo "=========================================="
echo "All evaluations complete!"
echo "Job finished at: $(date)"
echo "=========================================="
EOF

sbatch "${SLURM_SCRIPT}"

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  Probe:     ${PROBE_NAME}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Rollouts:  ${#ROLLOUT_DIRS[@]} datasets"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/probe_eval_*.out"
echo "=========================================="
