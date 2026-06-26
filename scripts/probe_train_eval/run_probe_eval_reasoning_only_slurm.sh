#!/bin/bash
# SLURM script to evaluate reasoning-only trained probes on all rollout dirs,
# then compute per-turn indicator ground truth metrics.
#
# Step 1: Run evaluate.py with --reasoning_only on probes from
#          probe/probes/reasoning_only/
# Step 2: Run indicator_ground_truth.py scoped to the reasoning-only results
#
# Usage:
#   bash scripts/run_probe_eval_reasoning_only_slurm.sh

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python
PROBES_DIR=${BASE_DIR}/probe/probes/reasoning_only

# SLURM settings
JOB_NAME="probe_eval_reasoning_only"
PARTITION="general"
MEMORY="128G"

# Rollout directories
ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
)

# ---- Validate ----
if [ ! -d "${PROBES_DIR}" ]; then
    echo "ERROR: Probes directory not found: ${PROBES_DIR}"
    echo "Run scripts/run_probe_train_reasoning_only_slurm.sh first."
    exit 1
fi

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_eval_reasoning_only_${timestamp}"
mkdir -p "${WORK_DIR}"

# Collect all experiment folders
ALL_PROBE_FOLDERS=$(find "${PROBES_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
NUM_PROBES=$(echo "${ALL_PROBE_FOLDERS}" | wc -w)
echo "Found ${NUM_PROBES} probes, ${#ROLLOUT_DIRS[@]} rollout dirs"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/eval.qsh"

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
echo "Reasoning-Only Probe Evaluation"
echo "Probes: ${NUM_PROBES}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Started: \$(date)"
echo "=========================================="

# Step 1: Evaluate probes (reasoning tokens only)
echo ""
echo "--- Step 1: Running probe evaluation ---"
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder ${ALL_PROBE_FOLDERS} \\
    --rollout_dir ${ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --reasoning_only

# Step 2: Compute per-turn indicator ground truth metrics
echo ""
echo "--- Step 2: Computing indicator ground truth metrics ---"
${PYTHON} -m probe_eval.indicator_ground_truth \\
    --results-subdir reasoning_only

echo ""
echo "Complete at \$(date)"
EOF

# ---- Submit ----
sbatch "${SLURM_SCRIPT}"

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  Probes:      ${NUM_PROBES} (reasoning-only)"
echo "  Rollouts:    ${#ROLLOUT_DIRS[@]}"
echo "  Model loads: 1"
echo ""
echo "Results will be saved to:"
echo "  probe_eval/results/reasoning_only/"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/eval_*.out"
echo "=========================================="
