#!/bin/bash
# SLURM script to train reasoning-only probes for all indicators.
# Trains probes using only reasoning tokens (inside <think>...</think>),
# saving to probe/probes/reasoning_only/ instead of the default.
#
# Usage:
#   bash scripts/run_probe_train_reasoning_only_slurm.sh [label_mode]
#
# Examples:
#   bash scripts/run_probe_train_reasoning_only_slurm.sh        # default: turn mode
#   bash scripts/run_probe_train_reasoning_only_slurm.sh span    # span mode

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python
OUTPUT_DIR=${BASE_DIR}/probe/probes/reasoning_only

LABEL_MODE=${1:-turn}
REG_COEFF=10.0

# SLURM settings
JOB_NAME="probe_train_reasoning_only"
PARTITION="general"
MEMORY="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_train_reasoning_only_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/train.qsh"

cat <<EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${WORK_DIR}/train_%j.out
#SBATCH --error=${WORK_DIR}/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=========================================="
echo "Training Reasoning-Only Probes"
echo "Label mode: ${LABEL_MODE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.train \\
    --all \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --reasoning-only \\
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Complete at \$(date)"
echo "Trained probes saved to ${OUTPUT_DIR}"
EOF

# ---- Submit ----
sbatch "${SLURM_SCRIPT}"

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  Label mode:  ${LABEL_MODE}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Reasoning:   only (no final response)"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/train_*.out"
echo "=========================================="
