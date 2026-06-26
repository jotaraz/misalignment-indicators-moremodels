#!/bin/bash
# Recompute val thresholds for trained probes with all 4 metric sets.
#
# Usage:
#   bash scripts/probe_train_eval/run_recompute_thresholds_slurm.sh

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

PROBE_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2
DATA_DIR=${BASE_DIR}/probe/data/v2_3_gen_prompt_v2

# SLURM settings
PARTITION="general"
MEMORY="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/recompute_thresholds_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/recompute.qsh"

cat <<EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=recompute_thresholds
#SBATCH --output=${WORK_DIR}/recompute_%j.out
#SBATCH --error=${WORK_DIR}/recompute_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Recomputing val thresholds"
echo "Probe dir: ${PROBE_DIR}"
echo "Data dir:  ${DATA_DIR}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.recompute_thresholds \\
    --probe-dir "${PROBE_DIR}" \\
    --data-dir "${DATA_DIR}" \\
    --val-fraction 0.2

echo ""
echo "Complete at \$(date)"
EOF

sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Recompute Thresholds Job Submitted"
echo "========================================"
echo "  Probe dir: ${PROBE_DIR}"
echo "  Data dir:  ${DATA_DIR}"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/recompute_*.out"
echo "========================================"
