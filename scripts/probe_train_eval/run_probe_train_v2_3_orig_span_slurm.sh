#!/bin/bash
# Train v2.3 probes with span-level labels on ORIGINAL v2_3 generated data
# (all 25 indicators).
#
# Uses --label-mode span: positive = tokens matching indicator span text,
#                         negative = all other assistant tokens
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_train_v2_3_orig_span_slurm.sh

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=span

# Input: original v2_3 generated data (all 25 indicators)
DATA_DIR=${BASE_DIR}/probe/data/v2_3

# Output: probes trained on original data with span labels (with val split thresholds)
PROBE_DIR=${BASE_DIR}/probe/probes/v2_3_span_v2

# Training params
REG_COEFF=10.0
DETECT_LAYERS="27 28 29 30"

# SLURM settings
PARTITION="general"
MEMORY="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_train_v2_3_orig_span_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/train.qsh"

cat <<EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_v2_3_orig_span
#SBATCH --output=${WORK_DIR}/train_%j.out
#SBATCH --error=${WORK_DIR}/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Training v2.3 Probes (span mode, original data)"
echo "Label mode: ${LABEL_MODE}"
echo "Detect layers: ${DETECT_LAYERS}"
echo "Data dir: ${DATA_DIR}"
echo "Probe dir: ${PROBE_DIR}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.train \\
    --indicator-set v2_3 \\
    --all \\
    --label-mode ${LABEL_MODE} \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --data-dir ${DATA_DIR} \\
    --output-dir ${PROBE_DIR} \\
    --val-fraction 0.2

echo ""
echo "Complete at \$(date)"
echo "  Data:   ${DATA_DIR}"
echo "  Probes: ${PROBE_DIR}"
EOF

sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Training Job Submitted"
echo "========================================"
echo "  Label mode:     ${LABEL_MODE}"
echo "  Detect layers:  ${DETECT_LAYERS}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Probe dir:      ${PROBE_DIR}"
echo "  All indicators: yes (25)"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/train_*.out"
echo "========================================"
