#!/bin/bash
# Train v2.3 probes for all 14 indicators via SLURM (GPU).
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_train_all.sh [label_mode]
#
# Examples:
#   bash scripts/probe_train_eval/run_probe_train_all.sh        # default: turn mode
#   bash scripts/probe_train_eval/run_probe_train_all.sh span    # span mode

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=${1:-turn}
REG_COEFF=10.0

# SLURM settings
PARTITION="general"
MEMORY="128G"

# v2.3 indicators to train
INDICATORS=(
    "Recognized-Problem Suppression"
    "Manufactured Agreement Shift"
    "Reward Gaming"
    "Action Concealment"
    "Unauthorized Consequential Action Planning"
    "Internal-Output Divergence"
    "Adversarial User Framing"
    "Oversight as Obstacle"
    "Distinct Self-Goal Representation"
    "Evaluation Awareness"
    "Situational Awareness of Constraints"
    "Reward Awareness"
    "Threat Awareness"
    "Self-Consistency Monitoring"
)

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_train_v2_3_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/train.qsh"

cat <<'SLURM_EOF' > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_v2_3
#SBATCH --output=WORK_DIR_PLACEHOLDER/train_%j.out
#SBATCH --error=WORK_DIR_PLACEHOLDER/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --mem=MEMORY_PLACEHOLDER
#SBATCH --chdir=BASE_DIR_PLACEHOLDER

source BASE_DIR_PLACEHOLDER/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Training v2.3 Probes"
echo "Label mode: LABEL_MODE_PLACEHOLDER"
echo "Started: $(date)"
echo "=========================================="

SLURM_EOF

# Replace placeholders
sed -i "s|WORK_DIR_PLACEHOLDER|${WORK_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|PARTITION_PLACEHOLDER|${PARTITION}|g" "${SLURM_SCRIPT}"
sed -i "s|MEMORY_PLACEHOLDER|${MEMORY}|g" "${SLURM_SCRIPT}"
sed -i "s|BASE_DIR_PLACEHOLDER|${BASE_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|LABEL_MODE_PLACEHOLDER|${LABEL_MODE}|g" "${SLURM_SCRIPT}"

# Append training commands for each indicator
for ind in "${INDICATORS[@]}"; do
    cat <<EOF >> "${SLURM_SCRIPT}"

echo ""
echo "--- Training: ${ind} ---"
${PYTHON} -m probe.train \\
    --indicator-set v2_3 \\
    --indicator "${ind}" \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF}
EOF
done

cat <<EOF >> "${SLURM_SCRIPT}"

echo ""
echo "Complete at \$(date)"
echo "  Data:   ${BASE_DIR}/probe/data/v2_3/"
echo "  Probes: ${BASE_DIR}/probe/probes/v2_3/"
EOF

sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Training Job Submitted"
echo "========================================"
echo "  Label mode:  ${LABEL_MODE}"
echo "  Indicators:  ${#INDICATORS[@]}"
for ind in "${INDICATORS[@]}"; do
    echo "    - ${ind}"
done
echo ""
echo "  Data dir:    ${BASE_DIR}/probe/data/v2_3/"
echo "  Probe dir:   ${BASE_DIR}/probe/probes/v2_3/"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/train_*.out"
echo "========================================"
