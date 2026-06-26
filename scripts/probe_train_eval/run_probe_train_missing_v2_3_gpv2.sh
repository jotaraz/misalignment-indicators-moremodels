#!/bin/bash
# Train the 3 missing v2_3_gen_prompt_v2 probes that failed during the initial
# training run (likely OOM during warmup when two indicators ran concurrently).
#
# Missing indicators:
#   - Strategic Attention Manipulation
#   - Reward Gaming
#   - Misalignment Rationalization
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_train_missing_v2_3_gpv2.sh

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=turn

# Data and probe output directories (same as original run)
DATA_DIR=${BASE_DIR}/probe/data/v2_3_gen_prompt_v2
PROBE_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2

# Training params (same as original run)
REG_COEFF=10.0
DETECT_LAYERS="27 28 29 30"

# SLURM settings
PARTITION="general"
MEMORY="128G"

# The 3 indicators that failed to train
INDICATORS=(
    "Strategic Attention Manipulation"
    "Reward Gaming"
    "Misalignment Rationalization"
)

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_train_missing_v2_3_gpv2_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/train_missing.qsh"

cat <<'SLURM_EOF' > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_missing
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
echo "Training Missing v2_3_gen_prompt_v2 Probes"
echo "Label mode: LABEL_MODE_PLACEHOLDER"
echo "Detect layers: DETECT_LAYERS_PLACEHOLDER"
echo "Data dir: DATA_DIR_PLACEHOLDER"
echo "Probe dir: PROBE_DIR_PLACEHOLDER"
echo "Started: $(date)"
echo "=========================================="

SLURM_EOF

# Replace placeholders
sed -i "s|WORK_DIR_PLACEHOLDER|${WORK_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|PARTITION_PLACEHOLDER|${PARTITION}|g" "${SLURM_SCRIPT}"
sed -i "s|MEMORY_PLACEHOLDER|${MEMORY}|g" "${SLURM_SCRIPT}"
sed -i "s|BASE_DIR_PLACEHOLDER|${BASE_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|LABEL_MODE_PLACEHOLDER|${LABEL_MODE}|g" "${SLURM_SCRIPT}"
sed -i "s|DETECT_LAYERS_PLACEHOLDER|${DETECT_LAYERS}|g" "${SLURM_SCRIPT}"
sed -i "s|DATA_DIR_PLACEHOLDER|${DATA_DIR}|g" "${SLURM_SCRIPT}"
sed -i "s|PROBE_DIR_PLACEHOLDER|${PROBE_DIR}|g" "${SLURM_SCRIPT}"

# Append training commands for each missing indicator
for ind in "${INDICATORS[@]}"; do
    cat <<EOF >> "${SLURM_SCRIPT}"

echo ""
echo "--- Training: ${ind} ---"
${PYTHON} -m probe.train \\
    --indicator-set v2_3 \\
    --indicator "${ind}" \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${PROBE_DIR}"
EOF
done

cat <<EOF >> "${SLURM_SCRIPT}"

echo ""
echo "Complete at \$(date)"
echo "  Data:   ${DATA_DIR}"
echo "  Probes: ${PROBE_DIR}"
EOF

sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Training Job Submitted (missing probes)"
echo "========================================"
echo "  Indicators:"
for ind in "${INDICATORS[@]}"; do
    echo "    - ${ind}"
done
echo ""
echo "  Data dir:    ${DATA_DIR}"
echo "  Probe dir:   ${PROBE_DIR}"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/train_*.out"
echo "========================================"
