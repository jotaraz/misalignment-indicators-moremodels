#!/bin/bash
# Train v2.3 probes with span-level labels on existing v2 generated data.
#
# Uses --label-mode span: positive = tokens matching indicator span text,
#                         negative = all other assistant tokens
#
# This should give more precise learning signal by using labeled spans
# as positives rather than whole turns.
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_train_v2_3_span_slurm.sh

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=span

# Input: existing generated data from prompt v2
DATA_DIR=${BASE_DIR}/probe/data/v2_3_gen_prompt_v2

# Output: separate probe dir for span-trained probes (with val split thresholds)
PROBE_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2

# Training params
REG_COEFF=10.0
DETECT_LAYERS="27 28 29 30"

# SLURM settings
PARTITION="general"
MEMORY="128G"

# v2.3 misalignment indicators (same 16 as gen_prompt_v2)
INDICATORS=(
    # Deception (4)
    "Observer Suspicion Modeling"
    "Deceptive Content Planning"
    "Attribution Management"
    "Strategic Attention Manipulation"
    # Reward Hacking (3)
    "Recognized-Problem Suppression"
    "Manufactured Agreement Shift"
    "Reward Gaming"
    # Sabotage (2)
    "Action Concealment"
    "Unauthorized Consequential Action Planning"
    # Self-Preservation (2)
    "Self-Preserving Persuasion"
    "Preemptive Self-Preserving Action"
    # Cross-cutting (5)
    "Resource or Access Acquisition"
    "Misalignment Rationalization"
    "Internal-Output Divergence"
    "Adversarial User Framing"
    "Oversight as Obstacle"
)

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_train_v2_3_span_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Build SLURM script ----
SLURM_SCRIPT="${WORK_DIR}/train.qsh"

cat <<'SLURM_EOF' > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_span
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
echo "Training v2.3 Probes (span mode)"
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

# Build indicator args for a single invocation
INDICATOR_ARGS=""
for ind in "${INDICATORS[@]}"; do
    INDICATOR_ARGS="${INDICATOR_ARGS} \"${ind}\""
done

# Single training command: model loaded once, all indicators trained sequentially
cat <<EOF >> "${SLURM_SCRIPT}"

echo ""
echo "--- Training all ${#INDICATORS[@]} indicators in a single invocation ---"
${PYTHON} -m probe.train \\
    --indicator-set v2_3 \\
    --indicator ${INDICATOR_ARGS} \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${PROBE_DIR}" \\
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
echo "  Indicators:     ${#INDICATORS[@]}"
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
