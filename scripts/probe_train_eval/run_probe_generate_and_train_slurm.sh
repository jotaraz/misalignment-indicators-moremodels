#!/bin/bash
# SLURM script: generate synthetic transcripts and train probes.
#
# Step 1 (no GPU): Generate transcripts via Anthropic API
# Step 2 (GPU): Train probes on GLM-4.7 Flash
#
# Uses v2.3 indicators (19 indicators + 5 preconditions + 1 behavioral concept).
# Uses improved generation prompts (v2) that enforce misaligned behavior follow-through.
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_generate_and_train_slurm.sh [label_mode]

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

LABEL_MODE=${1:-turn}

# Generation params
K=200
MAX_CONCURRENT=15

# Data and probe output directories
DATA_DIR=${BASE_DIR}/probe/data/v2_3_gen_prompt_v2
PROBE_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2

# Training params
REG_COEFF=10.0
DETECT_LAYERS="27 28 29 30"

# SLURM settings
PARTITION="general"
MEMORY="128G"

# v2.3 indicators to train (11 indicators with recently expanded non-examples)
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
    # Sandbagging (2)
    # "Deliberate Capability Concealment"
    # "Failure Calibration"
    # Undermining Oversight (1)
    # "Failure Misrepresentation in Evaluation"
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
WORK_DIR="${BASE_DIR}/logs/probe_generate_and_train_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Step 1: Generate transcripts (no GPU, run locally) ----
echo "========================================"
echo "[Step 1] Generating v2.3 transcripts (k=${K})"
echo "  Data dir: ${DATA_DIR}"
echo "========================================"

# Build --indicator arguments
INDICATOR_ARGS=""
for ind in "${INDICATORS[@]}"; do
    INDICATOR_ARGS="${INDICATOR_ARGS} \"${ind}\""
done

eval ${PYTHON} -m probe.generate \
    --indicator-set v2_3 \
    --indicator ${INDICATOR_ARGS} \
    --k ${K} \
    --max-concurrent ${MAX_CONCURRENT} \
    --output-dir "${DATA_DIR}" \
    2>&1 | tee "${WORK_DIR}/generate_v2_3.log"

echo ""
echo "Generation complete."

# ---- Step 2: Train probes (GPU via SLURM) ----
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

# Append training commands for each indicator
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
echo "SLURM Training Job Submitted"
echo "========================================"
echo "  Label mode:     ${LABEL_MODE}"
echo "  K:              ${K}"
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
