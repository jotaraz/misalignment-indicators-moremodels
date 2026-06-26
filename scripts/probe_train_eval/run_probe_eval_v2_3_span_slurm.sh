#!/bin/bash
# SLURM script to evaluate v2_3 span-trained probes on:
#   1. Positive data (normal evaluation)
#   2. Benign datasets (all_negative: sycophancy_benign, sandbagging_benign, etc.)
# Then compute indicator and misalignment ground-truth metrics including
# all-negative data for false positive rate analysis, and generate visualization.
#
# Usage:
#   bash scripts/probe_train_eval/run_probe_eval_v2_3_span_slurm.sh <probes_dir> [results_subdir]
#
# Examples:
#   # Evaluate original (non-cleaned) probes:
#   bash scripts/probe_train_eval/run_probe_eval_v2_3_span_slurm.sh probe/probes/v2_3_gen_prompt_v2_span
#
#   # Evaluate PCA-cleaned probes:
#   bash scripts/probe_train_eval/run_probe_eval_v2_3_span_slurm.sh probe/probes/v2_3_gen_prompt_v2_span_clean
#
#   # Custom results subdir:
#   bash scripts/probe_train_eval/run_probe_eval_v2_3_span_slurm.sh probe/probes/v2_3_gen_prompt_v2_span my_results

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

# Parse arguments
PROBES_DIR_REL=${1:?Usage: bash scripts/probe_train_eval/run_probe_eval_v2_3_span_slurm.sh <probes_dir> [results_subdir]}
PROBES_DIR="${BASE_DIR}/${PROBES_DIR_REL}"

# Default results_subdir = basename of probes_dir
# NOTE: evaluate.py derives output path from probe folder structure, so this
# MUST match the top-level probe dir basename for steps 3-5 to find results.
RESULTS_SUBDIR=${2:-$(basename "${PROBES_DIR}")}

# Derive job name from results subdir
JOB_NAME="probe_eval_${RESULTS_SUBDIR}"

# SLURM settings
PARTITION="general"
MEMORY="128G"

# Positive rollout directories
POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glmflash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_glm_4_7_flash"
)

# Benign (all-negative) rollout directories
BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
)

# ---- Validate ----
if [ ! -d "${PROBES_DIR}" ]; then
    echo "ERROR: Probes directory not found: ${PROBES_DIR}"
    exit 1
fi

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/${JOB_NAME}_${timestamp}"
mkdir -p "${WORK_DIR}"

# Collect probe folders (each has cfg.yaml + detector.pt)
PROBE_FOLDERS=$(find "${PROBES_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
NUM_PROBES=$(echo "${PROBE_FOLDERS}" | wc -w)

echo "Probes dir: ${PROBES_DIR}"
echo "Results subdir: ${RESULTS_SUBDIR}"
echo "Found ${NUM_PROBES} probes"
echo "Positive rollout dirs: ${#POSITIVE_ROLLOUT_DIRS[@]}"
echo "Benign rollout dirs: ${#BENIGN_ROLLOUT_DIRS[@]}"

# ---- Build SLURM script ----
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
echo "Probe Evaluation - ${RESULTS_SUBDIR}"
echo "Probes: ${NUM_PROBES}"
echo "Positive rollouts: ${#POSITIVE_ROLLOUT_DIRS[@]}"
echo "Benign rollouts: ${#BENIGN_ROLLOUT_DIRS[@]}"
echo "Started: \$(date)"
echo "=========================================="

# ---- Step 1: Evaluate probes on positive data ----
echo ""
echo "========== Step 1: Evaluate on positive data =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder ${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5

# ---- Step 2: Evaluate probes on benign data (all negative) ----
echo ""
echo "========== Step 2: Evaluate on benign data (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder ${PROBE_FOLDERS} \\
    --rollout_dir ${BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative

# ---- Step 3: Indicator ground-truth metrics (v2.3, with all-negative) ----
echo ""
echo "========== Step 3: Indicator ground truth (v2.3) =========="
${PYTHON} -m probe_eval.indicator_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --indicator-gt v2.3 \\
    --include-all-negative

# ---- Step 4: Misalignment ground-truth metrics (with all-negative) ----
echo ""
echo "========== Step 4: Misalignment ground truth =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative

# ---- Step 5: Generate misalignment dashboard visualization ----
echo ""
echo "========== Step 5: Misalignment dashboard visualization =========="
${PYTHON} probe_eval/visualize_misalignment.py \\
    --results-subdir ${RESULTS_SUBDIR}

echo ""
echo "=========================================="
echo "Complete at \$(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Probe eval:       probe_eval/results/${RESULTS_SUBDIR}/"
echo "  Indicator GT:     probe_eval/results/${RESULTS_SUBDIR}/indicator_gt_summary.json"
echo "  Misalignment GT:  probe_eval/results/${RESULTS_SUBDIR}/misalignment_gt_summary.json"
echo "  Dashboard:        probe_eval/results/${RESULTS_SUBDIR}/misalignment_dashboard.html"
EOF

# ---- Submit ----
sbatch "${SLURM_SCRIPT}"

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  Probes dir:    ${PROBES_DIR}"
echo "  Results:       ${RESULTS_SUBDIR}"
echo "  Num probes:    ${NUM_PROBES}"
echo "  Positive:      ${#POSITIVE_ROLLOUT_DIRS[@]} rollouts"
echo "  Benign:        ${#BENIGN_ROLLOUT_DIRS[@]} rollouts"
echo ""
echo "Monitor:"
echo "  tail -f ${WORK_DIR}/eval_*.out"
echo "=========================================="
