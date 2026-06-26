#!/bin/bash
# =============================================================================
# v2.6 Combined Probe Evaluation — Deferred (both variants)
# =============================================================================
#
# Submits 4 SLURM eval jobs with --begin=now+NHOURS (default 1.5h):
#   A-dev: variant A (reasoning-positive-only) × dev bloom + OOD
#   A-test: variant A × test bloom
#   B-dev: variant B (reasoning-only) × dev bloom + OOD
#   B-test: variant B × test bloom
#
# Usage:
#   bash scripts/probe_train_eval/run_v2_6_combined_eval_deferred.sh [--hours 1.5] [--batch-size 2]
# =============================================================================

set -e

HOURS=1.5
BATCH_SIZE=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours) HOURS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

PARTITION="general,overflow"
QOS="high"
CACHE_MEM="128G"

# ---- Probe variants ----
declare -A PROBE_DIRS
PROBE_DIRS[A]="${BASE_DIR}/probe/probes/v4_v2_6_combined_span"
PROBE_DIRS[B]="${BASE_DIR}/probe/probes/v4_v2_6_combined_ronly_span"

declare -A PROBE_LABELS
PROBE_LABELS[A]="reasoning-positive-only"
PROBE_LABELS[B]="reasoning-only"

# ---- Dev bloom (5 behaviors × 2 = 10 dirs) ----
DEV_POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-covert-code-sabotage_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-strategic-sandbagging_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glm_4_7_flash"
)
DEV_BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-covert-code-sabotage_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-strategic-sandbagging_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
)

# ---- Test bloom (5 behaviors × 2 = 10 dirs) ----
TEST_POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-covert-code-sabotage_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-strategic-sandbagging_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_glm_4_7_flash"
)
TEST_BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-covert-code-sabotage_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-strategic-sandbagging_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_benign_glm_4_7_flash"
)

# ---- OOD test (5 dirs) ----
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/mask/rollouts/glm-4.7-flash/mask"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer_filtered"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback_filtered"
)

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/v2_6_combined_eval_deferred_${timestamp}"
mkdir -p "${WORK_DIR}"

# SLURM --begin doesn't accept fractional hours; convert to minutes
MINUTES=$(python3 -c "print(int(float('${HOURS}') * 60))")
BEGIN_TIME="now+${MINUTES}minutes"

submit_eval() {
    local variant=$1         # "A" or "B"
    local suite=$2           # "dev" or "test"
    local probe_dir="${PROBE_DIRS[$variant]}"
    local probe_label="${PROBE_LABELS[$variant]}"
    local script="${WORK_DIR}/eval_${variant}_${suite}.qsh"

    cat > "${script}" <<SLURM
#!/bin/bash
#SBATCH --job-name=probe_eval_v26c_${variant}_${suite}
#SBATCH --output=${WORK_DIR}/${variant}_${suite}_%j.out
#SBATCH --error=${WORK_DIR}/${variant}_${suite}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "v2.6 Combined Probe Evaluation"
echo "Variant:    ${variant} (${probe_label})"
echo "Suite:      ${suite}"
echo "Probe dir:  ${probe_dir}"
echo "Batch size: ${BATCH_SIZE}"
echo "Started:    \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${probe_dir}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

SLURM

    if [[ "$suite" == "dev" ]]; then
        cat >> "${script}" <<SLURM
echo ""
echo "========== Dev: positive bloom =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== Dev: benign bloom (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== OOD test =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EVAL_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}
SLURM
    else
        # test suite
        cat >> "${script}" <<SLURM
echo ""
echo "========== Test: positive bloom =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== Test: benign bloom (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}
SLURM
    fi

    cat >> "${script}" <<SLURM

echo ""
echo "Variant ${variant} ${suite} evaluation complete at \$(date)"
SLURM

    local JID=$(sbatch --parsable --begin="${BEGIN_TIME}" "${script}")
    echo "[${variant} ${suite}] job ${JID} (begins ${BEGIN_TIME})"
}

for v in A B; do
    submit_eval "$v" "dev"
    submit_eval "$v" "test"
done

echo ""
echo "========================================"
echo "Deferred Eval Jobs Submitted"
echo "========================================"
echo "  Variant A: ${PROBE_DIRS[A]}"
echo "  Variant B: ${PROBE_DIRS[B]}"
echo "  Start:     ${BEGIN_TIME}  (approx $(date -d "+${HOURS} hours" 2>/dev/null || date))"
echo "  Batch:     ${BATCH_SIZE}"
echo "  Coverage:  10 dev bloom + 10 test bloom + 5 OOD test"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out  (after jobs start)"
echo "========================================"
