#!/bin/bash
# =============================================================================
# v2.6 Probe Evaluation — Dev+OOD and Test (two parallel 1-GPU jobs)
# =============================================================================
#
# Submits:
#   Job 3a (1 GPU): dev bloom positive + benign, plus three OOD eval dirs
#   Job 3b (1 GPU): test bloom positive + benign (runs parallel to 3a)
#
# No ground-truth / visualization step (run separately when needed).
#
# Usage:
#   bash scripts/probe_train_eval/run_v2_6_v4_eval.sh [--label-mode span] [--batch-size 4]
# =============================================================================

set -e

LABEL_MODE="span"
BATCH_SIZE=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PROBE_DIR=${BASE_DIR}/probe/probes/v4_v2_6_${LABEL_MODE}

PARTITION="general,overflow"
QOS="high"
CACHE_MEM="128G"

# ---- Dev bloom (malicious + benign) — 5 behaviors x 2 ----
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

# ---- Test bloom (malicious + benign) — 5 behaviors x 2 ----
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

# ---- OOD eval dirs (treated as malicious — use judgment threshold) ----
# Note: mask/ has no judgment.json — skip until we decide how to label it.
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
)

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/v2_6_v4_eval_${timestamp}"
mkdir -p "${WORK_DIR}"

# ======================================================================
# Job 3a: Dev bloom + OOD
# ======================================================================
EVAL_DEV_SCRIPT="${WORK_DIR}/eval_dev.qsh"
cat <<EOF > "${EVAL_DEV_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_v26_dev
#SBATCH --output=${WORK_DIR}/eval_dev_%j.out
#SBATCH --error=${WORK_DIR}/eval_dev_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "v2.6 Probe Evaluation — Dev Bloom + OOD"
echo "Probe dir:  ${PROBE_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

# ---- Dev positive ----
echo ""
echo "========== Dev: positive bloom =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

# ---- Dev benign ----
echo ""
echo "========== Dev: benign bloom (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}

# ---- OOD evals ----
echo ""
echo "========== OOD evals =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EVAL_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "Dev + OOD evaluation complete at \$(date)"
EOF

EVAL_DEV_JOB=$(sbatch --parsable "${EVAL_DEV_SCRIPT}")
echo "[3a] Eval dev+OOD job ${EVAL_DEV_JOB} submitted (1 GPU)"

# ======================================================================
# Job 3b: Test bloom
# ======================================================================
EVAL_TEST_SCRIPT="${WORK_DIR}/eval_test.qsh"
cat <<EOF > "${EVAL_TEST_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_v26_test
#SBATCH --output=${WORK_DIR}/eval_test_%j.out
#SBATCH --error=${WORK_DIR}/eval_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "v2.6 Probe Evaluation — Test Bloom"
echo "Probe dir:  ${PROBE_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

# ---- Test positive ----
echo ""
echo "========== Test: positive bloom =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

# ---- Test benign ----
echo ""
echo "========== Test: benign bloom (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "Test evaluation complete at \$(date)"
EOF

EVAL_TEST_JOB=$(sbatch --parsable "${EVAL_TEST_SCRIPT}")
echo "[3b] Eval test job ${EVAL_TEST_JOB} submitted (1 GPU)"

# ---- Summary ----
echo ""
echo "========================================"
echo "SLURM Jobs Submitted"
echo "========================================"
echo "  Probe dir:  ${PROBE_DIR}"
echo "  Label mode: ${LABEL_MODE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Job 3a: dev bloom + OOD"
echo "  Job 3b: test bloom (parallel)"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
