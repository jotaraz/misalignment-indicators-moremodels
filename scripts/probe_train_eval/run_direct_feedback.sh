#!/bin/bash
# =============================================================================
# Direct Feedback: Use actual FP/FN transcripts as hard examples for retraining
# =============================================================================
#
# A simpler alternative to the full feedback loop.  Instead of Opus analysis +
# Sonnet generation, this directly adds the mis-scored bloom/neutral transcripts
# to the training set as hard negatives/positives:
#
#   Job 1 (no GPU):  Collect direct FP/FN examples + merge with original data
#   Job 2 (1 GPU):   Train probes on combined data
#   Job 3 (2 GPUs):  Evaluate probes on dev + held-out test sets
#
# Prerequisites:
#   - collected_errors.json from a feedback collect step
#   - Neutral FP scoring output
#   - Original training data in probe/data/{indicator_set}/
#
# Usage:
#   bash scripts/probe_train_eval/run_direct_feedback.sh [options]
# =============================================================================

set -e

# ---- Defaults ----
INDICATOR_SET="v2_4"
LABEL_MODE="span"
RUN_ID="direct_v1"
DETECT_LAYERS="27 28 29 30"
LAYER=30

# Input paths (from a prior feedback collect run)
COLLECTED_ERRORS=""
NEUTRAL_FPS=""
RESULTS_SUBDIR="v2_4_span"  # existing results used for error collection

# Caps per indicator
MAX_BLOOM_FP=100
MAX_BLOOM_FN=100
MAX_NEUTRAL_FP=100

# Skip flags
SKIP_COLLECT=false
SKIP_TRAIN=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --indicator-set) INDICATOR_SET="$2"; shift 2 ;;
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --layer) LAYER="$2"; shift 2 ;;
        --collected-errors) COLLECTED_ERRORS="$2"; shift 2 ;;
        --neutral-fps) NEUTRAL_FPS="$2"; shift 2 ;;
        --results-subdir) RESULTS_SUBDIR="$2"; shift 2 ;;
        --max-bloom-fp) MAX_BLOOM_FP="$2"; shift 2 ;;
        --max-bloom-fn) MAX_BLOOM_FN="$2"; shift 2 ;;
        --max-neutral-fp) MAX_NEUTRAL_FP="$2"; shift 2 ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

ORIGINAL_DATA_DIR=${BASE_DIR}/probe/data/${INDICATOR_SET}
COMBINED_DATA_DIR=${BASE_DIR}/probe/data/${INDICATOR_SET}_${RUN_ID}
COMBINED_PROBE_DIR=${BASE_DIR}/probe/probes/${INDICATOR_SET}_${RUN_ID}_${LABEL_MODE}
COMBINED_RESULTS_SUBDIR="${INDICATOR_SET}_${RUN_ID}_${LABEL_MODE}"

# Default collected errors path
if [ -z "${COLLECTED_ERRORS}" ]; then
    # Try the most recent feedback run
    COLLECTED_ERRORS=${BASE_DIR}/probe/data/feedback/${RESULTS_SUBDIR}_v3/collected_errors.json
fi
if [ -z "${NEUTRAL_FPS}" ]; then
    NEUTRAL_FPS=${BASE_DIR}/probe/data/neutral/false_positives_${RESULTS_SUBDIR}_filtered.json
fi
NEUTRAL_DIALOGUES=${BASE_DIR}/probe/data/neutral/dialogues_filtered_v2.json

REG_COEFF=10.0

# Bloom rollout directories
POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glmflash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_glm_4_7_flash"
)
BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/sycophancy_benign"
    "${BASE_DIR}/bloom/bloom-results/sandbagging_benign"
    "${BASE_DIR}/bloom/bloom-results/undermining_oversight_benign"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-info_benign"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
)
TEST_POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-backdoor__glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-bug_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-promotion_glm_4_7_flash"
)
TEST_BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-promotion_benign_glm_4_7_flash"
)
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_are_you_sure"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback"
)

# SLURM settings
PARTITION="general,overflow"
QOS="low"
MEMORY="200G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/direct_feedback_${timestamp}"
mkdir -p "${WORK_DIR}"

PREV_JOB=""

echo "========================================"
echo "Direct Feedback Pipeline"
echo "========================================"
echo "  Indicator set:        ${INDICATOR_SET}"
echo "  Label mode:           ${LABEL_MODE}"
echo "  Run ID:               ${RUN_ID}"
echo "  Collected errors:     ${COLLECTED_ERRORS}"
echo "  Neutral FPs:          ${NEUTRAL_FPS}"
echo "  Max bloom FP/FN/neutral: ${MAX_BLOOM_FP}/${MAX_BLOOM_FN}/${MAX_NEUTRAL_FP}"
echo "  Combined data dir:    ${COMBINED_DATA_DIR}"
echo "  Combined probe dir:   ${COMBINED_PROBE_DIR}"
echo "  Results subdir:       ${COMBINED_RESULTS_SUBDIR}"
echo "========================================"

# ======================================================================
# Job 1: Collect direct examples + merge (no GPU)
# ======================================================================
if [ "${SKIP_COLLECT}" = false ]; then
    COLLECT_SCRIPT="${WORK_DIR}/collect.qsh"
    cat <<EOF > "${COLLECT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=df_collect
#SBATCH --output=${WORK_DIR}/collect_%j.out
#SBATCH --error=${WORK_DIR}/collect_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 1: Collect direct FP/FN examples"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.collect_direct_examples \\
    --collected-errors "${COLLECTED_ERRORS}" \\
    --neutral-fps "${NEUTRAL_FPS}" \\
    --neutral-dialogues "${NEUTRAL_DIALOGUES}" \\
    --original-data-dir "${ORIGINAL_DATA_DIR}" \\
    --output-dir "${COMBINED_DATA_DIR}" \\
    --max-bloom-fp ${MAX_BLOOM_FP} \\
    --max-bloom-fn ${MAX_BLOOM_FN} \\
    --max-neutral-fp ${MAX_NEUTRAL_FP}

echo ""
echo "Collection complete at \$(date)"
EOF

    COLLECT_JOB=$(sbatch --parsable "${COLLECT_SCRIPT}")
    echo "[1/3] Collect job ${COLLECT_JOB} submitted (no GPU)"
    PREV_JOB="${COLLECT_JOB}"
else
    echo "[1/3] Skipping collection (--skip-collect)"
fi

# ======================================================================
# Job 2: Train probes (1 GPU)
# ======================================================================
if [ "${SKIP_TRAIN}" = false ]; then
    TRAIN_SCRIPT="${WORK_DIR}/train.qsh"
    cat <<EOF > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=df_train
#SBATCH --output=${WORK_DIR}/train_%j.out
#SBATCH --error=${WORK_DIR}/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=64G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Stage 2: Train probes on combined data"
echo "Data dir: ${COMBINED_DATA_DIR}"
echo "Probe dir: ${COMBINED_PROBE_DIR}"
echo "Started: \$(date)"
echo "=========================================="

for DATA_FILE in "${COMBINED_DATA_DIR}"/*.json; do
    [ -f "\${DATA_FILE}" ] || continue
    INDICATOR_SLUG=\$(basename "\${DATA_FILE}" .json)
    # Skip stats file
    [[ "\${INDICATOR_SLUG}" == _* ]] && continue

    INDICATOR_NAME=\$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('indicator_name',''))" "\${DATA_FILE}" 2>/dev/null)
    if [ -z "\${INDICATOR_NAME}" ]; then
        echo "  Skipping \${INDICATOR_SLUG}: could not read indicator_name"
        continue
    fi

    echo ""
    echo "--- Training: \${INDICATOR_NAME} ---"
    ${PYTHON} -m probe.train \\
        --indicator-set ${INDICATOR_SET} \\
        --indicator "\${INDICATOR_NAME}" \\
        --label-mode "${LABEL_MODE}" \\
        --reg-coeff ${REG_COEFF} \\
        --detect-layers ${DETECT_LAYERS} \\
        --reasoning-only \\
        --threshold-mode sentence \\
        --val-fraction 0.2 \\
        --data-dir "${COMBINED_DATA_DIR}" \\
        --output-dir "${COMBINED_PROBE_DIR}"
done

echo ""
echo "Training complete at \$(date)"
EOF

    if [ -n "${PREV_JOB}" ]; then
        TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${TRAIN_SCRIPT}")
        echo "[2/3] Train job ${TRAIN_JOB} submitted (1 GPU, after ${PREV_JOB})"
    else
        TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
        echo "[2/3] Train job ${TRAIN_JOB} submitted (1 GPU)"
    fi
    PREV_JOB="${TRAIN_JOB}"
else
    echo "[2/3] Skipping training (--skip-train)"
fi

# ======================================================================
# Job 3: Evaluate (2 GPUs)
# ======================================================================
EVAL_SCRIPT="${WORK_DIR}/eval.qsh"

cat <<EOF > "${EVAL_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=df_eval
#SBATCH --output=${WORK_DIR}/eval_%j.out
#SBATCH --error=${WORK_DIR}/eval_%j.err
#SBATCH --gres=gpu:2
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Stage 3: Evaluate probes"
echo "Probe dir: ${COMBINED_PROBE_DIR}"
echo "Results:   probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
NUM_PROBES=\$(echo "\${PROBE_FOLDERS}" | wc -w)
echo "Found \${NUM_PROBES} probes"

if [ "\${NUM_PROBES}" -eq 0 ]; then
    echo "ERROR: No probes found in ${COMBINED_PROBE_DIR}"
    exit 1
fi

# ---- Dev set: positive ----
echo ""
echo "========== Evaluate on positive data =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

# ---- Dev set: benign ----
echo ""
echo "========== Evaluate on benign data (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# Behavior filter patterns for dev/test split
TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"

# ---- Dev ground-truth metrics ----
echo ""
echo "========== Indicator ground truth (dev) =========="
${PYTHON} -m probe_eval.indicator_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --indicator-gt v2.3 \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

echo ""
echo "========== Misalignment ground truth (dev) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

# ---- Visualization ----
echo ""
echo "========== Visualization =========="
${PYTHON} probe_eval/visualize_misalignment.py \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR}

# ======================================================================
# Held-out test set evaluation
# ======================================================================
echo ""
echo "=========================================="
echo "Held-out Test Set Evaluation"
echo "=========================================="

echo ""
echo "========== Test set: bloom positive =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

echo ""
echo "========== Test set: bloom benign =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

echo ""
echo "========== Test set: OOD evals =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EVAL_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

echo ""
echo "========== Test set: misalignment ground truth =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test

echo ""
echo "=========================================="
echo "Evaluation complete at \$(date)"
echo "=========================================="
echo "Results:"
echo "  Probe eval:              probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "  Indicator GT (dev):      probe_eval/results/${COMBINED_RESULTS_SUBDIR}/indicator_gt_summary_dev.json"
echo "  Misalignment GT (dev):   probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_dev.json"
echo "  Misalignment GT (test):  probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_test.json"
echo "  Dashboard:               probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_dashboard.html"
EOF

if [ -n "${PREV_JOB}" ]; then
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${EVAL_SCRIPT}")
    echo "[3/3] Eval job ${EVAL_JOB} submitted (2 GPUs, after ${PREV_JOB})"
else
    EVAL_JOB=$(sbatch --parsable "${EVAL_SCRIPT}")
    echo "[3/3] Eval job ${EVAL_JOB} submitted (2 GPUs)"
fi

# ---- Summary ----
echo ""
echo "========================================"
echo "Direct Feedback Pipeline Submitted"
echo "========================================"
echo "  Indicator set:       ${INDICATOR_SET}"
echo "  Run ID:              ${RUN_ID}"
echo "  Label mode:          ${LABEL_MODE}"
echo "  Max FP/FN/neutral:   ${MAX_BLOOM_FP}/${MAX_BLOOM_FN}/${MAX_NEUTRAL_FP}"
echo ""
echo "  Combined data:       ${COMBINED_DATA_DIR}"
echo "  Retrained probes:    ${COMBINED_PROBE_DIR}"
echo "  Eval results:        probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u kaiwen"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
