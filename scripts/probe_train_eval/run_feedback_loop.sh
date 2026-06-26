#!/bin/bash
# =============================================================================
# Feedback Loop: OOD & Neutral Validation → Analysis → Generation → Retrain → Eval
# =============================================================================
#
# End-to-end pipeline for feedback-driven probe training data augmentation:
#   Job 1 (1 GPU):  Score neutral validation set + collect OOD/neutral errors
#   Job 2 (no GPU): Multi-agent Opus 4.6 analysis of error patterns
#   Job 3 (no GPU): Sonnet 4.6 data generation + merge with original data
#   Job 4 (1 GPU):  Retrain probes on combined data
#   Job 5 (2 GPUs): Evaluate retrained probes on bloom + neutral FP scoring
#
# Original training data is NEVER modified — all output goes to new directories:
#   Combined data:   probe/data/{indicator_set}_combined/
#   Retrained probes: probe/probes/{indicator_set}_combined_{label_mode}/
#   Eval results:    probe_eval/results/{indicator_set}_combined_{label_mode}/
#
# Prerequisites:
#   - Probes trained (e.g., via run_v2_4_existing_pipeline.sh)
#   - Bloom eval results exist in probe_eval/results/{results_subdir}/
#   - Neutral dialogues exist at probe/data/neutral/dialogues_filtered_v2.json
#
# Usage:
#   bash scripts/probe_train_eval/run_feedback_loop.sh [options]
#
# Options:
#   --results-subdir DIR     Existing results subdir to analyze (default: v2_4_span)
#   --indicator-set SET      v2_3 or v2_4 (default: v2_4)
#   --label-mode MODE        turn or span (default: span)
#   --run-id ID              Version tag for output paths (default: v2)
#   --k-per-scenario N       Transcripts per scenario (default: 5)
#   --max-concurrent N       Max concurrent API calls (default: 10)
#   --detect-layers L...     Layers to probe (default: "27 28 29 30")
#   --skip-neutral-score     Skip neutral scoring (use pre-scored FPs)
#   --skip-collect           Skip error collection (use existing collected_errors.json)
#   --skip-analyze           Skip error analysis (use existing analysis_suggestions.json)
#   --skip-generate          Skip data generation + merge
#   --skip-train             Skip retraining
#   --neutral-fps PATH       Path to pre-scored neutral FPs JSON
#   --layer N                Primary probe layer for thresholds (default: 30)
# =============================================================================

set -e

# ---- Defaults ----
RESULTS_SUBDIR="v2_4_span"
INDICATOR_SET="v2_4"
LABEL_MODE="span"
RUN_ID="v2"
N_NEUTRAL_DEV=1500
N_NEUTRAL_TEST=1500
K_PER_SCENARIO=15
MAX_CONCURRENT=30
DETECT_LAYERS="27 28 29 30"
SKIP_NEUTRAL_SCORE=false
SKIP_COLLECT=false
SKIP_ANALYZE=false
SKIP_GENERATE=false
SKIP_TRAIN=false
NEUTRAL_FPS=""
LAYER=30

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-subdir) RESULTS_SUBDIR="$2"; shift 2 ;;
        --indicator-set) INDICATOR_SET="$2"; shift 2 ;;
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --n-neutral-dev) N_NEUTRAL_DEV="$2"; shift 2 ;;
        --n-neutral-test) N_NEUTRAL_TEST="$2"; shift 2 ;;
        --k-per-scenario) K_PER_SCENARIO="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --skip-neutral-score) SKIP_NEUTRAL_SCORE=true; shift ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --skip-analyze) SKIP_ANALYZE=true; shift ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --neutral-fps) NEUTRAL_FPS="$2"; shift 2 ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

# Existing probe/results dirs (input — never modified)
EXISTING_PROBES_DIR=${BASE_DIR}/probe/probes/${RESULTS_SUBDIR}
ORIGINAL_DATA_DIR=${BASE_DIR}/probe/data/${INDICATOR_SET}

# Feedback intermediate outputs (versioned by RUN_ID)
FEEDBACK_DIR=${BASE_DIR}/probe/data/feedback/${RESULTS_SUBDIR}_${RUN_ID}
FEEDBACK_DATA_DIR=${BASE_DIR}/probe/data/${INDICATOR_SET}_feedback_${RUN_ID}
COLLECTED_ERRORS=${FEEDBACK_DIR}/collected_errors.json
ANALYSIS_SUGGESTIONS=${FEEDBACK_DIR}/analysis_suggestions.json

# New combined outputs (training data, probes, results — versioned by RUN_ID)
COMBINED_DATA_DIR=${BASE_DIR}/probe/data/${INDICATOR_SET}_combined_${RUN_ID}
COMBINED_PROBE_DIR=${BASE_DIR}/probe/probes/${INDICATOR_SET}_combined_${RUN_ID}_${LABEL_MODE}
COMBINED_RESULTS_SUBDIR="${INDICATOR_SET}_combined_${RUN_ID}_${LABEL_MODE}"

# Training params
REG_COEFF=10.0

# Bloom rollout directories (same as run_v2_4_existing_pipeline.sh)
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

# Held-out test set rollout directories
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
# OOD evaluation sets (outside bloom)
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_are_you_sure"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback"
)

# If no neutral FPs path specified, use a default
if [ -z "${NEUTRAL_FPS}" ]; then
    NEUTRAL_FPS=${BASE_DIR}/probe/data/neutral/false_positives_${RESULTS_SUBDIR}_filtered.json
fi

# SLURM settings
PARTITION="general,overflow"
QOS="low"
MEMORY="200G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/feedback_loop_${timestamp}"
mkdir -p "${WORK_DIR}" "${FEEDBACK_DIR}"

PREV_JOB=""

# ======================================================================
# Job 1: Collect errors — OOD + neutral (1 GPU for neutral scoring)
# ======================================================================
if [ "${SKIP_COLLECT}" = false ]; then
    COLLECT_SCRIPT="${WORK_DIR}/collect.qsh"

    # Build neutral scoring flags and determine GPU need
    COLLECT_GPU=""
    if [ "${SKIP_NEUTRAL_SCORE}" = false ]; then
        # Check if pre-scored file already exists — use it to avoid GPU
        if [ -n "${NEUTRAL_FPS}" ] && [ -f "${NEUTRAL_FPS}" ]; then
            NEUTRAL_FLAGS="--neutral-fps ${NEUTRAL_FPS} --n-neutral ${N_NEUTRAL_DEV}"
        else
            NEUTRAL_FLAGS="--score-neutral --n-neutral ${N_NEUTRAL_DEV}"
            COLLECT_GPU="#SBATCH --gres=gpu:1"
        fi
    elif [ -n "${NEUTRAL_FPS}" ] && [ -f "${NEUTRAL_FPS}" ]; then
        NEUTRAL_FLAGS="--neutral-fps ${NEUTRAL_FPS} --n-neutral ${N_NEUTRAL_DEV}"
    else
        NEUTRAL_FLAGS=""
    fi

    cat <<SLURM_HEADER > "${COLLECT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb_collect
#SBATCH --output=${WORK_DIR}/collect_%j.out
#SBATCH --error=${WORK_DIR}/collect_%j.err
${COLLECT_GPU}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=64G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Stage 1: Collect errors"
echo "Results subdir: ${RESULTS_SUBDIR}"
echo "Started: \$(date)"
echo "=========================================="
SLURM_HEADER

    cat <<EOF >> "${COLLECT_SCRIPT}"

echo ""
echo "--- Collecting OOD + neutral errors ---"
${PYTHON} -m probe.feedback.collect_errors \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --probes-dir "${EXISTING_PROBES_DIR}" \\
    --indicator-set ${INDICATOR_SET} \\
    --layer ${LAYER} \\
    ${NEUTRAL_FLAGS} \\
    --output "${COLLECTED_ERRORS}"

echo ""
echo "Collection complete at \$(date)"
echo "  Output: ${COLLECTED_ERRORS}"
EOF

    COLLECT_JOB=$(sbatch --parsable "${COLLECT_SCRIPT}")
    PREV_JOB="${COLLECT_JOB}"
    if [ -n "${COLLECT_GPU}" ]; then
        echo "[1/5] Collect job ${COLLECT_JOB} submitted (1 GPU, on-the-fly neutral scoring)"
    else
        echo "[1/5] Collect job ${COLLECT_JOB} submitted (CPU-only, using pre-scored neutral FPs)"
    fi
else
    echo "[1/5] Skipping collection (--skip-collect)"
fi

# ======================================================================
# Job 2: Multi-agent Opus 4.6 analysis (no GPU)
# ======================================================================
if [ "${SKIP_ANALYZE}" = false ]; then
    ANALYZE_SCRIPT="${WORK_DIR}/analyze.qsh"

    cat <<EOF > "${ANALYZE_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb_analyze
#SBATCH --output=${WORK_DIR}/analyze_%j.out
#SBATCH --error=${WORK_DIR}/analyze_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 2: Multi-agent Opus 4.6 analysis"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.analyze_errors \\
    --errors-file "${COLLECTED_ERRORS}" \\
    --output "${ANALYSIS_SUGGESTIONS}" \\
    --max-concurrent ${MAX_CONCURRENT}

echo ""
echo "Analysis complete at \$(date)"
echo "  Output: ${ANALYSIS_SUGGESTIONS}"
EOF

    if [ -n "${PREV_JOB}" ]; then
        ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${ANALYZE_SCRIPT}")
        echo "[2/5] Analyze job ${ANALYZE_JOB} submitted (no GPU, after ${PREV_JOB})"
    else
        ANALYZE_JOB=$(sbatch --parsable "${ANALYZE_SCRIPT}")
        echo "[2/5] Analyze job ${ANALYZE_JOB} submitted (no GPU)"
    fi
    PREV_JOB="${ANALYZE_JOB}"
else
    echo "[2/5] Skipping analysis (--skip-analyze)"
fi

# ======================================================================
# Job 3: Sonnet 4.6 data generation + merge (no GPU)
# ======================================================================
if [ "${SKIP_GENERATE}" = false ]; then
    GENERATE_SCRIPT="${WORK_DIR}/generate.qsh"

    cat <<EOF > "${GENERATE_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb_generate
#SBATCH --output=${WORK_DIR}/generate_%j.out
#SBATCH --error=${WORK_DIR}/generate_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 3: Sonnet 4.6 data generation"
echo "K per scenario: ${K_PER_SCENARIO}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.generate_data \\
    --suggestions-file "${ANALYSIS_SUGGESTIONS}" \\
    --output-dir "${FEEDBACK_DATA_DIR}" \\
    --indicator-set ${INDICATOR_SET} \\
    --k ${K_PER_SCENARIO} \\
    --max-concurrent ${MAX_CONCURRENT}

echo ""
echo "Generation complete at \$(date)"
echo "  Feedback data: ${FEEDBACK_DATA_DIR}"

echo ""
echo "=========================================="
echo "Stage 4: Merge feedback + original data"
echo "=========================================="
echo "  Original data (UNCHANGED): ${ORIGINAL_DATA_DIR}"
echo "  Feedback data:             ${FEEDBACK_DATA_DIR}"
echo "  Combined output (NEW):     ${COMBINED_DATA_DIR}"

${PYTHON} -m probe.feedback.merge_data \\
    --original-dir "${ORIGINAL_DATA_DIR}" \\
    --feedback-dir "${FEEDBACK_DATA_DIR}" \\
    --output-dir "${COMBINED_DATA_DIR}"

echo ""
echo "Merge complete at \$(date)"
EOF

    if [ -n "${PREV_JOB}" ]; then
        GENERATE_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${GENERATE_SCRIPT}")
        echo "[3/5] Generate+Merge job ${GENERATE_JOB} submitted (no GPU, after ${PREV_JOB})"
    else
        GENERATE_JOB=$(sbatch --parsable "${GENERATE_SCRIPT}")
        echo "[3/5] Generate+Merge job ${GENERATE_JOB} submitted (no GPU)"
    fi
    PREV_JOB="${GENERATE_JOB}"
else
    echo "[3/5] Skipping generation+merge (--skip-generate)"
fi

# ======================================================================
# Job 4: Retrain probes on combined data (1 GPU)
# ======================================================================
if [ "${SKIP_TRAIN}" = false ]; then
    TRAIN_SCRIPT="${WORK_DIR}/train.qsh"

    cat <<'SLURM_HEADER' > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb_train
#SBATCH --output=__WORK_DIR__/train_%j.out
#SBATCH --error=__WORK_DIR__/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=__PARTITION__
#SBATCH --qos=__QOS__
#SBATCH --mem=__MEMORY__
#SBATCH --chdir=__BASE_DIR__
set -e

source __BASE_DIR__/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Stage 5: Retrain probes on combined data"
echo "Label mode: __LABEL_MODE__"
echo "Detect layers: __DETECT_LAYERS__"
echo "Data dir: __COMBINED_DATA_DIR__"
echo "Probe dir: __COMBINED_PROBE_DIR__"
echo "Started: $(date)"
echo "=========================================="
SLURM_HEADER

    sed -i "s|__WORK_DIR__|${WORK_DIR}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__PARTITION__|${PARTITION}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__QOS__|${QOS}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__MEMORY__|${MEMORY}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__BASE_DIR__|${BASE_DIR}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__LABEL_MODE__|${LABEL_MODE}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__DETECT_LAYERS__|${DETECT_LAYERS}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__COMBINED_DATA_DIR__|${COMBINED_DATA_DIR}|g" "${TRAIN_SCRIPT}"
    sed -i "s|__COMBINED_PROBE_DIR__|${COMBINED_PROBE_DIR}|g" "${TRAIN_SCRIPT}"

    # Discover which indicators have combined data files and train each
    cat <<EOF >> "${TRAIN_SCRIPT}"

# Train each indicator that has combined data
for DATA_FILE in "${COMBINED_DATA_DIR}"/*.json; do
    [ -f "\${DATA_FILE}" ] || continue
    INDICATOR_SLUG=\$(basename "\${DATA_FILE}" .json)

    # Read indicator name from the JSON
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
echo "  Data:   ${COMBINED_DATA_DIR}"
echo "  Probes: ${COMBINED_PROBE_DIR}"
EOF

    if [ -n "${PREV_JOB}" ]; then
        TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${TRAIN_SCRIPT}")
        echo "[4/5] Train job ${TRAIN_JOB} submitted (1 GPU, after ${PREV_JOB})"
    else
        TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
        echo "[4/5] Train job ${TRAIN_JOB} submitted (1 GPU)"
    fi
    PREV_JOB="${TRAIN_JOB}"
else
    echo "[4/5] Skipping training (--skip-train)"
fi

# ======================================================================
# Job 5: Evaluate retrained probes + neutral FP scoring (2 GPUs)
# ======================================================================
EVAL_SCRIPT="${WORK_DIR}/eval.qsh"

cat <<EOF > "${EVAL_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb_eval
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
echo "Stage 6: Evaluate retrained probes"
echo "Probe dir:    ${COMBINED_PROBE_DIR}"
echo "Results:      probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "Started: \$(date)"
echo "=========================================="

# Collect probe folders from the retrained probes dir
PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
NUM_PROBES=\$(echo "\${PROBE_FOLDERS}" | wc -w)
echo "Found \${NUM_PROBES} retrained probes"

if [ "\${NUM_PROBES}" -eq 0 ]; then
    echo "ERROR: No probes found in ${COMBINED_PROBE_DIR}"
    exit 1
fi

# ---- Evaluate on positive bloom rollouts ----
echo ""
echo "========== Evaluate on positive data =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

# ---- Evaluate on benign bloom rollouts ----
echo ""
echo "========== Evaluate on benign data (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# Behavior filter patterns for dev/test split
TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"

# ---- Indicator ground-truth metrics (dev) ----
echo ""
echo "========== Indicator ground truth (dev) =========="
${PYTHON} -m probe_eval.indicator_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --indicator-gt v2.3 \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

# ---- Misalignment ground-truth metrics (dev) ----
echo ""
echo "========== Misalignment ground truth (dev) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

# ---- Visualization dashboard ----
echo ""
echo "========== Visualization =========="
${PYTHON} probe_eval/visualize_misalignment.py \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR}

# ---- Neutral FP scoring on retrained probes (dev set: 0..${N_NEUTRAL_DEV}) ----
echo ""
echo "========== Neutral FP scoring — dev set (${N_NEUTRAL_DEV} dialogues, offset=0) =========="
${PYTHON} -u -m probe.neutral.score_neutral \\
    --probes-dir "${COMBINED_PROBE_DIR}" \\
    --layer ${LAYER} \\
    --offset 0 \\
    --max-dialogues ${N_NEUTRAL_DEV} \\
    --output "${BASE_DIR}/probe/data/neutral/false_positives_${COMBINED_RESULTS_SUBDIR}_dev.json"

# ---- Neutral FP scoring on retrained probes (test set: ${N_NEUTRAL_DEV}..${N_NEUTRAL_DEV}+${N_NEUTRAL_TEST}) ----
echo ""
echo "========== Neutral FP scoring — test set (${N_NEUTRAL_TEST} dialogues, offset=${N_NEUTRAL_DEV}) =========="
${PYTHON} -u -m probe.neutral.score_neutral \\
    --probes-dir "${COMBINED_PROBE_DIR}" \\
    --layer ${LAYER} \\
    --offset ${N_NEUTRAL_DEV} \\
    --max-dialogues ${N_NEUTRAL_TEST} \\
    --output "${BASE_DIR}/probe/data/neutral/false_positives_${COMBINED_RESULTS_SUBDIR}_test.json"

# ======================================================================
# Held-out test set evaluation
# ======================================================================

echo ""
echo "=========================================="
echo "Held-out Test Set Evaluation"
echo "=========================================="

# ---- Bloom test set — positive ----
echo ""
echo "========== Test set: bloom positive =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

# ---- Bloom test set — benign ----
echo ""
echo "========== Test set: bloom benign =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# ---- OOD evaluation sets ----
echo ""
echo "========== Test set: OOD evals =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EVAL_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

# ---- Test set ground-truth metrics ----
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
echo ""
echo "Results (retrained):"
echo "  Probe eval:       probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "  Indicator GT (dev):  probe_eval/results/${COMBINED_RESULTS_SUBDIR}/indicator_gt_summary_dev.json"
echo "  Misalignment GT (dev):  probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_dev.json"
echo "  Misalignment GT (test): probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_test.json"
echo "  Dashboard:        probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_dashboard.html"
echo "  Neutral FPs (dev):  probe/data/neutral/false_positives_${COMBINED_RESULTS_SUBDIR}_dev.json"
echo "  Neutral FPs (test): probe/data/neutral/false_positives_${COMBINED_RESULTS_SUBDIR}_test.json"
echo ""
echo "Compare with original:"
echo "  Original results: probe_eval/results/${RESULTS_SUBDIR}/"
EOF

if [ -n "${PREV_JOB}" ]; then
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${EVAL_SCRIPT}")
    echo "[5/5] Eval job ${EVAL_JOB} submitted (2 GPUs, after ${PREV_JOB})"
else
    EVAL_JOB=$(sbatch --parsable "${EVAL_SCRIPT}")
    echo "[5/5] Eval job ${EVAL_JOB} submitted (2 GPUs)"
fi

# ---- Summary ----
echo ""
echo "========================================"
echo "Feedback Loop Jobs Submitted"
echo "========================================"
echo "  Results subdir (input):  ${RESULTS_SUBDIR}"
echo "  Indicator set:           ${INDICATOR_SET}"
echo "  Run ID:                  ${RUN_ID}"
echo "  Neutral dev size:        ${N_NEUTRAL_DEV}"
echo "  Neutral test size:       ${N_NEUTRAL_TEST}"
echo "  K per scenario:          ${K_PER_SCENARIO}"
echo ""
echo "  Original data (UNCHANGED):  ${ORIGINAL_DATA_DIR}"
echo "  Original probes (UNCHANGED): ${EXISTING_PROBES_DIR}"
echo ""
echo "  Feedback data (NEW):     ${FEEDBACK_DATA_DIR}"
echo "  Combined data (NEW):     ${COMBINED_DATA_DIR}"
echo "  Retrained probes (NEW):  ${COMBINED_PROBE_DIR}"
echo "  Eval results (NEW):      probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
