#!/bin/bash
# =============================================================================
# Strict Feedback Loop for v3_v2_5 probes
# =============================================================================
#
# Same as run_feedback_loop_v3_v2_5.sh but uses strict indicator assignment
# (max 1-2 indicators per evidence span) to produce sharper training data.
#
#   Job 0 (no GPU): Strict indicator assignment (writes applicable_indicators_strict)
#   Job 1 (no GPU): Collect OOD errors using strict indicators
#   Job 2 (no GPU): Batched Opus 4.6 analysis of error patterns
#   Job 3 (no GPU): Sonnet 4.6 data generation + merge with original data
#   Job 4a/4b (2 GPUs): Retrain probes on combined data (parallel splits)
#   Job 5a (1 GPU): Evaluate on dev + OOD rollouts (parallel with 5b)
#   Job 5b (1 GPU): Evaluate on test rollouts (parallel with 5a)
#   Job 5c (no GPU): Misalignment GT + visualization (after 5a+5b)
#
# Original training data is NEVER modified — all output goes to new directories.
#
# Usage:
#   bash scripts/probe_train_eval/run_feedback_loop_v3_v2_5_strict.sh [options]
#
# Options:
#   --run-id ID              Version tag for output paths (default: v1_strict)
#   --k-per-scenario N       Transcripts per (behavior,scenario) pair (default: 3)
#   --max-concurrent N       Max concurrent API calls (default: 30)
#   --detect-layers L...     Layers to probe (default: "27 28 29 30")
#   --skip-assign            Skip strict indicator assignment
#   --skip-collect           Skip error collection
#   --skip-analyze           Skip error analysis
#   --skip-generate          Skip data generation + merge
#   --skip-train             Skip retraining
#   --layer N                Primary probe layer for thresholds (default: 30)
# =============================================================================

set -e

# ---- Defaults ----
RESULTS_SUBDIR="v3_v2_5_span"
INDICATOR_SET="v2_5"
LABEL_MODE="span"
RUN_ID="v1_strict"
INDICATOR_KEY="applicable_indicators_strict"
N_NEUTRAL_DEV=1500
N_NEUTRAL_TEST=1500
K_PER_SCENARIO=3
MAX_CONCURRENT_ANALYZE=4
MAX_CONCURRENT_GENERATE=25
SKIP_ASSIGN=false
DETECT_LAYERS="27 28 29 30"
SKIP_COLLECT=false
SKIP_ANALYZE=false
SKIP_GENERATE=false
SKIP_TRAIN=false
LAYER=30

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --k-per-scenario) K_PER_SCENARIO="$2"; shift 2 ;;
        --max-concurrent-analyze) MAX_CONCURRENT_ANALYZE="$2"; shift 2 ;;
        --max-concurrent-generate) MAX_CONCURRENT_GENERATE="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --skip-assign) SKIP_ASSIGN=true; shift ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --skip-analyze) SKIP_ANALYZE=true; shift ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

# Existing probe/results dirs (input — never modified)
EXISTING_PROBES_DIR=${BASE_DIR}/probe/probes/${RESULTS_SUBDIR}
ORIGINAL_DATA_DIR=${BASE_DIR}/probe/data/v3_v2_5

# Pre-scored neutral FPs
NEUTRAL_FPS_DEV=${BASE_DIR}/probe/data/neutral/false_positives_v3_v2_5_tuned_dev.json
NEUTRAL_FPS_TEST=${BASE_DIR}/probe/data/neutral/false_positives_v3_v2_5_tuned_test.json

# Feedback intermediate outputs (versioned by RUN_ID)
FEEDBACK_DIR=${BASE_DIR}/probe/data/feedback/${RESULTS_SUBDIR}_${RUN_ID}
FEEDBACK_DATA_DIR=${BASE_DIR}/probe/data/v3_v2_5_feedback_${RUN_ID}
COLLECTED_ERRORS=${FEEDBACK_DIR}/collected_errors.json
ANALYSIS_SUGGESTIONS=${FEEDBACK_DIR}/analysis_suggestions.json

# New combined outputs
COMBINED_DATA_DIR=${BASE_DIR}/probe/data/v3_v2_5_combined_${RUN_ID}
COMBINED_PROBE_DIR=${BASE_DIR}/probe/probes/v3_v2_5_combined_${RUN_ID}_${LABEL_MODE}
COMBINED_RESULTS_SUBDIR="v3_v2_5_combined_${RUN_ID}_${LABEL_MODE}"

# Training params
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
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_benign_glm_4_7_flash"
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

# OOD evaluation sets
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
CACHE_MEM="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/feedback_loop_v3_v2_5_strict_${timestamp}"
mkdir -p "${WORK_DIR}" "${FEEDBACK_DIR}"

PREV_JOB=""

# ======================================================================
# Job 0: Strict indicator assignment (no GPU)
# ======================================================================
if [ "${SKIP_ASSIGN}" = false ]; then
    ASSIGN_SCRIPT="${WORK_DIR}/assign.qsh"

    cat <<EOF > "${ASSIGN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fbs_assign
#SBATCH --output=${WORK_DIR}/assign_%j.out
#SBATCH --error=${WORK_DIR}/assign_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 0: Strict indicator assignment"
echo "Output key: ${INDICATOR_KEY}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.assign_indicators \\
    --strict \\
    --output-key ${INDICATOR_KEY} \\
    --max-concurrent 20

echo ""
echo "Strict assignment complete at \$(date)"
EOF

    ASSIGN_JOB=$(sbatch --parsable "${ASSIGN_SCRIPT}")
    PREV_JOB="${ASSIGN_JOB}"
    echo "[0/5] Strict assign job ${ASSIGN_JOB} submitted (no GPU)"
else
    echo "[0/5] Skipping strict assignment (--skip-assign)"
fi

# ======================================================================
# Job 1: Collect errors — OOD + pre-scored neutral FPs (no GPU)
# ======================================================================
if [ "${SKIP_COLLECT}" = false ]; then
    COLLECT_SCRIPT="${WORK_DIR}/collect.qsh"

    cat <<EOF > "${COLLECT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb3_collect
#SBATCH --output=${WORK_DIR}/collect_%j.out
#SBATCH --error=${WORK_DIR}/collect_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 1: Collect errors"
echo "Results subdir: ${RESULTS_SUBDIR}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.collect_errors \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --probes-dir "${EXISTING_PROBES_DIR}" \\
    --indicator-set ${INDICATOR_SET} \\
    --layer ${LAYER} \\
    --neutral-fps "${NEUTRAL_FPS_DEV}" \\
    --n-neutral ${N_NEUTRAL_DEV} \\
    --indicator-key ${INDICATOR_KEY} \\
    --output "${COLLECTED_ERRORS}"

echo ""
echo "Collection complete at \$(date)"
echo "  Output: ${COLLECTED_ERRORS}"
EOF

    COLLECT_JOB=$(sbatch --parsable "${COLLECT_SCRIPT}")
    PREV_JOB="${COLLECT_JOB}"
    echo "[1/5] Collect job ${COLLECT_JOB} submitted (no GPU, pre-scored neutral FPs)"
else
    echo "[1/5] Skipping collection (--skip-collect)"
fi

# ======================================================================
# Job 2: Batched Opus 4.6 analysis (no GPU)
# ======================================================================
if [ "${SKIP_ANALYZE}" = false ]; then
    ANALYZE_SCRIPT="${WORK_DIR}/analyze.qsh"

    cat <<EOF > "${ANALYZE_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb3_analyze
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
echo "Stage 2: Batched Opus 4.6 analysis"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.analyze_errors \\
    --errors-file "${COLLECTED_ERRORS}" \\
    --output "${ANALYSIS_SUGGESTIONS}" \\
    --max-concurrent ${MAX_CONCURRENT_ANALYZE}

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
#SBATCH --job-name=fb3_generate
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
echo "K per (behavior,scenario): ${K_PER_SCENARIO}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.generate_data \\
    --suggestions-file "${ANALYSIS_SUGGESTIONS}" \\
    --output-dir "${FEEDBACK_DATA_DIR}" \\
    --indicator-set ${INDICATOR_SET} \\
    --k ${K_PER_SCENARIO} \\
    --max-concurrent ${MAX_CONCURRENT_GENERATE}

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
# Job 4a/4b: Retrain probes on combined data (2 parallel GPU jobs)
# ======================================================================
TRAIN_JOBS=()

if [ "${SKIP_TRAIN}" = false ]; then
    for SPLIT_IDX in 1 2; do
        TRAIN_SCRIPT="${WORK_DIR}/train_s${SPLIT_IDX}.qsh"

        cat <<'SLURM_HEADER' > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fbs_train_s__SPLIT_IDX__
#SBATCH --output=__WORK_DIR__/train_s__SPLIT_IDX___%j.out
#SBATCH --error=__WORK_DIR__/train_s__SPLIT_IDX___%j.err
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
echo "Retrain probes (strict) — split __SPLIT_IDX__/2"
echo "Data dir: __COMBINED_DATA_DIR__"
echo "Probe dir: __COMBINED_PROBE_DIR__"
echo "Started: $(date)"
echo "=========================================="

# Discover indicators at runtime and split into 2
ALL_FILES=($(ls __COMBINED_DATA_DIR__/*.json 2>/dev/null))
NUM_FILES=${#ALL_FILES[@]}
SPLIT_SIZE=$(( (NUM_FILES + 1) / 2 ))

if [ __SPLIT_IDX__ -eq 1 ]; then
    SPLIT_FILES=("${ALL_FILES[@]:0:${SPLIT_SIZE}}")
else
    SPLIT_FILES=("${ALL_FILES[@]:${SPLIT_SIZE}}")
fi

echo "Total indicators: ${NUM_FILES}, this split: ${#SPLIT_FILES[@]}"

for DATA_FILE in "${SPLIT_FILES[@]}"; do
    [ -f "${DATA_FILE}" ] || continue
    INDICATOR_SLUG=$(basename "${DATA_FILE}" .json)

    INDICATOR_NAME=$(__PYTHON__ -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('indicator_name',''))" "${DATA_FILE}" 2>/dev/null)
    if [ -z "${INDICATOR_NAME}" ]; then
        echo "  Skipping ${INDICATOR_SLUG}: could not read indicator_name"
        continue
    fi

    echo ""
    echo "--- Training: ${INDICATOR_NAME} ---"
    __PYTHON__ -m probe.train \
        --indicator-set __INDICATOR_SET__ \
        --indicator "${INDICATOR_NAME}" \
        --label-mode "__LABEL_MODE__" \
        --reg-coeff __REG_COEFF__ \
        --detect-layers __DETECT_LAYERS__ \
        --reasoning-only \
        --threshold-mode sentence \
        --val-fraction 0.2 \
        --data-dir "__COMBINED_DATA_DIR__" \
        --output-dir "__COMBINED_PROBE_DIR__"
done

echo ""
echo "Training split __SPLIT_IDX__ complete at $(date)"
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
        sed -i "s|__SPLIT_IDX__|${SPLIT_IDX}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__PYTHON__|${PYTHON}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__INDICATOR_SET__|${INDICATOR_SET}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__REG_COEFF__|${REG_COEFF}|g" "${TRAIN_SCRIPT}"

        if [ -n "${PREV_JOB}" ]; then
            TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${TRAIN_SCRIPT}")
            echo "[4${SPLIT_IDX}/5] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU, after ${PREV_JOB})"
        else
            TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
            echo "[4${SPLIT_IDX}/5] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU)"
        fi
        TRAIN_JOBS+=("${TRAIN_JOB}")
    done
else
    echo "[4/5] Skipping training (--skip-train)"
fi

# Build dependency string for eval jobs (wait for ALL train jobs)
TRAIN_DEP=""
if [ ${#TRAIN_JOBS[@]} -gt 0 ]; then
    TRAIN_DEP=$(IFS=:; echo "${TRAIN_JOBS[*]}")
fi

# ======================================================================
# Job 5a: Evaluate on dev + OOD rollouts (1 GPU)
# Job 5b: Evaluate on test rollouts (1 GPU) — parallel with 5a
# Job 5c: Misalignment GT + visualization + neutral FP scoring (no GPU)
# ======================================================================
EVAL_JOBS=()

# ---- Job 5a: Dev + OOD rollouts ----
EVAL_DEV_SCRIPT="${WORK_DIR}/eval_dev.qsh"
cat <<EOF > "${EVAL_DEV_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb3_eval_dev
#SBATCH --output=${WORK_DIR}/eval_dev_%j.out
#SBATCH --error=${WORK_DIR}/eval_dev_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Evaluate retrained probes — Dev + OOD"
echo "Probe dir:    ${COMBINED_PROBE_DIR}"
echo "Results:      probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) retrained probes"

# ---- Dev positive (all_negative to bypass judgment issues) ----
echo ""
echo "========== Dev: positive data =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# ---- Dev benign ----
echo ""
echo "========== Dev: benign data (all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# ---- OOD evals ----
echo ""
echo "========== OOD evals =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EVAL_DIRS[@]} \\
    --behavior_threshold 5 \\
    --skip_existing

echo ""
echo "Dev + OOD evaluation complete at \$(date)"
EOF

if [ -n "${TRAIN_DEP}" ]; then
    EVAL_DEV_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_DEP} "${EVAL_DEV_SCRIPT}")
else
    EVAL_DEV_JOB=$(sbatch --parsable "${EVAL_DEV_SCRIPT}")
fi
EVAL_JOBS+=("${EVAL_DEV_JOB}")
echo "[5a/5] Eval dev+OOD job ${EVAL_DEV_JOB} submitted (1 GPU)"

# ---- Job 5b: Test rollouts ----
EVAL_TEST_SCRIPT="${WORK_DIR}/eval_test.qsh"
cat <<EOF > "${EVAL_TEST_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb3_eval_test
#SBATCH --output=${WORK_DIR}/eval_test_%j.out
#SBATCH --error=${WORK_DIR}/eval_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Evaluate retrained probes — Test Rollouts"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) retrained probes"

# ---- Test positive (all_negative to bypass judgment issues) ----
echo ""
echo "========== Test: bloom positive =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

# ---- Test benign ----
echo ""
echo "========== Test: bloom benign =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

echo ""
echo "Test evaluation complete at \$(date)"
EOF

if [ -n "${TRAIN_DEP}" ]; then
    EVAL_TEST_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_DEP} "${EVAL_TEST_SCRIPT}")
else
    EVAL_TEST_JOB=$(sbatch --parsable "${EVAL_TEST_SCRIPT}")
fi
EVAL_JOBS+=("${EVAL_TEST_JOB}")
echo "[5b/5] Eval test job ${EVAL_TEST_JOB} submitted (1 GPU)"

# ---- Job 5c: Misalignment GT + visualization + neutral FP scoring ----
EVAL_GT_SCRIPT="${WORK_DIR}/eval_gt.qsh"
EVAL_JOBS_DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")

cat <<EOF > "${EVAL_GT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=fb3_gt_viz
#SBATCH --output=${WORK_DIR}/eval_gt_%j.out
#SBATCH --error=${WORK_DIR}/eval_gt_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Misalignment Ground Truth + Visualization"
echo "Started: \$(date)"
echo "=========================================="

TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"

TUNED_PATH="probe_eval/results/${COMBINED_RESULTS_SUBDIR}/tuned_thresholds.json"

# ---- Misalignment ground-truth (dev) — tune thresholds ----
echo ""
echo "========== Misalignment ground truth (dev) + tune thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev \\
    --tune-thresholds \\
    --tuned-thresholds-save "\${TUNED_PATH}"

# ---- Misalignment ground-truth (test) — load tuned thresholds ----
echo ""
echo "========== Misalignment ground truth (test) + tuned thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test \\
    --load-tuned-thresholds "\${TUNED_PATH}"

# ---- Sentence visualization dashboard ----
echo ""
echo "========== Sentence Visualization =========="
${PYTHON} -m probe_eval.visualize_misalignment_sentence \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --layer ${LAYER}

echo ""
echo "=========================================="
echo "All evaluation complete at \$(date)"
echo "=========================================="
echo ""
echo "Results (retrained):"
echo "  Probe eval:             probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo "  Misalignment GT (dev):  probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_dev.json"
echo "  Misalignment GT (test): probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_gt_summary_test.json"
echo "  Dashboard:              probe_eval/results/${COMBINED_RESULTS_SUBDIR}/misalignment_sentence_dashboard.html"
echo ""
echo "Compare with original:"
echo "  Original results: probe_eval/results/${RESULTS_SUBDIR}/"
EOF

EVAL_GT_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOBS_DEP} "${EVAL_GT_SCRIPT}")
echo "[5c/5] GT + viz job ${EVAL_GT_JOB} submitted (no GPU, after eval jobs: ${EVAL_JOBS_DEP})"

# ---- Summary ----
echo ""
echo "========================================"
echo "Strict Feedback Loop Jobs Submitted"
echo "========================================"
echo "  Results subdir (input):  ${RESULTS_SUBDIR}"
echo "  Indicator set:           ${INDICATOR_SET}"
echo "  Indicator key:           ${INDICATOR_KEY}"
echo "  Run ID:                  ${RUN_ID}"
echo "  K per (behavior,scenario): ${K_PER_SCENARIO}"
echo "  Neutral dev FPs:         ${NEUTRAL_FPS_DEV}"
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
