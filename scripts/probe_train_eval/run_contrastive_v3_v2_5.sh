#!/bin/bash
# =============================================================================
# Contrastive Pair Augmentation for v3_v2_5 probes
# =============================================================================
#
# Generates contrastive training pairs from existing analysis suggestions,
# then generates transcripts, merges with combined data, retrains, and evaluates.
#
#   Job 1 (no GPU): Opus contrastive pair ideation (batched)
#   Job 2 (no GPU): Sonnet transcript generation + merge
#   Job 3a/3b (2 GPUs): Retrain probes on augmented data (parallel splits)
#   Job 4a (1 GPU): Evaluate on dev + OOD rollouts
#   Job 4b (1 GPU): Evaluate on test rollouts
#   Job 4c (no GPU): Misalignment GT + visualization
#
# Usage:
#   bash scripts/probe_train_eval/run_contrastive_v3_v2_5.sh [options]
#
# Options:
#   --run-id ID              Version tag (default: v1)
#   --k-per-scenario N       Transcripts per scenario (default: 3)
#   --max-concurrent N       Max concurrent API calls for generation (default: 25)
#   --detect-layers L...     Layers to probe (default: "27 28 29 30")
#   --skip-contrastive       Skip contrastive pair generation
#   --skip-generate          Skip transcript generation + merge
#   --skip-train             Skip retraining
# =============================================================================

set -e

# ---- Defaults ----
RESULTS_SUBDIR="v3_v2_5_span"
INDICATOR_SET="v2_5"
LABEL_MODE="span"
RUN_ID="v1"
K_PER_SCENARIO=3
MAX_CONCURRENT_CONTRASTIVE=5
MAX_CONCURRENT_GENERATE=25
DETECT_LAYERS="27 28 29 30"
SKIP_CONTRASTIVE=false
SKIP_GENERATE=false
SKIP_TRAIN=false
LAYER=30

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --k-per-scenario) K_PER_SCENARIO="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT_GENERATE="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --skip-contrastive) SKIP_CONTRASTIVE=true; shift ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

# Input: existing analysis suggestions from feedback loop
SUGGESTIONS_FILE=${BASE_DIR}/probe/data/feedback/${RESULTS_SUBDIR}_v1/analysis_suggestions.json

# Input: existing combined data from feedback loop v1
EXISTING_COMBINED_DIR=${BASE_DIR}/probe/data/v3_v2_5_combined_v1

# Contrastive outputs (versioned by RUN_ID)
CONTRASTIVE_DIR=${BASE_DIR}/probe/data/feedback/${RESULTS_SUBDIR}_contrastive_${RUN_ID}
CONTRASTIVE_SUGGESTIONS=${CONTRASTIVE_DIR}/analysis_suggestions_contrastive.json
CONTRASTIVE_FEEDBACK_DIR=${BASE_DIR}/probe/data/v3_v2_5_feedback_contrastive_${RUN_ID}

# Final combined output
COMBINED_DATA_DIR=${BASE_DIR}/probe/data/v3_v2_5_combined_contrastive_${RUN_ID}
COMBINED_PROBE_DIR=${BASE_DIR}/probe/probes/v3_v2_5_combined_contrastive_${RUN_ID}_${LABEL_MODE}
COMBINED_RESULTS_SUBDIR="v3_v2_5_combined_contrastive_${RUN_ID}_${LABEL_MODE}"

# Pre-scored neutral FPs
NEUTRAL_FPS_DEV=${BASE_DIR}/probe/data/neutral/false_positives_v3_v2_5_tuned_dev.json
NEUTRAL_FPS_TEST=${BASE_DIR}/probe/data/neutral/false_positives_v3_v2_5_tuned_test.json

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
CACHE_MEM="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/contrastive_v3_v2_5_${timestamp}"
mkdir -p "${WORK_DIR}" "${CONTRASTIVE_DIR}"

PREV_JOB=""

# ======================================================================
# Job 1: Generate contrastive pair scenarios (no GPU)
# ======================================================================
if [ "${SKIP_CONTRASTIVE}" = false ]; then
    CONTRASTIVE_SCRIPT="${WORK_DIR}/contrastive.qsh"

    cat <<EOF > "${CONTRASTIVE_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_ideation
#SBATCH --output=${WORK_DIR}/contrastive_%j.out
#SBATCH --error=${WORK_DIR}/contrastive_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Stage 1: Contrastive pair ideation"
echo "Input: ${SUGGESTIONS_FILE}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.generate_contrastive \\
    --suggestions-file "${SUGGESTIONS_FILE}" \\
    --output "${CONTRASTIVE_SUGGESTIONS}" \\
    --max-concurrent ${MAX_CONCURRENT_CONTRASTIVE} \\
    --batch-size 15

echo ""
echo "Contrastive ideation complete at \$(date)"
echo "  Output: ${CONTRASTIVE_SUGGESTIONS}"
EOF

    CONTRASTIVE_JOB=$(sbatch --parsable "${CONTRASTIVE_SCRIPT}")
    PREV_JOB="${CONTRASTIVE_JOB}"
    echo "[1/4] Contrastive ideation job ${CONTRASTIVE_JOB} submitted (no GPU)"
else
    echo "[1/4] Skipping contrastive ideation (--skip-contrastive)"
fi

# ======================================================================
# Job 2: Generate transcripts + merge (no GPU)
# ======================================================================
if [ "${SKIP_GENERATE}" = false ]; then
    GENERATE_SCRIPT="${WORK_DIR}/generate.qsh"

    cat <<EOF > "${GENERATE_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_generate
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
echo "Stage 2: Transcript generation from contrastive pairs"
echo "K per scenario: ${K_PER_SCENARIO}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.feedback.generate_data \\
    --suggestions-file "${CONTRASTIVE_SUGGESTIONS}" \\
    --output-dir "${CONTRASTIVE_FEEDBACK_DIR}" \\
    --indicator-set ${INDICATOR_SET} \\
    --k ${K_PER_SCENARIO} \\
    --max-concurrent ${MAX_CONCURRENT_GENERATE}

echo ""
echo "Generation complete at \$(date)"
echo "  Contrastive data: ${CONTRASTIVE_FEEDBACK_DIR}"

echo ""
echo "=========================================="
echo "Stage 3: Merge contrastive + existing combined data"
echo "=========================================="
echo "  Existing combined (UNCHANGED): ${EXISTING_COMBINED_DIR}"
echo "  Contrastive data:              ${CONTRASTIVE_FEEDBACK_DIR}"
echo "  Output (NEW):                  ${COMBINED_DATA_DIR}"

${PYTHON} -m probe.feedback.merge_data \\
    --original-dir "${EXISTING_COMBINED_DIR}" \\
    --feedback-dir "${CONTRASTIVE_FEEDBACK_DIR}" \\
    --output-dir "${COMBINED_DATA_DIR}"

echo ""
echo "Merge complete at \$(date)"
EOF

    if [ -n "${PREV_JOB}" ]; then
        GENERATE_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} "${GENERATE_SCRIPT}")
        echo "[2/4] Generate+Merge job ${GENERATE_JOB} submitted (no GPU, after ${PREV_JOB})"
    else
        GENERATE_JOB=$(sbatch --parsable "${GENERATE_SCRIPT}")
        echo "[2/4] Generate+Merge job ${GENERATE_JOB} submitted (no GPU)"
    fi
    PREV_JOB="${GENERATE_JOB}"
else
    echo "[2/4] Skipping generation+merge (--skip-generate)"
fi

# ======================================================================
# Job 3a/3b: Retrain probes (2 parallel GPU jobs)
# ======================================================================
TRAIN_JOBS=()

if [ "${SKIP_TRAIN}" = false ]; then
    for SPLIT_IDX in 1 2; do
        TRAIN_SCRIPT="${WORK_DIR}/train_s${SPLIT_IDX}.qsh"

        cat <<'SLURM_HEADER' > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_train_s__SPLIT_IDX__
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
echo "Retrain probes — split __SPLIT_IDX__/2"
echo "Label mode: __LABEL_MODE__"
echo "Detect layers: __DETECT_LAYERS__"
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
            echo "[3${SPLIT_IDX}/4] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU, after ${PREV_JOB})"
        else
            TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
            echo "[3${SPLIT_IDX}/4] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU)"
        fi
        TRAIN_JOBS+=("${TRAIN_JOB}")
    done
else
    echo "[3/4] Skipping training (--skip-train)"
fi

# Build dependency string for eval jobs
TRAIN_DEP=""
if [ ${#TRAIN_JOBS[@]} -gt 0 ]; then
    TRAIN_DEP=$(IFS=:; echo "${TRAIN_JOBS[*]}")
fi

# ======================================================================
# Job 4a: Evaluate on dev + OOD rollouts (1 GPU)
# Job 4b: Evaluate on test rollouts (1 GPU) — parallel with 4a
# Job 4c: Misalignment GT + visualization (no GPU)
# ======================================================================
EVAL_JOBS=()

# ---- Job 4a: Dev + OOD rollouts ----
EVAL_DEV_SCRIPT="${WORK_DIR}/eval_dev.qsh"
cat <<EOF > "${EVAL_DEV_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_eval_dev
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
echo "Evaluate — Dev + OOD"
echo "Probe dir: ${COMBINED_PROBE_DIR}"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

echo ""
echo "========== Dev: positive =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

echo ""
echo "========== Dev: benign =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${BENIGN_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

echo ""
echo "========== OOD =========="
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
echo "[4a/4] Eval dev+OOD job ${EVAL_DEV_JOB} submitted (1 GPU)"

# ---- Job 4b: Test rollouts ----
EVAL_TEST_SCRIPT="${WORK_DIR}/eval_test.qsh"
cat <<EOF > "${EVAL_TEST_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_eval_test
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
echo "Evaluate — Test Rollouts"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${COMBINED_PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')

echo ""
echo "========== Test: positive =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POSITIVE_ROLLOUT_DIRS[@]} \\
    --all_negative \\
    --skip_existing

echo ""
echo "========== Test: benign =========="
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
echo "[4b/4] Eval test job ${EVAL_TEST_JOB} submitted (1 GPU)"

# ---- Job 4c: Misalignment GT + visualization ----
EVAL_GT_SCRIPT="${WORK_DIR}/eval_gt.qsh"
EVAL_JOBS_DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")

cat <<EOF > "${EVAL_GT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=ct_gt_viz
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

echo ""
echo "========== GT (dev) + tune thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev \\
    --tune-thresholds \\
    --tuned-thresholds-save "\${TUNED_PATH}"

echo ""
echo "========== GT (test) + tuned thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test \\
    --load-tuned-thresholds "\${TUNED_PATH}"

echo ""
echo "========== Visualization =========="
${PYTHON} -m probe_eval.visualize_misalignment_sentence \\
    --results-subdir ${COMBINED_RESULTS_SUBDIR} \\
    --layer ${LAYER}

echo ""
echo "All evaluation complete at \$(date)"
echo "  Results: probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
EOF

EVAL_GT_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOBS_DEP} "${EVAL_GT_SCRIPT}")
echo "[4c/4] GT + viz job ${EVAL_GT_JOB} submitted (no GPU, after ${EVAL_JOBS_DEP})"

# ---- Summary ----
echo ""
echo "========================================"
echo "Contrastive Augmentation Jobs Submitted"
echo "========================================"
echo "  Input suggestions:     ${SUGGESTIONS_FILE}"
echo "  Existing combined:     ${EXISTING_COMBINED_DIR}"
echo "  Run ID:                ${RUN_ID}"
echo "  K per scenario:        ${K_PER_SCENARIO}"
echo ""
echo "  Contrastive scenarios: ${CONTRASTIVE_SUGGESTIONS}"
echo "  Contrastive feedback:  ${CONTRASTIVE_FEEDBACK_DIR}"
echo "  Combined data (NEW):   ${COMBINED_DATA_DIR}"
echo "  Probes (NEW):          ${COMBINED_PROBE_DIR}"
echo "  Eval results (NEW):    probe_eval/results/${COMBINED_RESULTS_SUBDIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
