#!/bin/bash
# =============================================================================
# Train and evaluate probes on behavior-level and mechanism-level combined datasets.
# =============================================================================
#
# For each dataset version (behavior, mechanism):
#   Job 1a/1b: Train probes (2 parallel GPU jobs, ~5 indicators each)
#   Job 2a:    Evaluate on dev + OOD rollouts (1 GPU, after train)
#   Job 2b:    Evaluate on test rollouts (1 GPU, after train)
#   Job 2c:    Misalignment GT + visualization (no GPU, after 2a+2b)
#
# Usage:
#   bash scripts/probe_train_eval/run_combined_probe_pipeline.sh
#   bash scripts/probe_train_eval/run_combined_probe_pipeline.sh --skip-train
#   bash scripts/probe_train_eval/run_combined_probe_pipeline.sh --version behavior
#   bash scripts/probe_train_eval/run_combined_probe_pipeline.sh --version mechanism
#
# Options:
#   --version VERSION      behavior|mechanism|both (default: both)
#   --label-mode MODE      turn or span (default: span)
#   --detect-layers L...   Layers to probe (default: "27 28 29 30")
#   --skip-train           Skip training (probes already trained)
#   --layer N              Primary probe layer for viz (default: 27)
# =============================================================================

set -e

# ---- Parse arguments ----
VERSION="both"
LABEL_MODE="span"
DETECT_LAYERS="27 28 29 30"
SKIP_TRAIN=false
LAYER=27

while [[ $# -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift 2 ;;
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

REG_COEFF=10.0

# Dataset versions to process
if [ "${VERSION}" = "both" ]; then
    VERSIONS=("behavior" "mechanism")
elif [ "${VERSION}" = "behavior" ] || [ "${VERSION}" = "mechanism" ]; then
    VERSIONS=("${VERSION}")
else
    echo "Invalid --version: ${VERSION}. Use behavior, mechanism, or both."
    exit 1
fi

# Data directories
declare -A DATA_DIRS
DATA_DIRS[behavior]="${BASE_DIR}/probe/data/v3_v2_5_combined_v1_behavior"
DATA_DIRS[mechanism]="${BASE_DIR}/probe/data/v3_v2_5_combined_v1_mechanism"

# Probe output directories (suffixed with label_mode)
declare -A PROBE_DIRS
PROBE_DIRS[behavior]="${BASE_DIR}/probe/probes/v3_v2_5_combined_v1_behavior_${LABEL_MODE}"
PROBE_DIRS[mechanism]="${BASE_DIR}/probe/probes/v3_v2_5_combined_v1_mechanism_${LABEL_MODE}"

# Dev positive rollout directories (8)
POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glmflash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_glm_4_7_flash"
)

# Dev benign rollout directories (8)
BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_benign_glm_4_7_flash"
)

# Test positive rollout directories (8)
TEST_POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-backdoor__glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-bug_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-promotion_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_glm_4_7_flash"
)

# Test benign rollout directories (8)
TEST_BENIGN_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preferential-bias_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-promotion_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_benign_glm_4_7_flash"
)

# OOD evaluation directories (6)
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_are_you_sure"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback"
    "${BASE_DIR}/ood_misalignment_eval/mask/rollouts/glm-4.7-flash/mask"
)

# SLURM settings
PARTITION="general,overflow"
QOS="high"
MEMORY="200G"
CACHE_MEM="128G"

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/combined_probe_pipeline_${timestamp}"
mkdir -p "${WORK_DIR}"

echo "========================================"
echo "Combined Probe Pipeline"
echo "========================================"
echo "  Versions:      ${VERSIONS[*]}"
echo "  Label mode:    ${LABEL_MODE}"
echo "  Detect layers: ${DETECT_LAYERS}"
echo "  Skip train:    ${SKIP_TRAIN}"
echo "  Work dir:      ${WORK_DIR}"
echo "========================================"

# ======================================================================
# Process each dataset version
# ======================================================================
for VER in "${VERSIONS[@]}"; do
    DATA_DIR="${DATA_DIRS[${VER}]}"
    PROBE_DIR="${PROBE_DIRS[${VER}]}"
    RESULTS_SUBDIR=$(basename "${PROBE_DIR}")

    echo ""
    echo "========================================"
    echo "Version: ${VER}"
    echo "  Data dir:    ${DATA_DIR}"
    echo "  Probe dir:   ${PROBE_DIR}"
    echo "  Results sub: ${RESULTS_SUBDIR}"
    echo "========================================"

    # Verify data directory exists
    if [ ! -d "${DATA_DIR}" ]; then
        echo "ERROR: Data directory not found: ${DATA_DIR}"
        exit 1
    fi

    # Collect indicator JSON files
    ALL_DATA_FILES=(${DATA_DIR}/*.json)
    NUM_INDICATORS=${#ALL_DATA_FILES[@]}
    echo "  Indicators:  ${NUM_INDICATORS}"

    # ==================================================================
    # Job 1a/1b: Train probes (2 parallel GPU jobs)
    # ==================================================================
    TRAIN_JOBS=()

    if [ "${SKIP_TRAIN}" = false ]; then
        SPLIT_SIZE=$(( (NUM_INDICATORS + 1) / 2 ))
        SPLIT_1=("${ALL_DATA_FILES[@]:0:${SPLIT_SIZE}}")
        SPLIT_2=("${ALL_DATA_FILES[@]:${SPLIT_SIZE}}")

        for SPLIT_IDX in 1 2; do
            eval "SPLIT_FILES=(\"\${SPLIT_${SPLIT_IDX}[@]}\")"
            TRAIN_SCRIPT="${WORK_DIR}/train_${VER}_s${SPLIT_IDX}.qsh"

            cat <<'SLURM_HEADER' > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=comb_train___VER___s__SPLIT_IDX__
#SBATCH --output=__WORK_DIR__/train___VER___s__SPLIT_IDX___%j.out
#SBATCH --error=__WORK_DIR__/train___VER___s__SPLIT_IDX___%j.err
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
echo "Training __VER__ probes — split __SPLIT_IDX__/2"
echo "Label mode: __LABEL_MODE__"
echo "Detect layers: __DETECT_LAYERS__"
echo "Data dir: __DATA_DIR__"
echo "Probe dir: __PROBE_DIR__"
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
            sed -i "s|__DATA_DIR__|${DATA_DIR}|g" "${TRAIN_SCRIPT}"
            sed -i "s|__PROBE_DIR__|${PROBE_DIR}|g" "${TRAIN_SCRIPT}"
            sed -i "s|__VER__|${VER}|g" "${TRAIN_SCRIPT}"
            sed -i "s|__SPLIT_IDX__|${SPLIT_IDX}|g" "${TRAIN_SCRIPT}"

            # Extract indicator names from this split's data files
            SPLIT_INDICATOR_ARGS=""
            for DATA_FILE in "${SPLIT_FILES[@]}"; do
                INDICATOR_NAME=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('indicator_name',''))" "${DATA_FILE}" 2>/dev/null)
                if [ -n "${INDICATOR_NAME}" ]; then
                    SPLIT_INDICATOR_ARGS="${SPLIT_INDICATOR_ARGS} \"${INDICATOR_NAME}\""
                fi
            done

            cat <<EOF >> "${TRAIN_SCRIPT}"

echo ""
echo "--- Training ${#SPLIT_FILES[@]} indicators ---"
${PYTHON} -m probe.train \\
    --indicator-set v2_5 \\
    --indicator ${SPLIT_INDICATOR_ARGS} \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --reasoning-only \\
    --threshold-mode sentence \\
    --val-fraction 0.2 \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${PROBE_DIR}"
EOF

            cat <<EOF >> "${TRAIN_SCRIPT}"

echo ""
echo "Training ${VER} split ${SPLIT_IDX} complete at \$(date)"
EOF

            TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
            TRAIN_JOBS+=("${TRAIN_JOB}")
            echo "  [train ${VER} s${SPLIT_IDX}] Job ${TRAIN_JOB} submitted (${#SPLIT_FILES[@]} indicators)"
        done
    else
        echo "  [train] Skipping (--skip-train)"
    fi

    # Build dependency string for eval jobs
    TRAIN_DEP=""
    if [ ${#TRAIN_JOBS[@]} -gt 0 ]; then
        TRAIN_DEP=$(IFS=:; echo "${TRAIN_JOBS[*]}")
    fi

    # ==================================================================
    # Job 2a: Evaluate on dev + OOD rollouts (1 GPU)
    # ==================================================================
    EVAL_JOBS=()

    EVAL_DEV_SCRIPT="${WORK_DIR}/eval_${VER}_dev.qsh"
    cat <<EOF > "${EVAL_DEV_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=comb_eval_${VER}_dev
#SBATCH --output=${WORK_DIR}/eval_${VER}_dev_%j.out
#SBATCH --error=${WORK_DIR}/eval_${VER}_dev_%j.err
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
echo "${VER} Probe Evaluation — Dev + OOD"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

# ---- Dev positive (all_negative to use rollout_misalignment_turns.json GT) ----
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
    --all_negative \\
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
    echo "  [eval ${VER} dev+ood] Job ${EVAL_DEV_JOB}"

    # ==================================================================
    # Job 2b: Evaluate on test rollouts (1 GPU)
    # ==================================================================
    EVAL_TEST_SCRIPT="${WORK_DIR}/eval_${VER}_test.qsh"
    cat <<EOF > "${EVAL_TEST_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=comb_eval_${VER}_test
#SBATCH --output=${WORK_DIR}/eval_${VER}_test_%j.out
#SBATCH --error=${WORK_DIR}/eval_${VER}_test_%j.err
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
echo "${VER} Probe Evaluation — Test"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

# ---- Test positive ----
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
    echo "  [eval ${VER} test] Job ${EVAL_TEST_JOB}"

    # ==================================================================
    # Job 2c: Misalignment GT + visualization (no GPU, after 2a+2b)
    # ==================================================================
    EVAL_GT_SCRIPT="${WORK_DIR}/eval_${VER}_gt.qsh"
    EVAL_JOBS_DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")

    cat <<EOF > "${EVAL_GT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=comb_gt_${VER}
#SBATCH --output=${WORK_DIR}/eval_${VER}_gt_%j.out
#SBATCH --error=${WORK_DIR}/eval_${VER}_gt_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}
set -e

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "${VER} — Misalignment GT + Visualization"
echo "Started: \$(date)"
echo "=========================================="

TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback,mask"

TUNED_PATH="probe_eval/results/${RESULTS_SUBDIR}/tuned_thresholds.json"

# ---- Misalignment ground-truth (dev) — tune thresholds ----
echo ""
echo "========== Misalignment ground truth (dev) + tune thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev \\
    --tune-thresholds \\
    --tuned-thresholds-save "\${TUNED_PATH}" \\
    --short-sentence-mode discard

# ---- Misalignment ground-truth (test) — load tuned thresholds ----
echo ""
echo "========== Misalignment ground truth (test) + tuned thresholds =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test \\
    --load-tuned-thresholds "\${TUNED_PATH}" \\
    --short-sentence-mode discard

# ---- Sentence visualization dashboard ----
echo ""
echo "========== Sentence Visualization =========="
${PYTHON} -m probe_eval.visualize_misalignment_sentence \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --layer ${LAYER}

echo ""
echo "=========================================="
echo "${VER} evaluation complete at \$(date)"
echo "=========================================="
echo "Results:"
echo "  Probe eval:             probe_eval/results/${RESULTS_SUBDIR}/"
echo "  Misalignment GT (dev):  probe_eval/results/${RESULTS_SUBDIR}/misalignment_gt_summary_dev.json"
echo "  Misalignment GT (test): probe_eval/results/${RESULTS_SUBDIR}/misalignment_gt_summary_test.json"
echo "  Dashboard:              probe_eval/results/${RESULTS_SUBDIR}/misalignment_sentence_dashboard.html"
EOF

    EVAL_GT_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOBS_DEP} "${EVAL_GT_SCRIPT}")
    echo "  [gt ${VER}] Job ${EVAL_GT_JOB} (after ${EVAL_JOBS_DEP})"

done

# ---- Summary ----
echo ""
echo "========================================"
echo "All Jobs Submitted"
echo "========================================"
echo "  Versions:      ${VERSIONS[*]}"
echo "  Label mode:    ${LABEL_MODE}"
echo "  Detect layers: ${DETECT_LAYERS}"
echo ""
for VER in "${VERSIONS[@]}"; do
    echo "  ${VER}:"
    echo "    Data:    ${DATA_DIRS[${VER}]}"
    echo "    Probes:  ${PROBE_DIRS[${VER}]}"
    echo "    Results: probe_eval/results/$(basename ${PROBE_DIRS[${VER}]})/"
done
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
