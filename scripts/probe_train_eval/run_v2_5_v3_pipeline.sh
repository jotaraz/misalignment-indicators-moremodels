#!/bin/bash
# =============================================================================
# v2.5 Indicator Probes — V3 Pipeline (Ideation → Generate → Train → Eval)
# =============================================================================
#
# End-to-end pipeline using the v3 generation framework:
#   Job 1 (no GPU):  Opus ideation + Sonnet transcript generation via API
#   Job 2 (1 GPU):   Train probes on GLM-4.7 Flash
#   Job 3 (2 GPUs):  Evaluate probes on bloom rollout transcripts
#                     + indicator & misalignment ground-truth metrics
#                     + visualization dashboard
#
# All steps run as chained SLURM jobs (generate → train → eval).
#
# Usage:
#   bash scripts/probe_train_eval/run_v2_5_v3_pipeline.sh [options]
#
# Options:
#   --label-mode MODE      turn or span (default: span)
#   --k NUM                Positive transcripts per indicator (default: 300)
#   --skip-generate        Skip job 1 (data already generated)
#   --skip-train           Skip job 2 (probes already trained)
#   --include-preconditions Include precondition and behavioral concept indicators
#   --hard-negative-ratio R Fraction of hard negatives (default: 0.4)
#   --detect-layers L...   Layers to probe (default: "27 28 29 30")
#   --max-concurrent N     API concurrency (default: 28)
# =============================================================================

set -e

# ---- Parse arguments ----
LABEL_MODE="span"
K=300
SKIP_GENERATE=false
SKIP_TRAIN=false
INCLUDE_PRECONDITIONS=""
HARD_NEG_RATIO=0.4
DETECT_LAYERS="27 28 29 30"
MAX_CONCURRENT=28

while [[ $# -gt 0 ]]; do
    case $1 in
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --include-preconditions) INCLUDE_PRECONDITIONS="--include-preconditions"; shift ;;
        --hard-negative-ratio) HARD_NEG_RATIO="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON_GEN=${BASE_DIR}/automating-probing-and-steering/venv/bin/python   # for generation (anthropic, no torch)
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python  # for train/eval (torch, jaxtyping)

# Data and probe output directories
DATA_DIR=${BASE_DIR}/probe/data/v3_v2_5
PROBE_DIR=${BASE_DIR}/probe/probes/v3_v2_5_${LABEL_MODE}

# Training params
REG_COEFF=10.0

# SLURM settings
PARTITION="general,overflow"
QOS="high"
CACHE_MEM="128G"
TRAIN_MEM="128G"

# v2.5 indicators — 6 remaining not currently in v2.5 (to complete the full 18-probe set)
INDICATORS=(
    "Observer Suspicion Modeling"
    "Oversight as Obstacle"
    "Reward Gaming"
    "Reward Tampering"
    "Strategic Attention Manipulation"
    "Unauthorized Consequential Action Planning"
)

# Positive bloom rollout directories (6)
POSITIVE_ROLLOUT_DIRS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-backdoor"
    "${BASE_DIR}/bloom/bloom-results/instructed-long-horizon-sabotage-bug"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glmflash"
    "${BASE_DIR}/bloom/bloom-results/self-preferential-bias_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-promotion_glm_4_7_flash"
)

# Benign (all-negative) bloom rollout directories (6)
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
OOD_EVAL_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_are_you_sure"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback"
)

# Results subdir (must match probe dir basename for ground-truth scripts)
RESULTS_SUBDIR=$(basename "${PROBE_DIR}")

# ---- Setup ----
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/v2_5_v3_pipeline_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Split indicators into 3 subsets for parallel jobs ----
NUM_INDICATORS=${#INDICATORS[@]}
SPLIT_SIZE=$(( (NUM_INDICATORS + 2) / 3 ))  # ceiling division

SPLIT_1=("${INDICATORS[@]:0:${SPLIT_SIZE}}")
SPLIT_2=("${INDICATORS[@]:${SPLIT_SIZE}:${SPLIT_SIZE}}")
SPLIT_3=("${INDICATORS[@]:$((SPLIT_SIZE * 2))}")

# Build indicator args per split
build_indicator_args() {
    local args=""
    for ind in "$@"; do
        args="${args} \"${ind}\""
    done
    echo "${args}"
}

SPLIT_1_ARGS=$(build_indicator_args "${SPLIT_1[@]}")
SPLIT_2_ARGS=$(build_indicator_args "${SPLIT_2[@]}")
SPLIT_3_ARGS=$(build_indicator_args "${SPLIT_3[@]}")

echo "Indicator splits:"
echo "  Split 1 (${#SPLIT_1[@]}): ${SPLIT_1[0]} ... ${SPLIT_1[-1]}"
echo "  Split 2 (${#SPLIT_2[@]}): ${SPLIT_2[0]} ... ${SPLIT_2[-1]}"
echo "  Split 3 (${#SPLIT_3[@]}): ${SPLIT_3[0]} ... ${SPLIT_3[-1]}"
echo ""

# Track job IDs
GEN_JOBS=()
TRAIN_JOBS=()

# ======================================================================
# Job 1a/1b/1c: Generate synthetic transcripts (3 parallel, no GPU)
#               Uses v3 pipeline: Opus ideation → Sonnet generation
# ======================================================================
if [ "${SKIP_GENERATE}" = false ]; then
    for SPLIT_IDX in 1 2 3; do
        eval "SPLIT_INDS=(\"\${SPLIT_${SPLIT_IDX}[@]}\")"
        eval "SPLIT_ARGS=\${SPLIT_${SPLIT_IDX}_ARGS}"

        GEN_SCRIPT="${WORK_DIR}/generate_split${SPLIT_IDX}.qsh"

        cat <<EOF > "${GEN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_gen_v3_s${SPLIT_IDX}
#SBATCH --output=${WORK_DIR}/generate_split${SPLIT_IDX}_%j.out
#SBATCH --error=${WORK_DIR}/generate_split${SPLIT_IDX}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/automating-probing-and-steering/venv/bin/activate

echo "=========================================="
echo "Generating v2.5 transcripts — split ${SPLIT_IDX}/3"
echo "K: ${K}"
echo "Hard neg ratio: ${HARD_NEG_RATIO}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Indicators: ${#SPLIT_INDS[@]}"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON_GEN} -m probe.generate_v3 \\
    --indicator-set v2_5 \\
    --indicator ${SPLIT_ARGS} \\
    --k ${K} \\
    --max-concurrent ${MAX_CONCURRENT} \\
    --hard-negative-ratio ${HARD_NEG_RATIO} \\
    ${INCLUDE_PRECONDITIONS} \\
    --output-dir "${DATA_DIR}"

echo ""
echo "Generation split ${SPLIT_IDX} complete at \$(date)"
EOF

        GEN_JOB=$(sbatch --parsable "${GEN_SCRIPT}")
        GEN_JOBS+=("${GEN_JOB}")
        echo "[1/3] Generate split ${SPLIT_IDX} job ${GEN_JOB} submitted (${#SPLIT_INDS[@]} indicators)"
    done
else
    echo "[1/3] Skipping generation (--skip-generate)"
fi

# ======================================================================
# Job 2a/2b/2c: Train probes on GLM-4.7 Flash (3 parallel, 1 GPU each)
# ======================================================================
if [ "${SKIP_TRAIN}" = false ]; then
    for SPLIT_IDX in 1 2 3; do
        eval "SPLIT_INDS=(\"\${SPLIT_${SPLIT_IDX}[@]}\")"

        TRAIN_SCRIPT="${WORK_DIR}/train_split${SPLIT_IDX}.qsh"

        cat <<'SLURM_HEADER' > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_v3_s__SPLIT_IDX__
#SBATCH --output=__WORK_DIR__/train_split__SPLIT_IDX___%j.out
#SBATCH --error=__WORK_DIR__/train_split__SPLIT_IDX___%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=__PARTITION__
#SBATCH --qos=__QOS__
#SBATCH --mem=__TRAIN_MEM__
#SBATCH --chdir=__BASE_DIR__

source __BASE_DIR__/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Training v2.5 Probes — split __SPLIT_IDX__/3"
echo "Label mode: __LABEL_MODE__"
echo "Detect layers: __DETECT_LAYERS__"
echo "Started: $(date)"
echo "=========================================="
SLURM_HEADER

        # Replace placeholders
        sed -i "s|__WORK_DIR__|${WORK_DIR}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__PARTITION__|${PARTITION}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__QOS__|${QOS}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__TRAIN_MEM__|${TRAIN_MEM}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__BASE_DIR__|${BASE_DIR}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__LABEL_MODE__|${LABEL_MODE}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__DETECT_LAYERS__|${DETECT_LAYERS}|g" "${TRAIN_SCRIPT}"
        sed -i "s|__SPLIT_IDX__|${SPLIT_IDX}|g" "${TRAIN_SCRIPT}"

        # Append per-indicator training commands for this split
        for ind in "${SPLIT_INDS[@]}"; do
            cat <<EOF >> "${TRAIN_SCRIPT}"

echo ""
echo "--- Training: ${ind} ---"
${PYTHON} -m probe.train \\
    --indicator-set v2_5 \\
    --indicator "${ind}" \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --reasoning-only \\
    --threshold-mode sentence \\
    --val-fraction 0.2 \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${PROBE_DIR}"
EOF
        done

        cat <<EOF >> "${TRAIN_SCRIPT}"

echo ""
echo "Training split ${SPLIT_IDX} complete at \$(date)"
echo "  Data:   ${DATA_DIR}"
echo "  Probes: ${PROBE_DIR}"
EOF

        # Each train job depends on its corresponding generate job
        if [ ${#GEN_JOBS[@]} -gt 0 ]; then
            GEN_DEP=${GEN_JOBS[$((SPLIT_IDX - 1))]}
            TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_DEP} "${TRAIN_SCRIPT}")
            echo "[2/3] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU, after gen ${GEN_DEP})"
        else
            TRAIN_JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
            echo "[2/3] Train split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (1 GPU)"
        fi
        TRAIN_JOBS+=("${TRAIN_JOB}")
    done
else
    echo "[2/3] Skipping training (--skip-train)"
fi

# Build dependency string for eval job (wait for ALL train jobs)
EVAL_DEP=""
if [ ${#TRAIN_JOBS[@]} -gt 0 ]; then
    EVAL_DEP=$(IFS=:; echo "${TRAIN_JOBS[*]}")
fi

# ======================================================================
# Job 3a: Evaluate on dev bloom + OOD rollouts (1 GPU)
# Job 3b: Evaluate on test bloom rollouts (1 GPU) — parallel with 3a
# Job 3c: Misalignment GT + visualization (no GPU, after 3a+3b)
# ======================================================================
EVAL_JOBS=()

# ---- Job 3a: Dev + OOD rollouts ----
EVAL_DEV_SCRIPT="${WORK_DIR}/eval_dev.qsh"
cat <<EOF > "${EVAL_DEV_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_dev
#SBATCH --output=${WORK_DIR}/eval_dev_%j.out
#SBATCH --error=${WORK_DIR}/eval_dev_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=========================================="
echo "v2.5 Probe Evaluation — Dev + OOD Rollouts"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

# ---- Dev positive ----
echo ""
echo "========== Dev: positive data =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${POSITIVE_ROLLOUT_DIRS[@]} \\
    --behavior_threshold 5 \\
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

if [ -n "${EVAL_DEP}" ]; then
    EVAL_DEV_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_DEP} "${EVAL_DEV_SCRIPT}")
else
    EVAL_DEV_JOB=$(sbatch --parsable "${EVAL_DEV_SCRIPT}")
fi
EVAL_JOBS+=("${EVAL_DEV_JOB}")
echo "[3a/3] Eval dev+OOD job ${EVAL_DEV_JOB} submitted (1 GPU)"

# ---- Job 3b: Test rollouts ----
EVAL_TEST_SCRIPT="${WORK_DIR}/eval_test.qsh"
cat <<EOF > "${EVAL_TEST_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_test
#SBATCH --output=${WORK_DIR}/eval_test_%j.out
#SBATCH --error=${WORK_DIR}/eval_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=========================================="
echo "v2.5 Probe Evaluation — Test Rollouts"
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
    --behavior_threshold 5 \\
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

if [ -n "${EVAL_DEP}" ]; then
    EVAL_TEST_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_DEP} "${EVAL_TEST_SCRIPT}")
else
    EVAL_TEST_JOB=$(sbatch --parsable "${EVAL_TEST_SCRIPT}")
fi
EVAL_JOBS+=("${EVAL_TEST_JOB}")
echo "[3b/3] Eval test job ${EVAL_TEST_JOB} submitted (1 GPU)"

# ---- Job 3c: Misalignment GT + visualization (no GPU, after 3a+3b) ----
EVAL_GT_SCRIPT="${WORK_DIR}/eval_gt.qsh"
EVAL_JOBS_DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")

cat <<EOF > "${EVAL_GT_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_gt_viz
#SBATCH --output=${WORK_DIR}/eval_gt_%j.out
#SBATCH --error=${WORK_DIR}/eval_gt_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=========================================="
echo "Misalignment Ground Truth + Visualization"
echo "Started: \$(date)"
echo "=========================================="

TEST_BEHAVIOR_PATTERNS="test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback"

# ---- Misalignment ground-truth (dev) ----
echo ""
echo "========== Misalignment ground truth (dev) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --exclude-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _dev

# ---- Misalignment ground-truth (test) ----
echo ""
echo "========== Misalignment ground truth (test) =========="
${PYTHON} -m probe_eval.misalignment_ground_truth \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --include-all-negative \\
    --include-behaviors "\${TEST_BEHAVIOR_PATTERNS}" \\
    --output-suffix _test

# ---- Sentence visualization dashboard ----
echo ""
echo "========== Sentence Visualization =========="
${PYTHON} -m probe_eval.visualize_misalignment_sentence \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --layer 27

echo ""
echo "=========================================="
echo "All evaluation complete at \$(date)"
echo "=========================================="
echo "Results:"
echo "  Probe eval:            probe_eval/results/${RESULTS_SUBDIR}/"
echo "  Misalignment GT (dev): probe_eval/results/${RESULTS_SUBDIR}/misalignment_gt_summary_dev.json"
echo "  Misalignment GT (test):probe_eval/results/${RESULTS_SUBDIR}/misalignment_gt_summary_test.json"
echo "  Dashboard:             probe_eval/results/${RESULTS_SUBDIR}/misalignment_sentence_dashboard.html"
EOF

EVAL_GT_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOBS_DEP} "${EVAL_GT_SCRIPT}")
echo "[3c/3] GT + viz job ${EVAL_GT_JOB} submitted (no GPU, after eval jobs: ${EVAL_JOBS_DEP})"

# ---- Summary ----
echo ""
echo "========================================"
echo "SLURM Jobs Submitted"
echo "========================================"
echo "  Pipeline:      v3 (Opus ideation → Sonnet generation)"
echo "  Indicator set: v2.5"
echo "  Indicators:    ${#INDICATORS[@]}"
echo "  Label mode:    ${LABEL_MODE}"
echo "  K:             ${K}"
echo "  Hard neg ratio: ${HARD_NEG_RATIO}"
echo "  Data dir:      ${DATA_DIR}"
echo "  Probe dir:     ${PROBE_DIR}"
echo "  Results:       probe_eval/results/${RESULTS_SUBDIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
