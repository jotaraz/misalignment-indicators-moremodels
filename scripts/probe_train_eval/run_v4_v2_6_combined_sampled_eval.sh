#!/bin/bash
# =============================================================================
# Evaluate sampled combined probes on 10 dev + 15 test sets.
#
# Probe dirs:
#   1probe_sampled  -> probe/probes/v4_v2_6_combined_v2_1probe_sampled_span
#   behavior_sampled -> probe/probes/v4_v2_6_combined_v2_behavior_sampled_span
#
# Dev (10):   5 positive + 5 benign bloom
# Test (15):  5 positive + 5 benign bloom + 5 OOD
# =============================================================================

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
BATCH_SIZE=4
PARTITION="general,overflow"
QOS="high"
CACHE_MEM="128G"

DEV_POS=(
    "${BASE_DIR}/bloom/bloom-results/instructed-covert-code-sabotage_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-strategic-sandbagging_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_glm_4_7_flash"
)
DEV_BENIGN=(
    "${BASE_DIR}/bloom/bloom-results/instructed-covert-code-sabotage_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/instructed-strategic-sandbagging_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/self-preservation_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results/strategic-deception_benign_glm_4_7_flash"
)
TEST_POS=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-covert-code-sabotage_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-strategic-sandbagging_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_glm_4_7_flash"
)
TEST_BENIGN=(
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-covert-code-sabotage_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_instructed-strategic-sandbagging_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_sycophancy_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_self-preservation_benign_glm_4_7_flash"
    "${BASE_DIR}/bloom/bloom-results-test/test_strategic-deception_benign_glm_4_7_flash"
)
OOD=(
    "${BASE_DIR}/ood_misalignment_eval/agentic-misalignment/results/ood_eval_glm/bloom_rollout"
    "${BASE_DIR}/ood_misalignment_eval/mask/rollouts/glm-4.7-flash/mask"
    "${BASE_DIR}/ood_misalignment_eval/deception-bench/rollouts/glm-4.7-flash/bloom"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer_filtered"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback_filtered"
)

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/combined_sampled_eval_${timestamp}"
mkdir -p "${WORK_DIR}"

CONFIGS=(
    "1probe_sampled:probe/probes/v4_v2_6_combined_v2_1probe_sampled_span"
    "behavior_sampled:probe/probes/v4_v2_6_combined_v2_behavior_sampled_span"
)

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r NAME PROBE_SUBDIR <<< "${CFG}"
    PROBE_DIR="${BASE_DIR}/${PROBE_SUBDIR}"

    SCRIPT="${WORK_DIR}/eval_${NAME}.qsh"

    cat > "${SCRIPT}" <<SLURM
#!/bin/bash
#SBATCH --job-name=eval_${NAME}
#SBATCH --output=${WORK_DIR}/eval_${NAME}_%j.out
#SBATCH --error=${WORK_DIR}/eval_${NAME}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Eval: ${NAME}"
echo "Probe dir:  ${PROBE_DIR}"
echo "Started:    \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

echo ""
echo "========== Dev positive (5) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_POS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== Dev benign (5, all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${DEV_BENIGN[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== Test positive (5) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_POS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== Test benign (5, all_negative) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${TEST_BENIGN[@]} \\
    --all_negative \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "========== OOD (5) =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "${NAME} eval complete at \$(date)"
SLURM

    JOB=$(sbatch --parsable "${SCRIPT}")
    echo "[${NAME}] submitted job ${JOB}  probes=${PROBE_SUBDIR}"
done

echo ""
echo "Logs: ${WORK_DIR}"
echo "Monitor: squeue -u $(whoami)"
