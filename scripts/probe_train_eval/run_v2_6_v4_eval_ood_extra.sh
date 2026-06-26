#!/bin/bash
# =============================================================================
# v2.6 Probe Evaluation — Extra OOD dirs (sycophancy_answer/feedback + mask)
# =============================================================================
#
# Follow-up to run_v2_6_v4_eval.sh covering OOD dirs that weren't in the
# original run:
#   - sycophancy-eval/sycophancy_answer_filtered
#   - sycophancy-eval/sycophancy_feedback_filtered
#   - mask/ (synthetic judgment.json derived from rollout_misalignment_turns.json)
# =============================================================================

set -e

LABEL_MODE="span"
BATCH_SIZE=2

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

OOD_EXTRA_DIRS=(
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer_filtered"
    "${BASE_DIR}/ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback_filtered"
    "${BASE_DIR}/ood_misalignment_eval/mask/rollouts/glm-4.7-flash/mask"
)

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/v2_6_v4_eval_ood_extra_${timestamp}"
mkdir -p "${WORK_DIR}"

EVAL_SCRIPT="${WORK_DIR}/eval.qsh"
cat <<EOF > "${EVAL_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_eval_v26_ood_extra
#SBATCH --output=${WORK_DIR}/eval_%j.out
#SBATCH --error=${WORK_DIR}/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${CACHE_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "v2.6 Probe Evaluation — Extra OOD dirs"
echo "Probe dir:  ${PROBE_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Started: \$(date)"
echo "=========================================="

PROBE_FOLDERS=\$(find "${PROBE_DIR}" -name "cfg.yaml" -exec dirname {} \; | sort | tr '\n' ' ')
echo "Found \$(echo "\${PROBE_FOLDERS}" | wc -w) probes"

echo ""
echo "========== OOD extra: sycophancy_answer/feedback + mask =========="
${PYTHON} -m probe_eval.evaluate \\
    --experiment_folder \${PROBE_FOLDERS} \\
    --rollout_dir ${OOD_EXTRA_DIRS[@]} \\
    --behavior_threshold 5 \\
    --batch_size ${BATCH_SIZE}

echo ""
echo "Extra OOD evaluation complete at \$(date)"
EOF

EVAL_JOB=$(sbatch --parsable "${EVAL_SCRIPT}")
echo "[ood-extra] job ${EVAL_JOB} submitted (1 GPU)"

echo ""
echo "========================================"
echo "  Probe dir:  ${PROBE_DIR}"
echo "  OOD dirs:   ${#OOD_EXTRA_DIRS[@]}"
echo "  Batch size: ${BATCH_SIZE}"
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out"
echo "========================================"
