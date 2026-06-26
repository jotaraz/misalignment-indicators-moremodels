#!/bin/bash
# =============================================================================
# Train combined-sampled probes for v4_v2_6_combined_v2.
#
# Two configs:
#   1. 1-probe:     probe/data/v4_v2_6_combined_v2_1probe_sampled  -> 1 probe
#      (indicator_name="misalignment")
#   2. per-behavior: probe/data/v4_v2_6_combined_v2_behavior_sampled -> 5 probes
#      (one per base behavior)
#
# Each config = one SLURM job with 1 GPU. Uses --all so the indicator names
# are read from each data file's JSON.
# =============================================================================

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
LABEL_MODE="span"
DETECT_LAYERS="27 29"
REG_COEFF=10.0
PARTITION="general,overflow"
QOS="high"
TRAIN_MEM="128G"

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/combined_sampled_train_${timestamp}"
mkdir -p "${WORK_DIR}"

# config_name, data_subdir, output_subdir
CONFIGS=(
    "1probe_sampled:v4_v2_6_combined_v2_1probe_sampled:v4_v2_6_combined_v2_1probe_sampled_${LABEL_MODE}"
    "behavior_sampled:v4_v2_6_combined_v2_behavior_sampled:v4_v2_6_combined_v2_behavior_sampled_${LABEL_MODE}"
)

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r NAME DATA OUT <<< "${CFG}"
    DATA_DIR="${BASE_DIR}/probe/data/${DATA}"
    PROBE_DIR="${BASE_DIR}/probe/probes/${OUT}"

    TRAIN_SCRIPT="${WORK_DIR}/train_${NAME}.qsh"

    cat > "${TRAIN_SCRIPT}" <<SLURM
#!/bin/bash
#SBATCH --job-name=probe_train_${NAME}
#SBATCH --output=${WORK_DIR}/train_${NAME}_%j.out
#SBATCH --error=${WORK_DIR}/train_${NAME}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Training ${NAME} probes"
echo "Data dir:  ${DATA_DIR}"
echo "Probe dir: ${PROBE_DIR}"
echo "Label:     ${LABEL_MODE}    layers: ${DETECT_LAYERS}"
echo "Started:   \$(date)"
echo "=========================================="

${PYTHON} -m probe.train \\
    --indicator-set v2_6 \\
    --all \\
    --label-mode "${LABEL_MODE}" \\
    --reg-coeff ${REG_COEFF} \\
    --detect-layers ${DETECT_LAYERS} \\
    --reasoning-positive-only \\
    --threshold-mode sentence \\
    --val-fraction 0.2 \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${PROBE_DIR}"

echo ""
echo "${NAME} complete at \$(date)"
SLURM

    JOB=$(sbatch --parsable "${TRAIN_SCRIPT}")
    echo "[${NAME}] submitted job ${JOB}  data=${DATA}  out=${OUT}"
done

echo ""
echo "Logs: ${WORK_DIR}"
echo "Monitor: squeue -u $(whoami)"
