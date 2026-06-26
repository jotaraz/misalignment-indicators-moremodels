#!/bin/bash
# =============================================================================
# v2.6 Indicator Probes — V4 Pipeline (Deferred Training, After Generation)
# =============================================================================
#
# Submits 3 SLURM training jobs with --begin=now+NHOURS (default 6h).
# Each SLURM job discovers finished indicators from the generate_v4 logs at
# *execution time* (not submission time), so the set reflects whatever has
# completed generation by the time the job starts.
#
# Splits discovered indicators across 3 jobs (ceil-division).
# train.py's per-layer existence check ensures already-trained probes are
# skipped, so re-running after earlier partial training is safe.
#
# Usage:
#   bash scripts/probe_train_eval/run_v2_6_v4_train_deferred.sh [--hours 6]
#
# Options:
#   --hours N              Delay before job starts (default: 6)
#   --label-mode MODE      turn or span (default: span)
#   --detect-layers L...   Layers to probe (default: "27 29")
# =============================================================================

set -e

HOURS=6
LABEL_MODE="span"
DETECT_LAYERS="27 29"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours) HOURS="$2"; shift 2 ;;
        --label-mode) LABEL_MODE="$2"; shift 2 ;;
        --detect-layers) DETECT_LAYERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

DATA_DIR=${BASE_DIR}/probe/data/v4_v2_6
PROBE_DIR=${BASE_DIR}/probe/probes/v4_v2_6_${LABEL_MODE}

REG_COEFF=10.0
PARTITION="general,overflow"
QOS="high"
TRAIN_MEM="128G"
NUM_SPLITS=3

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/v2_6_v4_train_deferred_${timestamp}"
mkdir -p "${WORK_DIR}"

BEGIN_TIME="now+${HOURS}hours"

for SPLIT_IDX in $(seq 1 ${NUM_SPLITS}); do
    TRAIN_SCRIPT="${WORK_DIR}/train_split${SPLIT_IDX}.qsh"

    cat <<SLURM_SCRIPT > "${TRAIN_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=probe_train_v26_def_s${SPLIT_IDX}
#SBATCH --output=${WORK_DIR}/train_split${SPLIT_IDX}_%j.out
#SBATCH --error=${WORK_DIR}/train_split${SPLIT_IDX}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Training v2.6 Probes (deferred) — split ${SPLIT_IDX}/${NUM_SPLITS}"
echo "Label mode: ${LABEL_MODE}"
echo "Detect layers: ${DETECT_LAYERS}"
echo "Started: \$(date)"
echo "=========================================="

# ---- Discover finished indicators at JOB execution time ----
# Parse "Saved to <path>" lines from all generate_v4 logs.
FINISHED_FILES=\$(grep -h "^  Saved to " ${BASE_DIR}/logs/generate_v4_*.log 2>/dev/null \\
    | awk '{print \$3}' | sort -u)

if [ -z "\${FINISHED_FILES}" ]; then
    echo "ERROR: no finished indicators found in generate_v4 logs"
    exit 1
fi

# Build per-split indicator list via modulo
ALL_INDICATORS=()
idx=0
for f in \${FINISHED_FILES}; do
    if [ -f "\$f" ]; then
        # Assign to split by round-robin
        if [ \$((idx % ${NUM_SPLITS})) -eq $((SPLIT_IDX - 1)) ]; then
            name=\$(${PYTHON} -c "import json,sys; print(json.load(open(sys.argv[1]))['indicator_name'])" "\$f")
            ALL_INDICATORS+=("\$name")
        fi
        idx=\$((idx + 1))
    fi
done

echo "Split ${SPLIT_IDX} will train \${#ALL_INDICATORS[@]} indicators:"
for ind in "\${ALL_INDICATORS[@]}"; do echo "  - \${ind}"; done
echo ""

for ind in "\${ALL_INDICATORS[@]}"; do
    echo ""
    echo "--- Training: \${ind} ---"
    ${PYTHON} -m probe.train \\
        --indicator-set v2_6 \\
        --indicator "\${ind}" \\
        --label-mode "${LABEL_MODE}" \\
        --reg-coeff ${REG_COEFF} \\
        --detect-layers ${DETECT_LAYERS} \\
        --reasoning-only \\
        --threshold-mode sentence \\
        --val-fraction 0.2 \\
        --data-dir "${DATA_DIR}" \\
        --output-dir "${PROBE_DIR}"
done

echo ""
echo "Training split ${SPLIT_IDX} complete at \$(date)"
SLURM_SCRIPT

    TRAIN_JOB=$(sbatch --parsable --begin="${BEGIN_TIME}" "${TRAIN_SCRIPT}")
    echo "[train-deferred] split ${SPLIT_IDX} job ${TRAIN_JOB} submitted (begins ${BEGIN_TIME})"
done

echo ""
echo "========================================"
echo "Deferred SLURM Jobs Submitted"
echo "========================================"
echo "  Indicator set: v2.6"
echo "  Start time:    ${BEGIN_TIME}  (approx $(date -d "+${HOURS} hours"))"
echo "  Splits:        ${NUM_SPLITS}"
echo "  Label mode:    ${LABEL_MODE}"
echo "  Data dir:      ${DATA_DIR}"
echo "  Probe dir:     ${PROBE_DIR}"
echo ""
echo "Monitor:"
echo "  squeue -u $(whoami)"
echo "  tail -f ${WORK_DIR}/*.out  (after job starts)"
echo "========================================"
