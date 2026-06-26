#!/bin/bash
# Score neutral dialogues with all probes to find false positives.
#
# For each probe (layer 27), checks which neutral conversation sentences
# exceed the clean_label threshold_f1. Results saved to
# probe/data/neutral/false_positives.json.
#
# Usage:
#   bash scripts/probe_train_eval/run_score_neutral_slurm.sh
#   bash scripts/probe_train_eval/run_score_neutral_slurm.sh --max-dialogues 500

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PROBES_DIR=${BASE_DIR}/probe/probes/v3_v2_5_span
DIALOGUES_PATH=${BASE_DIR}/probe/data/neutral/dialogues_filtered_v3.json
OUTPUT_PATH=${BASE_DIR}/probe/data/neutral/false_positives_v2_5_tuned.json
THRESHOLDS_FILE=${BASE_DIR}/probe_eval/results/v3_v2_5_span/tuned_thresholds.json
THRESHOLDS_VERSION=fpr_0.02
MAX_DIALOGUES=3000
LAYER=27

# Forward extra args (e.g. --max-dialogues 500)
EXTRA_ARGS="$@"

# SLURM settings
PARTITION="general"
MEMORY="128G"
NUM_GPUS=1

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/score_neutral_${timestamp}"
mkdir -p "${WORK_DIR}"

SLURM_SCRIPT="${WORK_DIR}/score_neutral.qsh"

cat <<SLURM_EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=score_neutral
#SBATCH --output=${WORK_DIR}/score_%j.out
#SBATCH --error=${WORK_DIR}/score_%j.err
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Score Neutral Dialogues"
echo "Started: \$(date)"
echo "=========================================="

${PYTHON} -m probe.neutral.score_neutral \\
    --probes-dir "${PROBES_DIR}" \\
    --dialogues-path "${DIALOGUES_PATH}" \\
    --layer ${LAYER} \\
    --output "${OUTPUT_PATH}" \\
    --max-dialogues ${MAX_DIALOGUES} \\
    --thresholds-file "${THRESHOLDS_FILE}" \\
    --thresholds-version "${THRESHOLDS_VERSION}" \\
    ${EXTRA_ARGS}

echo ""
echo "=========================================="
echo "Complete at \$(date)"
echo "Results: ${OUTPUT_PATH}"
echo "=========================================="
SLURM_EOF

echo "Submitting SLURM job..."
sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Job Submitted"
echo "========================================"
echo "  Monitor: tail -f ${WORK_DIR}/score_*.out"
echo "  Output:  ${OUTPUT_PATH}"
echo "========================================"
