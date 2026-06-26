#!/bin/bash
# Neutral PCA denoising pipeline for probe cleaning.
#
# 4 steps:
#   1. Sample 10k diverse neutral prompts (WildChat, LMSYS, FLAN/NI)
#   2. Generate on-policy completions from GLM-4.7 Flash + extract activations
#   3. Compute PCA on neutral activations + check probe overlap
#   4. Clean probes by projecting out top PCs
#
# Steps 1 and 3-4 are CPU-only. Step 2 requires GPU (submitted via SLURM).
# The script runs step 1 locally, submits step 2 as a SLURM job, then
# step 2's job chains steps 3-4 after extraction completes.
#
# Usage:
#   bash scripts/probe_train_eval/run_neutral_pca_clean_slurm.sh
#   bash scripts/probe_train_eval/run_neutral_pca_clean_slurm.sh --step 2   # skip sampling
#   bash scripts/probe_train_eval/run_neutral_pca_clean_slurm.sh --step 3   # skip gen+extract (already done)

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

# Paths
PROMPTS_PATH=${BASE_DIR}/probe/data/neutral/prompts.json
ACTIVATIONS_DIR=${BASE_DIR}/probe/data/neutral/activations
PCA_DIR=${BASE_DIR}/probe/data/neutral/pca
DIALOGUES_PATH=${BASE_DIR}/probe/data/neutral/dialogues_filtered_v2.json

# Probe dirs
PROBES_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2
CLEAN_PROBES_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2_clean

# Params
N_PROMPTS=10000
LAYERS="27 28 29 30"
TENSOR_PARALLEL=2
EXTRACT_BATCH_SIZE=32
MAX_NEW_TOKENS=4096
MAX_TOKENS_PER_LAYER=8000000
N_PCA_COMPONENTS=50
N_COMPONENTS_REMOVE=10

# SLURM settings
PARTITION="general"
MEMORY="128G"
NUM_GPUS=2

# ---- Parse args ----
START_STEP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) START_STEP="$2"; shift 2 ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        --n-remove) N_COMPONENTS_REMOVE="$2"; shift 2 ;;
        --tp) TENSOR_PARALLEL="$2"; shift 2 ;;
        --extract-batch-size) EXTRACT_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/neutral_pca_clean_${timestamp}"
mkdir -p "${WORK_DIR}"

echo "=========================================="
echo "Neutral PCA Denoising Pipeline"
echo "=========================================="
echo "  Start step:        ${START_STEP}"
echo "  N prompts:         ${N_PROMPTS}"
echo "  Layers:            ${LAYERS}"
echo "  PCA components:    ${N_PCA_COMPONENTS}"
echo "  PCs to remove:     ${N_COMPONENTS_REMOVE}"
echo "  Probes dir:        ${PROBES_DIR}"
echo "  Clean probes dir:  ${CLEAN_PROBES_DIR}"
echo "  Log dir:           ${WORK_DIR}"
echo "=========================================="
echo ""

# ---- Step 1: Sample prompts (CPU, runs locally) ----
if [ "${START_STEP}" -le 1 ]; then
    echo ">>> Step 1/4: Sampling ${N_PROMPTS} neutral prompts"
    echo "    (WildChat 50%, LMSYS 30%, FLAN/NI 20%)"
    echo ""

    cd "${BASE_DIR}"
    ${PYTHON} -m probe.neutral.sample_prompts \
        --n-total ${N_PROMPTS} \
        --output "${PROMPTS_PATH}" \
        2>&1 | tee "${WORK_DIR}/step1_sample.log"

    echo ""
    echo ">>> Step 1 complete: ${PROMPTS_PATH}"
    echo ""
fi

# ---- Steps 2-4: GPU job (SLURM) ----
# Step 2 needs GPU; steps 3-4 are CPU but chained for convenience.

if [ "${START_STEP}" -le 2 ]; then
    SLURM_SCRIPT="${WORK_DIR}/steps_2_3_4.qsh"

    cat <<SLURM_EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=neutral_pca
#SBATCH --output=${WORK_DIR}/steps_%j.out
#SBATCH --error=${WORK_DIR}/steps_%j.err
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Neutral PCA Pipeline — Steps 2-4"
echo "Started: \$(date)"
echo "=========================================="

# ---- Step 2: Generate + Extract Activations ----
echo ""
echo ">>> Step 2/4: Generating on-policy completions + extracting activations"
echo "    vLLM tp=${TENSOR_PARALLEL}, Extract batch: ${EXTRACT_BATCH_SIZE}, Max new tokens: ${MAX_NEW_TOKENS}"
echo "    Layers: ${LAYERS}"
echo ""

${PYTHON} -m probe.neutral.extract_activations \\
    --prompts-path "${PROMPTS_PATH}" \\
    --output-dir "${ACTIVATIONS_DIR}" \\
    --layers ${LAYERS} \\
    --tp ${TENSOR_PARALLEL} \\
    --extract-batch-size ${EXTRACT_BATCH_SIZE} \\
    --max-new-tokens ${MAX_NEW_TOKENS} \\
    --max-tokens-per-layer ${MAX_TOKENS_PER_LAYER}

echo ""
echo ">>> Step 2 complete: \$(date)"
echo "    Activations saved to ${ACTIVATIONS_DIR}"
echo ""

# ---- Step 3: PCA Analysis ----
echo ">>> Step 3/4: Computing PCA + analyzing probe overlap"
echo "    N components: ${N_PCA_COMPONENTS}"
echo ""

${PYTHON} -m probe.neutral.pca_analysis \\
    --activations-dir "${ACTIVATIONS_DIR}" \\
    --probes-dir "${PROBES_DIR}" \\
    --output-dir "${PCA_DIR}" \\
    --n-components ${N_PCA_COMPONENTS} \\
    --layers ${LAYERS}

echo ""
echo ">>> Step 3 complete: \$(date)"
echo "    PCA saved to ${PCA_DIR}"
echo ""

# ---- Step 4: Clean Probes ----
echo ">>> Step 4/4: Cleaning probes (removing top ${N_COMPONENTS_REMOVE} PCs)"
echo ""

${PYTHON} -m probe.neutral.clean_probes \\
    --pca-dir "${PCA_DIR}" \\
    --probes-dir "${PROBES_DIR}" \\
    --output-dir "${CLEAN_PROBES_DIR}" \\
    --n-components ${N_COMPONENTS_REMOVE} \\
    --layers ${LAYERS}

echo ""
echo "=========================================="
echo "Pipeline complete: \$(date)"
echo "  PCA results:    ${PCA_DIR}"
echo "  Cleaned probes: ${CLEAN_PROBES_DIR}"
echo "=========================================="
SLURM_EOF

    echo ">>> Submitting SLURM job for steps 2-4..."
    sbatch "${SLURM_SCRIPT}"

    echo ""
    echo "========================================"
    echo "SLURM Job Submitted (Steps 2-4)"
    echo "========================================"
    echo "  Monitor: tail -f ${WORK_DIR}/steps_*.out"
    echo "========================================"

elif [ "${START_STEP}" -eq 3 ]; then
    # Steps 3-4 only (no GPU needed, run locally)
    echo ">>> Step 3/4: Computing PCA + analyzing probe overlap"
    cd "${BASE_DIR}"

    ${PYTHON} -m probe.neutral.pca_analysis \
        --activations-dir "${ACTIVATIONS_DIR}" \
        --probes-dir "${PROBES_DIR}" \
        --output-dir "${PCA_DIR}" \
        --n-components ${N_PCA_COMPONENTS} \
        --layers ${LAYERS} \
        2>&1 | tee "${WORK_DIR}/step3_pca.log"

    echo ""
    echo ">>> Step 4/4: Cleaning probes (removing top ${N_COMPONENTS_REMOVE} PCs)"

    ${PYTHON} -m probe.neutral.clean_probes \
        --pca-dir "${PCA_DIR}" \
        --probes-dir "${PROBES_DIR}" \
        --output-dir "${CLEAN_PROBES_DIR}" \
        --n-components ${N_COMPONENTS_REMOVE} \
        --layers ${LAYERS} \
        2>&1 | tee "${WORK_DIR}/step4_clean.log"

    echo ""
    echo "========================================"
    echo "Pipeline complete"
    echo "  PCA results:    ${PCA_DIR}"
    echo "  Cleaned probes: ${CLEAN_PROBES_DIR}"
    echo "========================================"

elif [ "${START_STEP}" -eq 4 ]; then
    echo ">>> Step 4/4: Cleaning probes (removing top ${N_COMPONENTS_REMOVE} PCs)"
    cd "${BASE_DIR}"

    ${PYTHON} -m probe.neutral.clean_probes \
        --pca-dir "${PCA_DIR}" \
        --probes-dir "${PROBES_DIR}" \
        --output-dir "${CLEAN_PROBES_DIR}" \
        --n-components ${N_COMPONENTS_REMOVE} \
        --layers ${LAYERS} \
        2>&1 | tee "${WORK_DIR}/step4_clean.log"

    echo ""
    echo "========================================"
    echo "Cleaning complete: ${CLEAN_PROBES_DIR}"
    echo "========================================"
fi
