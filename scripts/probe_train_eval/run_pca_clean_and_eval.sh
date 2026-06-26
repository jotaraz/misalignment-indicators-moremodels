#!/bin/bash
# Combined pipeline: PCA denoise probes, recompute thresholds, score neutral FPs.
#
# Steps:
#   1. Compute PCA on neutral activations + check probe overlap (CPU)
#   2. Clean probes by projecting out top PCs (CPU)
#   3. Recompute sentence-level thresholds for cleaned probes (GPU)
#   4. Score neutral dialogues to find false positives (GPU)
#
# Usage (on a GPU node):
#   bash scripts/probe_train_eval/run_pca_clean_and_eval.sh
#   bash scripts/probe_train_eval/run_pca_clean_and_eval.sh --step 2   # skip PCA (already computed)
#   bash scripts/probe_train_eval/run_pca_clean_and_eval.sh --step 3   # skip PCA + cleaning
#   bash scripts/probe_train_eval/run_pca_clean_and_eval.sh --step 4   # only score neutral FPs

set -e

# ---- Configuration ----
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

# Paths
ACTIVATIONS_DIR=${BASE_DIR}/probe/data/neutral/activations
PCA_DIR=${BASE_DIR}/probe/data/neutral/pca
DIALOGUES_PATH=${BASE_DIR}/probe/data/neutral/dialogues_filtered_v2.json
DATA_DIR=${BASE_DIR}/probe/data/v2_3_gen_prompt_v2

# Probe dirs
PROBES_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2
CLEAN_PROBES_DIR=${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2_clean_50

# Params
LAYERS="27 28 29 30"
N_PCA_COMPONENTS=100
N_COMPONENTS_REMOVE=50
SCORE_LAYER=27

# ---- Parse args ----
START_STEP=1
EXTRA_SCORE_ARGS="--max-dialogues 2000"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) START_STEP="$2"; shift 2 ;;
        --n-remove) N_COMPONENTS_REMOVE="$2"; shift 2 ;;
        --max-dialogues) EXTRA_SCORE_ARGS="--max-dialogues $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1
cd "${BASE_DIR}"

echo "=========================================="
echo "PCA Denoise + Threshold + Neutral FP Pipeline"
echo "=========================================="
echo "  Start step:        ${START_STEP}"
echo "  Probes dir:        ${PROBES_DIR}"
echo "  Clean probes dir:  ${CLEAN_PROBES_DIR}"
echo "  PCs to remove:     ${N_COMPONENTS_REMOVE}"
echo "  Score layer:       ${SCORE_LAYER}"
echo "=========================================="
echo ""

# ---- Step 1: PCA Analysis (CPU) ----
if [ "${START_STEP}" -le 1 ]; then
    echo ">>> Step 1/4: Computing PCA on neutral activations"
    echo ""

    ${PYTHON} -m probe.neutral.pca_analysis \
        --activations-dir "${ACTIVATIONS_DIR}" \
        --probes-dir "${PROBES_DIR}" \
        --output-dir "${PCA_DIR}" \
        --n-components ${N_PCA_COMPONENTS} \
        --layers ${LAYERS}

    echo ""
    echo ">>> Step 1 complete"
    echo ""
fi

# ---- Step 2: Clean Probes (CPU) ----
if [ "${START_STEP}" -le 2 ]; then
    echo ">>> Step 2/4: Cleaning probes (removing top ${N_COMPONENTS_REMOVE} PCs)"
    echo ""

    ${PYTHON} -m probe.neutral.clean_probes \
        --pca-dir "${PCA_DIR}" \
        --probes-dir "${PROBES_DIR}" \
        --output-dir "${CLEAN_PROBES_DIR}" \
        --n-components ${N_COMPONENTS_REMOVE} \
        --layers ${LAYERS}

    echo ""
    echo ">>> Step 2 complete: ${CLEAN_PROBES_DIR}"
    echo ""
fi

# ---- Step 3: Recompute Thresholds (GPU) ----
if [ "${START_STEP}" -le 3 ]; then
    echo ">>> Step 3/4: Recomputing sentence-level thresholds for cleaned probes"
    echo ""

    ${PYTHON} -m probe.recompute_thresholds \
        --probe-dir "${CLEAN_PROBES_DIR}" \
        --data-dir "${DATA_DIR}"

    echo ""
    echo ">>> Step 3 complete"
    echo ""
fi

# ---- Step 4: Score Neutral FPs (GPU) ----
if [ "${START_STEP}" -le 4 ]; then
    echo ">>> Step 4/4: Scoring neutral dialogues for false positives"
    echo ""

    ${PYTHON} -m probe.neutral.score_neutral \
        --probes-dir "${CLEAN_PROBES_DIR}" \
        --dialogues-path "${DIALOGUES_PATH}" \
        --layer ${SCORE_LAYER} \
        --output ${BASE_DIR}/probe/data/neutral/false_positives_v2_clean_50.json \
        ${EXTRA_SCORE_ARGS}

    echo ""
    echo ">>> Step 4 complete"
    echo ""
fi

echo "=========================================="
echo "Pipeline complete at $(date)"
echo "  Cleaned probes: ${CLEAN_PROBES_DIR}"
echo "  FP results:     ${BASE_DIR}/probe/data/neutral/false_positives.json"
echo "=========================================="
