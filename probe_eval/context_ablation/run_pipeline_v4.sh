#!/bin/bash
# =============================================================================
# Context-Ablation Analysis Pipeline — v4_v2_6_combined_v2_span
# =============================================================================
#
# Runs the ANALYSIS-only part of the context_ablation pipeline (no training)
# against the v4_v2_6_combined_v2_span probe set:
#
#   1. Select context-dependent spans + generate benign contexts
#      (reads v4 dev context_dependency_raw.json, optional v4
#       keyword_overlap_raw.json if it exists, training data)
#   2. Build rollout transcripts (no / positive / negative context)
#   3. Extract activations on the v4 probe set (needs GPU)
#   4. Analyze probe-score shifts + residual-stream deltas
#
# Output under probe_eval/context_ablation/data/v4_v2_6_combined_v2_span/.
#
# Usage:
#   bash probe_eval/context_ablation/run_pipeline_v4.sh            # SLURM
#   bash probe_eval/context_ablation/run_pipeline_v4.sh --local    # inline
# =============================================================================

set -e

LOCAL_MODE=false
[[ "${1}" == "--local" ]] && LOCAL_MODE=true

# ---- Config ----
RESULTS_SUBDIR="v4_v2_6_combined_v2_span"
BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON_API=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PYTHON_GPU=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

PROBE_DIR=${BASE_DIR}/probe/probes/${RESULTS_SUBDIR}
OUTPUT_ROOT=${BASE_DIR}/probe_eval/context_ablation/data/${RESULTS_SUBDIR}
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts
ACT_DIR=${OUTPUT_ROOT}/activations
ANALYSIS_DIR=${OUTPUT_ROOT}/analysis
LAYER=27

PARTITION="general,overflow"
QOS="high"

mkdir -p "${OUTPUT_ROOT}"

echo "=========================================="
echo "Context-ablation analysis pipeline"
echo "  Results subdir : ${RESULTS_SUBDIR}"
echo "  Probe dir      : ${PROBE_DIR}"
echo "  Output root    : ${OUTPUT_ROOT}"
echo "  Layer          : ${LAYER}"
echo "=========================================="

# =============================================================================
# Local (inline) mode
# =============================================================================
if ${LOCAL_MODE}; then
    cd "${BASE_DIR}"
    source "${BASE_DIR}/.env"
    export HF_HOME=/workspace-vast/pretrained_ckpts

    echo ""
    echo "=== Step 0.5/6: Classify indicator spans (standalone vs context-dependent) ==="
    INDICATOR_CTX_FILE="${BASE_DIR}/probe_eval/results/${RESULTS_SUBDIR}/indicator_span_context_dependency_raw.json"
    if [[ -f "${INDICATOR_CTX_FILE}" ]]; then
        echo "  [skip] ${INDICATOR_CTX_FILE} already exists"
    else
        ${PYTHON_API} -m probe_eval.analyze_indicator_span_context_dependency \
            --results-subdir "${RESULTS_SUBDIR}" \
            --max-concurrent 60
    fi

    echo ""
    echo "=== Step 1/6: Select context-dependent spans ==="
    ${PYTHON_API} -m probe_eval.context_ablation.select_and_generate \
        --results-subdir "${RESULTS_SUBDIR}" \
        --output-dir "${OUTPUT_ROOT}" \
        --context-dependency-file "${BASE_DIR}/probe_eval/results/${RESULTS_SUBDIR}/indicator_span_context_dependency_raw.json" \
        --training-data-dir "${BASE_DIR}/probe/data/v4_v2_6_combined_v2" \
        --max-transcripts 15 \
        --max-dev-per-indicator 40 \
        --max-concurrent 60

    echo ""
    echo "=== Step 1.5/6: Re-classify sentence spans with stricter standalone criterion ==="
    ${PYTHON_API} -m probe_eval.context_ablation.reclassify_sentences \
        --input "${OUTPUT_ROOT}/spans_with_contexts.json" \
        --output "${OUTPUT_ROOT}/sentence_reclassification.json" \
        --max-concurrent 60

    echo ""
    echo "=== Step 2/6: Build rollouts ==="
    ${PYTHON_API} -m probe_eval.context_ablation.build_rollouts \
        --input "${OUTPUT_ROOT}/spans_with_contexts.json" \
        --output-dir "${ROLLOUT_DIR}"

    echo ""
    echo "=== Step 3/6: Extract activations (requires GPU) ==="
    ${PYTHON_GPU} -m probe_eval.context_ablation.extract_activations \
        --rollout-dir "${ROLLOUT_DIR}" \
        --probe-dir "${PROBE_DIR}" \
        --layers ${LAYER} \
        --output-dir "${ACT_DIR}"

    echo ""
    echo "=== Step 4/6: Analyze probe-score shifts + deltas ==="
    ${PYTHON_GPU} -m probe_eval.context_ablation.analyze \
        --activations-dir "${ACT_DIR}" \
        --probe-dir "${PROBE_DIR}" \
        --layer ${LAYER} \
        --data-dir "${OUTPUT_ROOT}" \
        --output-dir "${ANALYSIS_DIR}"

    echo ""
    echo "=========================================="
    echo "Done — outputs in ${OUTPUT_ROOT}"
    echo "=========================================="
    exit 0
fi

# =============================================================================
# SLURM mode (3 chained jobs)
# =============================================================================
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/context_ablation_${RESULTS_SUBDIR}_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Job 1: select + build (no GPU) ----
cat <<EOF > "${WORK_DIR}/step1_generate.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_v4_gen
#SBATCH --output=${WORK_DIR}/step1_%j.out
#SBATCH --error=${WORK_DIR}/step1_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
source ${BASE_DIR}/.env

echo "=== Step 0.5: analyze_indicator_span_context_dependency ==="
INDICATOR_CTX_FILE=${BASE_DIR}/probe_eval/results/${RESULTS_SUBDIR}/indicator_span_context_dependency_raw.json
if [[ -f "\${INDICATOR_CTX_FILE}" ]]; then
    echo "  [skip] \${INDICATOR_CTX_FILE} already exists"
else
    ${PYTHON_API} -m probe_eval.analyze_indicator_span_context_dependency \\
        --results-subdir ${RESULTS_SUBDIR} \\
        --max-concurrent 60
fi

echo ""
echo "=== Step 1: select_and_generate ==="
${PYTHON_API} -m probe_eval.context_ablation.select_and_generate \\
    --results-subdir ${RESULTS_SUBDIR} \\
    --output-dir ${OUTPUT_ROOT} \\
    --context-dependency-file ${BASE_DIR}/probe_eval/results/${RESULTS_SUBDIR}/indicator_span_context_dependency_raw.json \\
    --training-data-dir ${BASE_DIR}/probe/data/v4_v2_6_combined_v2 \\
    --max-transcripts 15 \\
    --max-dev-per-indicator 40 \\
    --max-concurrent 60

echo ""
echo "=== Step 1.5: reclassify_sentences (stricter standalone filter) ==="
${PYTHON_API} -m probe_eval.context_ablation.reclassify_sentences \\
    --input ${OUTPUT_ROOT}/spans_with_contexts.json \\
    --output ${OUTPUT_ROOT}/sentence_reclassification.json \\
    --max-concurrent 60

echo ""
echo "=== Step 2: build_rollouts ==="
${PYTHON_API} -m probe_eval.context_ablation.build_rollouts \\
    --input ${OUTPUT_ROOT}/spans_with_contexts.json \\
    --output-dir ${ROLLOUT_DIR}

echo "Step 1 done at \$(date)"
EOF
JOB1=$(sbatch --parsable "${WORK_DIR}/step1_generate.qsh")
echo "[1/3] select+build     job ${JOB1}"

# ---- Job 2: extract activations (1 GPU) ----
cat <<EOF > "${WORK_DIR}/step2_extract.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_v4_extract
#SBATCH --output=${WORK_DIR}/step2_%j.out
#SBATCH --error=${WORK_DIR}/step2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=128G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=== Step 3: extract_activations ==="
${PYTHON_GPU} -m probe_eval.context_ablation.extract_activations \\
    --rollout-dir ${ROLLOUT_DIR} \\
    --probe-dir ${PROBE_DIR} \\
    --layers ${LAYER} \\
    --output-dir ${ACT_DIR}

echo "Step 2 done at \$(date)"
EOF
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} "${WORK_DIR}/step2_extract.qsh")
echo "[2/3] extract          job ${JOB2} (1 GPU, after ${JOB1})"

# ---- Job 3: analyze (no GPU) ----
cat <<EOF > "${WORK_DIR}/step3_analyze.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_v4_analyze
#SBATCH --output=${WORK_DIR}/step3_%j.out
#SBATCH --error=${WORK_DIR}/step3_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=64G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=== Step 4: analyze ==="
${PYTHON_GPU} -m probe_eval.context_ablation.analyze \\
    --activations-dir ${ACT_DIR} \\
    --probe-dir ${PROBE_DIR} \\
    --layer ${LAYER} \\
    --data-dir ${OUTPUT_ROOT} \\
    --output-dir ${ANALYSIS_DIR}

echo "Step 3 done at \$(date)"
EOF
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} "${WORK_DIR}/step3_analyze.qsh")
echo "[3/3] analyze          job ${JOB3} (no GPU, after ${JOB2})"

echo ""
echo "=========================================="
echo "Pipeline: ${JOB1} -> ${JOB2} -> ${JOB3}"
echo "Outputs:  ${OUTPUT_ROOT}"
echo "Monitor:  tail -f ${WORK_DIR}/*.out"
echo "=========================================="
