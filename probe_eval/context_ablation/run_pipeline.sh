#!/bin/bash
# =============================================================================
# Context Ablation Pipeline
# =============================================================================
#
# Measures context-dependence of probe detection:
#   Job 1 (no GPU):  Select spans + generate benign contexts + build rollouts
#   Job 2 (1 GPU):   Extract activations + probe scores
#   Job 3 (no GPU):  Analyze deltas and alignment
#
# Usage:
#   bash probe_eval/context_ablation/run_pipeline.sh
#   bash probe_eval/context_ablation/run_pipeline.sh --local
# =============================================================================

set -e

LOCAL_MODE=false
[[ "${1}" == "--local" ]] && LOCAL_MODE=true

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON_API=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PYTHON_GPU=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python

PARTITION="general,overflow"
QOS="high"

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/context_ablation_${timestamp}"
mkdir -p "${WORK_DIR}"

# ---- Local mode ----
if ${LOCAL_MODE}; then
    cd "${BASE_DIR}"
    source "${BASE_DIR}/.env"
    export HF_HOME=/workspace-vast/pretrained_ckpts

    echo "=== Step 1: Select context-dependent spans + generate benign contexts ==="
    ${PYTHON_API} -m probe_eval.context_ablation.select_and_generate

    echo "=== Step 2: Build rollouts ==="
    ${PYTHON_API} -m probe_eval.context_ablation.build_rollouts

    echo "=== Step 3: Extract activations (requires GPU) ==="
    ${PYTHON_GPU} -m probe_eval.context_ablation.extract_activations

    echo "=== Step 4: Analyze ==="
    ${PYTHON_GPU} -m probe_eval.context_ablation.analyze

    echo "=== Step 5: Generate training data ==="
    ${PYTHON_API} -m probe_eval.context_ablation.generate_training_data

    echo "=== Step 6: Train + eval probes (requires GPU) ==="
    ${PYTHON_GPU} -m probe_eval.context_ablation.train_and_eval

    echo "Done!"
    exit 0
fi

# ---- SLURM mode ----

# Job 1: Select spans + generate contexts + build rollouts (no GPU)
cat <<EOF > "${WORK_DIR}/step1_generate.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_abl_gen
#SBATCH --output=${WORK_DIR}/step1_%j.out
#SBATCH --error=${WORK_DIR}/step1_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
source ${BASE_DIR}/.env

echo "=== Step 1: Select context-dependent spans + generate benign contexts ==="
${PYTHON_API} -m probe_eval.context_ablation.select_and_generate

echo ""
echo "=== Step 2: Build rollouts ==="
${PYTHON_API} -m probe_eval.context_ablation.build_rollouts

echo "Step 1 done at \$(date)"
EOF

JOB1=$(sbatch --parsable "${WORK_DIR}/step1_generate.qsh")
echo "[1/3] Generate job ${JOB1} (no GPU)"

# Job 2: Extract activations (1 GPU)
cat <<EOF > "${WORK_DIR}/step2_extract.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_abl_extract
#SBATCH --output=${WORK_DIR}/step2_%j.out
#SBATCH --error=${WORK_DIR}/step2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=128G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=== Step 2: Extract activations ==="
${PYTHON_GPU} -m probe_eval.context_ablation.extract_activations

echo "Step 2 done at \$(date)"
EOF

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} "${WORK_DIR}/step2_extract.qsh")
echo "[2/3] Extract job ${JOB2} (1 GPU, after ${JOB1})"

# Job 3: Analyze (no GPU)
cat <<EOF > "${WORK_DIR}/step3_analyze.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_abl_analyze
#SBATCH --output=${WORK_DIR}/step3_%j.out
#SBATCH --error=${WORK_DIR}/step3_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=64G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=== Step 3: Analyze deltas ==="
${PYTHON_GPU} -m probe_eval.context_ablation.analyze

echo "Step 3 done at \$(date)"
EOF

JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} "${WORK_DIR}/step3_analyze.qsh")
echo "[3/5] Analyze job ${JOB3} (no GPU, after ${JOB2})"

# Job 4: Generate training data (no GPU, after step 3)
cat <<EOF > "${WORK_DIR}/step4_training_data.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_abl_traindata
#SBATCH --output=${WORK_DIR}/step4_%j.out
#SBATCH --error=${WORK_DIR}/step4_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=32G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate

echo "=== Step 4: Generate training data ==="
${PYTHON_API} -m probe_eval.context_ablation.generate_training_data

echo "Step 4 done at \$(date)"
EOF

JOB4=$(sbatch --parsable --dependency=afterok:${JOB3} "${WORK_DIR}/step4_training_data.qsh")
echo "[4/5] Training data job ${JOB4} (no GPU, after ${JOB3})"

# Job 5: Train + eval probes (1 GPU, after step 4)
cat <<EOF > "${WORK_DIR}/step5_train_eval.qsh"
#!/bin/bash
#SBATCH --job-name=ctx_abl_train
#SBATCH --output=${WORK_DIR}/step5_%j.out
#SBATCH --error=${WORK_DIR}/step5_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --mem=128G
#SBATCH --chdir=${BASE_DIR}

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "=== Step 5: Train + eval probes ==="
${PYTHON_GPU} -m probe_eval.context_ablation.train_and_eval

echo "Step 5 done at \$(date)"
EOF

JOB5=$(sbatch --parsable --dependency=afterok:${JOB4} "${WORK_DIR}/step5_train_eval.qsh")
echo "[5/5] Train+eval job ${JOB5} (1 GPU, after ${JOB4})"

echo ""
echo "========================================"
echo "Pipeline: ${JOB1} -> ${JOB2} -> ${JOB3} -> ${JOB4} -> ${JOB5}"
echo "========================================"
echo "Data:     probe_eval/context_ablation/data/"
echo "Results:  probe_eval/context_ablation/data/analysis/"
echo "          probe_eval/context_ablation/data/probe_results/"
echo "Monitor:  tail -f ${WORK_DIR}/*.out"
echo "========================================"
