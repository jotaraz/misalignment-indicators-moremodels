#!/bin/bash
# Score neutral dialogues with all probes to find false positives.
# Run directly on a GPU node (no SLURM).
#
# Usage:
#   bash scripts/probe_train_eval/run_score_neutral.sh
#   bash scripts/probe_train_eval/run_score_neutral.sh --max-dialogues 500

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/deception-detection/.venv/bin/python

source ${BASE_DIR}/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1

cd "${BASE_DIR}"

${PYTHON} -m probe.neutral.score_neutral \
    --probes-dir ${BASE_DIR}/probe/probes/v2_3_gen_prompt_v2_span_v2_clean \
    --dialogues-path ${BASE_DIR}/probe/data/neutral/dialogues_filtered_v2.json \
    --layer 27 \
    --output ${BASE_DIR}/probe/data/neutral/false_positives.json \
    "$@"