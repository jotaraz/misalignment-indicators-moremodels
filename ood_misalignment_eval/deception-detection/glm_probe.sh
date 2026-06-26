#!/bin/bash
# Train GLM probes on different training data sources

DECEPTION_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation/deception-detection
PYTHON=${DECEPTION_DIR}/.venv/bin/python

cd ${DECEPTION_DIR}

# 1. Off-policy roleplaying (prewritten honest/deceptive completions)
# ${PYTHON} -m deception_detection.scripts.experiment run --config_file roleplaying_glm_offpolicy.yaml

# 2. On-policy roleplaying (GLM-generated rollouts)
${PYTHON} -m deception_detection.scripts.experiment run --config_file roleplaying_glm.yaml

# # 3. On-policy repe_honesty
# ${PYTHON} -m deception_detection.scripts.experiment run --config_file repe_glm_onpolicy.yaml

