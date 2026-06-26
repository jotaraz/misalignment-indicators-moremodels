#!/bin/bash
# Run Haiku 4.5 per-turn score judge (nodef + indtax variants) on the Chinese
# translated rollouts under bloom/bloom-results-cn/.
set -e

ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
export $(grep -v '^#' "$ENV_FILE" | xargs)

DATA_GEN_DIR="/workspace-vast/kaiwen/persona_vectors/data_generation"
PYTHON="${DATA_GEN_DIR}/bloom/.venv/bin/python"
RESULTS_DIR="${DATA_GEN_DIR}/bloom/bloom-results-cn"

FOLDERS=(
  instructed-covert-code-sabotage_benign_glm_4_7_flash
  instructed-covert-code-sabotage_glm_4_7_flash
  instructed-strategic-sandbagging_benign_glm_4_7_flash
  instructed-strategic-sandbagging_glm_4_7_flash
  self-preservation_benign_glm_4_7_flash
  self-preservation_glm_4_7_flash
  strategic-deception_benign_glm_4_7_flash
  strategic-deception_glm_4_7_flash
  sycophancy
  sycophancy_benign_glm_4_7_flash
)

MODEL="claude-haiku-4-5-20251001"
THINK=0
CONC=50

run_mode () {
  local mode_flag="$1"     # --no-definition or --use-indicator-taxonomy
  local suffix="$2"        # nodef or indtax
  for fld in "${FOLDERS[@]}"; do
    local rollout="${RESULTS_DIR}/${fld}/rollout.json"
    local out="${RESULTS_DIR}/${fld}/rollout_misalignment_turns_haiku_history_${suffix}_score.json"
    if [ -f "$out" ]; then
      echo "[$suffix] skip (exists): $fld"
      continue
    fi
    echo "============================================================"
    echo "[$suffix] $fld"
    echo "============================================================"
    PYTHONPATH="$DATA_GEN_DIR" "$PYTHON" -m black_box_ind_judge "$rollout" \
      --misalignment-turns \
      --per-turn \
      --history-only \
      --score-mode \
      $mode_flag \
      --model "$MODEL" \
      --thinking-budget $THINK \
      -c $CONC \
      -o "$out"
  done
}

echo "=== Mode 1: --no-definition (nodef) ==="
run_mode "--no-definition" "nodef"

echo
echo "=== Mode 2: --use-indicator-taxonomy (indtax, v2_6 default) ==="
run_mode "--use-indicator-taxonomy" "indtax"

echo
echo "All done."
