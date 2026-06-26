#!/bin/bash
# Run Haiku 4.5 per-turn score judge on bloom-results-vi/, both full and
# reasoning-only (--no-response), both nodef and indtax modes.
set -e

ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
export $(grep -v '^#' "$ENV_FILE" | xargs)
DATA_GEN_DIR="/workspace-vast/kaiwen/persona_vectors/data_generation"
PYTHON="${DATA_GEN_DIR}/bloom/.venv/bin/python"
RESULTS_DIR="${DATA_GEN_DIR}/bloom/bloom-results-vi"

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

run () {
  local mode_flag="$1"   # --no-definition or --use-indicator-taxonomy
  local view_flag="$2"   # "" or --no-response
  local suffix="$3"      # nodef / nodef_RONLY / indtax / indtax_RONLY
  for fld in "${FOLDERS[@]}"; do
    local rollout="${RESULTS_DIR}/${fld}/rollout.json"
    local out="${RESULTS_DIR}/${fld}/rollout_misalignment_turns_haiku_history_${suffix}_score.json"
    if [ -f "$out" ]; then
      echo "[VI/$suffix] skip: $fld"; continue
    fi
    echo "[VI/$suffix] $fld"
    PYTHONPATH="$DATA_GEN_DIR" "$PYTHON" -m black_box_ind_judge "$rollout" \
      --misalignment-turns --per-turn --history-only --score-mode \
      $mode_flag $view_flag \
      --model "$MODEL" --thinking-budget $THINK -c $CONC -o "$out"
  done
}

run "--no-definition"           ""              "nodef"
run "--no-definition"           "--no-response" "nodef_RONLY"
run "--use-indicator-taxonomy"  ""              "indtax"
run "--use-indicator-taxonomy"  "--no-response" "indtax_RONLY"

echo "All done."
