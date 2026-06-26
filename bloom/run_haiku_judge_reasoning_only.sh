#!/bin/bash
# Run Haiku 4.5 per-turn score judge with --no-response (reasoning-only) on
# both EN (bloom-results) and CN (bloom-results-cn) for the 10 dev folders.
# Outputs *_haiku_history_{nodef,indtax}_RONLY_score.json next to each rollout.
set -e

ENV_FILE="/workspace-vast/kaiwen/persona_vectors/data_generation/.env"
export $(grep -v '^#' "$ENV_FILE" | xargs)
DATA_GEN_DIR="/workspace-vast/kaiwen/persona_vectors/data_generation"
PYTHON="${DATA_GEN_DIR}/bloom/.venv/bin/python"

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

run_one () {
  local results_dir="$1"          # bloom-results or bloom-results-cn
  local mode_flag="$2"            # --no-definition or --use-indicator-taxonomy
  local suffix="$3"               # nodef or indtax
  for fld in "${FOLDERS[@]}"; do
    local rollout="${DATA_GEN_DIR}/bloom/${results_dir}/${fld}/rollout.json"
    local out="${DATA_GEN_DIR}/bloom/${results_dir}/${fld}/rollout_misalignment_turns_haiku_history_${suffix}_RONLY_score.json"
    if [ -f "$out" ]; then
      echo "[$results_dir/$suffix] skip: $fld"
      continue
    fi
    echo "============================================================"
    echo "[$results_dir/$suffix/RONLY] $fld"
    echo "============================================================"
    PYTHONPATH="$DATA_GEN_DIR" "$PYTHON" -m black_box_ind_judge "$rollout" \
      --misalignment-turns \
      --per-turn \
      --history-only \
      --score-mode \
      --no-response \
      $mode_flag \
      --model "$MODEL" \
      --thinking-budget $THINK \
      -c $CONC \
      -o "$out"
  done
}

for results in bloom-results bloom-results-cn; do
  run_one "$results" "--no-definition" "nodef"
  run_one "$results" "--use-indicator-taxonomy" "indtax"
done
echo "All done."
