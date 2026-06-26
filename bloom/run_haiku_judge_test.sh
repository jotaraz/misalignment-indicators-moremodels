#!/bin/bash
# Cross-lingual Haiku 4.5 per-turn score judge on the TEST set.
#
# Produces the four judge-output files per folder that compute_auroc_three_lang.py
# consumes: {nodef,indtax} x {full, RONLY}.
#   - CN (bloom-results-test-cn) and VI (bloom-results-test-vi): generate all four,
#     FORCING overwrite — the translation step copied the EN nodef/indtax full-score
#     files into these dirs, so a skip-if-exists guard would wrongly keep EN scores.
#   - EN (bloom-results-test): only the two RONLY views are generated here; the full
#     views already exist (rollout_misalignment_turns_haiku_history_{nodef,indtax}_score.json).
#
# Uses the repo's current paths + the hfdl venv (which has `anthropic`).
set -euo pipefail

REPO="/workspace-vast/kaiwen/misalignment-indicators"
# shellcheck disable=SC1091
source "${REPO}/.env"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set after sourcing .env}"

PYTHON="/workspace-vast/kaiwen/envs/hfdl/bin/python"
MODEL="claude-haiku-4-5-20251001"
THINK=0
CONC=50

# The 5 cross-lingual behaviors x {benign, misaligned} = 10 folders, matching the
# val/dev set (bloom-results-cn) and the appendix. The other 8 test folders
# (long-horizon-sabotage-{backdoor,bug}, self-preferential-bias, self-promotion)
# are NOT part of this ablation.
FOLDERS=(
  test_instructed-covert-code-sabotage_benign_glm_4_7_flash
  test_instructed-covert-code-sabotage_glm_4_7_flash
  test_instructed-strategic-sandbagging_benign_glm_4_7_flash
  test_instructed-strategic-sandbagging_glm_4_7_flash
  test_self-preservation_benign_glm_4_7_flash
  test_self-preservation_glm_4_7_flash
  test_strategic-deception_benign_glm_4_7_flash
  test_strategic-deception_glm_4_7_flash
  test_sycophancy_benign_glm_4_7_flash
  test_sycophancy_glm_4_7_flash
)

# run <results_dir> <mode_flag> <view_flag> <suffix> <force>
run () {
  local results_dir="$1" mode_flag="$2" view_flag="$3" suffix="$4" force="$5"
  for fld in "${FOLDERS[@]}"; do
    local rollout="${REPO}/bloom/${results_dir}/${fld}/rollout.json"
    local out="${REPO}/bloom/${results_dir}/${fld}/rollout_misalignment_turns_haiku_history_${suffix}_score.json"
    if [ ! -f "$rollout" ]; then echo "[${results_dir}/${suffix}] MISSING rollout: $fld"; continue; fi
    if [ "$force" != "1" ] && [ -f "$out" ]; then echo "[${results_dir}/${suffix}] skip (exists): $fld"; continue; fi
    echo "[${results_dir}/${suffix}] $fld"
    PYTHONPATH="$REPO" "$PYTHON" -m black_box_ind_judge "$rollout" \
      --misalignment-turns --per-turn --history-only --score-mode \
      $mode_flag $view_flag \
      --model "$MODEL" --thinking-budget $THINK -c $CONC -o "$out"
  done
}

LANG="${1:-all}"   # cn | vi | en | all

if [ "$LANG" = "cn" ] || [ "$LANG" = "all" ]; then
  echo "===== CN (force overwrite copied EN scores) ====="
  run bloom-results-test-cn "--no-definition"          ""              "nodef"        1
  run bloom-results-test-cn "--no-definition"          "--no-response" "nodef_RONLY"  1
  run bloom-results-test-cn "--use-indicator-taxonomy" ""              "indtax"       1
  run bloom-results-test-cn "--use-indicator-taxonomy" "--no-response" "indtax_RONLY" 1
fi

if [ "$LANG" = "vi" ] || [ "$LANG" = "all" ]; then
  echo "===== VI (force overwrite copied EN scores) ====="
  run bloom-results-test-vi "--no-definition"          ""              "nodef"        1
  run bloom-results-test-vi "--no-definition"          "--no-response" "nodef_RONLY"  1
  run bloom-results-test-vi "--use-indicator-taxonomy" ""              "indtax"       1
  run bloom-results-test-vi "--use-indicator-taxonomy" "--no-response" "indtax_RONLY" 1
fi

if [ "$LANG" = "en" ] || [ "$LANG" = "all" ]; then
  echo "===== EN (reasoning-only only; full views already exist) ====="
  run bloom-results-test "--no-definition"          "--no-response" "nodef_RONLY"  0
  run bloom-results-test "--use-indicator-taxonomy" "--no-response" "indtax_RONLY" 0
fi

echo "All done."
