#!/bin/bash
# Cross-lingual Haiku 4.5 per-turn score judge on the 3 OOD benchmarks with CoT
# (deceptionbench, mask, sycophancy_eval), staged under bloom/ood-xlingual/.
#
# Mirrors run_haiku_judge_test.sh:
#   - CN (<name>_cn) and VI (<name>_vi): all four variants {nodef,indtax}x{full,RONLY},
#     FORCE overwrite (translation copies the EN judge scores into these dirs).
#   - EN (<name>, the symlinked OOD dir): only the two RONLY views (full nodef/indtax
#     scores already exist there); generate if missing.
set -euo pipefail

REPO="/workspace-vast/kaiwen/misalignment-indicators"
# shellcheck disable=SC1091
source "${REPO}/.env"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set after sourcing .env}"

PYTHON="/workspace-vast/kaiwen/envs/hfdl/bin/python"
BASE="${REPO}/bloom/ood-xlingual"
MODEL="claude-haiku-4-5-20251001"
THINK=0
CONC=50

NAMES=(deceptionbench mask sycophancy_eval)

# run <dir> <mode_flag> <view_flag> <suffix> <force>
run () {
  local dir="$1" mode_flag="$2" view_flag="$3" suffix="$4" force="$5"
  local rollout="${dir}/rollout.json"
  local out="${dir}/rollout_misalignment_turns_haiku_history_${suffix}_score.json"
  if [ ! -f "$rollout" ]; then echo "[${dir##*/}/${suffix}] MISSING rollout"; return; fi
  if [ "$force" != "1" ] && [ -f "$out" ]; then echo "[${dir##*/}/${suffix}] skip (exists)"; return; fi
  echo "[${dir##*/}/${suffix}]"
  PYTHONPATH="$REPO" "$PYTHON" -m black_box_ind_judge "$rollout" \
    --misalignment-turns --per-turn --history-only --score-mode \
    $mode_flag $view_flag \
    --model "$MODEL" --thinking-budget $THINK -c $CONC -o "$out"
}

for name in "${NAMES[@]}"; do
  # CN + VI: all four, force overwrite
  for lang in cn vi; do
    d="${BASE}/${name}_${lang}"
    run "$d" "--no-definition"          ""              "nodef"        1
    run "$d" "--no-definition"          "--no-response" "nodef_RONLY"  1
    run "$d" "--use-indicator-taxonomy" ""              "indtax"       1
    run "$d" "--use-indicator-taxonomy" "--no-response" "indtax_RONLY" 1
  done
  # EN: reasoning-only only (full already exists)
  d="${BASE}/${name}"
  run "$d" "--no-definition"          "--no-response" "nodef_RONLY"  0
  run "$d" "--use-indicator-taxonomy" "--no-response" "indtax_RONLY" 0
done

echo "All done."
