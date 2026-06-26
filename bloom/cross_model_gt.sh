#!/bin/bash
# Ground-truth + audit judges (Opus-4.6) for one target's cross-model rollout
# dirs. README section 8, step 4. Run from repo/bloom with the bloom env and
# ANTHROPIC_API_KEY set. PYBIN selects the interpreter (default: python).
#   PYBIN=/fast/jtaraz/envs/bloom/bin/python bash cross_model_gt.sh <target_slug>
#   e.g. <target_slug> = llama_3_3_70b  (sanitized model name)
set -euo pipefail

SLUG="$1"
PY="${PYBIN:-python}"
JUDGE_MODEL="claude-opus-4-6"
THINK=20000

# behavior -> indicator-judge soft-prior label (matches cross_model_rollout.py)
declare -A TB=(
  [self-preservation]=self-preservation
  [strategic-deception]=strategic-deception
  [sycophancy]=sycophancy
  [instructed-covert-code-sabotage]=sabotage
  [instructed-strategic-sandbagging]=sandbagging
)

for beh in self-preservation strategic-deception sycophancy \
           instructed-covert-code-sabotage instructed-strategic-sandbagging; do
  md="bloom-results/${beh}_${SLUG}"
  bd="bloom-results/${beh}_benign_${SLUG}"
  label="${TB[$beh]}"

  if [ -f "$md/rollout.json" ]; then
    echo ">> GT misaligned: $md"
    # open-ended misalignment judge, then indicator judge (merges into same file)
    "$PY" -m black_box_ind_judge "$md/rollout.json" --misalignment-turns \
        --model "$JUDGE_MODEL" --thinking-budget "$THINK" \
        -o "$md/rollout_misalignment_turns.json"
    "$PY" -m black_box_ind_judge "$md/rollout.json" --indicator-turns \
        --test-behavior "$label" --thinking-budget "$THINK" \
        -o "$md/rollout_misalignment_turns.json"
    "$PY" -m black_box_ind_judge.audit --source "$md" || true
  else
    echo "skip (no rollout): $md"
  fi

  if [ -f "$bd/rollout.json" ]; then
    echo ">> GT benign: $bd"
    "$PY" -m black_box_ind_judge "$bd/rollout.json" --indicator-turns --benign \
        --test-behavior "$label" --thinking-budget "$THINK" \
        -o "$bd/rollout_misalignment_turns.json"
    "$PY" -m black_box_ind_judge.audit --source "$bd" --audit-benign || true
  else
    echo "skip (no rollout): $bd"
  fi
done
echo "Ground truth complete for slug=$SLUG"
