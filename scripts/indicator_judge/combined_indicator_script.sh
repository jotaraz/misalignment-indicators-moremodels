#!/bin/bash
set -e  # Stop on first error

# bash scripts/run_aggregated_indicator_judge.sh bloom/bloom-results/sandbagging/rollout.json bloom/indicator_results/sandbagging_multiturn/

bash scripts/run_aggre_per_turn_indicator_judge_.sh bloom/bloom-results/sandbagging_benign/rollout.json bloom/indicator_results/sandbagging_benign_finegrain/

bash scripts/run_aggre_per_turn_indicator_judge_.sh bloom/bloom-results/sycophancy_benign/rollout.json bloom/indicator_results/sycophancy_benign_finegrain/

bash scripts/run_aggre_per_turn_indicator_judge_.sh bloom/bloom-results/undermining_oversight_benign/rollout.json bloom/indicator_results/undermining_oversight_benign_finegrain/

# bash scripts/run_aggre_per_turn_indicator_judge.sh bloom/bloom-results/sandbagging/rollout.json bloom/indicator_results/sandbagging/

# bash scripts/run_aggre_per_turn_indicator_judge.sh sycophancy-eval/rollouts/glm-4.7-flash/feedback_rollout.json sycophancy-eval/indicator_results/feedback/

# bash scripts/run_aggregated_indicator_judge.sh bloom/bloom-results/sycophancy/rollout.json bloom/indicator_results/sycophancy-multiturn/

# bash scripts/run_aggregated_indicator_judge.sh bloom/bloom-results/undermining_oversight/rollout.json bloom/indicator_results/undermining_oversight-multiturn/

# bash scripts/run_aggregated_indicator_judge.sh bloom/bloom-results/sandbagging/rollout.json bloom/indicator_results/sandbagging-multiturn/
