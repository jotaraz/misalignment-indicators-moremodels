"""
Evaluate cross-model agreement by treating opus aggregated results as ground truth
and measuring how well haiku/glm_flash match.

Metrics:
- Binary accuracy (variation-level): does model agree with opus on presence/absence?
- Per-turn binary accuracy: per (variation, turn) agreement with opus
- Ind-Name Jaccard: indicator name set overlap with opus per variation
- Ind-Name Jaccard (per-turn, all): per (variation, turn) name set overlap with opus

Usage:
    python bloom/evaluate_cross_model.py bloom/indicator_results/v2.2
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_aggregated(path: Path) -> dict:
    """Load aggregated JSON and return {variation_number: [indicator_dicts]}."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for rollout in data.get("evaluated_rollouts", data.get("aggregated_rollouts", [])):
        vn = rollout["variation_number"]
        indicators = rollout.get("filtered_indicators", rollout.get("detected_indicators", []))
        result[vn] = indicators
    return result


def get_binary(indicators: list) -> int:
    return 1 if len(indicators) > 0 else 0


def get_indicator_names(indicators: list) -> set:
    return {ind["indicator_name"] for ind in indicators}


def get_indicator_names_at_turn(indicators: list, turn_number: int) -> set:
    return {ind["indicator_name"] for ind in indicators if ind.get("turn_number") == turn_number}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_model_to_opus(opus_data: dict, model_data: dict) -> dict:
    """Compare a model's aggregated results against opus as ground truth."""
    all_variations = sorted(set(opus_data.keys()) | set(model_data.keys()))

    # Binary accuracy (variation-level)
    binary_correct = 0
    for v in all_variations:
        opus_bin = get_binary(opus_data.get(v, []))
        model_bin = get_binary(model_data.get(v, []))
        if opus_bin == model_bin:
            binary_correct += 1
    binary_acc = binary_correct / len(all_variations) if all_variations else 1.0

    # Ind-Name Jaccard (variation-level)
    name_jaccards = []
    for v in all_variations:
        opus_names = get_indicator_names(opus_data.get(v, []))
        model_names = get_indicator_names(model_data.get(v, []))
        name_jaccards.append(jaccard(opus_names, model_names))
    mean_name_jaccard = float(np.mean(name_jaccards)) if name_jaccards else 1.0

    # Find max turn
    max_turn = 0
    for data in [opus_data, model_data]:
        for v in all_variations:
            for ind in data.get(v, []):
                max_turn = max(max_turn, ind.get("turn_number", 0))

    # Per-turn metrics
    n_all_pairs = 0
    n_nontrivial_pairs = 0
    per_turn_binary_correct_all = 0
    per_turn_binary_correct_nt = 0
    per_turn_jaccards_all = []
    per_turn_jaccards_nt = []

    for v in all_variations:
        for turn_num in range(1, max_turn + 1):
            n_all_pairs += 1
            opus_at_turn = [ind for ind in opus_data.get(v, []) if ind.get("turn_number") == turn_num]
            model_at_turn = [ind for ind in model_data.get(v, []) if ind.get("turn_number") == turn_num]
            opus_bin = 1 if opus_at_turn else 0
            model_bin = 1 if model_at_turn else 0
            correct = opus_bin == model_bin
            if correct:
                per_turn_binary_correct_all += 1
            opus_names = get_indicator_names_at_turn(opus_data.get(v, []), turn_num)
            model_names = get_indicator_names_at_turn(model_data.get(v, []), turn_num)
            jac = jaccard(opus_names, model_names)
            per_turn_jaccards_all.append(jac)
            # Non-trivial: either opus or model has indicators at this turn
            if opus_bin or model_bin:
                n_nontrivial_pairs += 1
                if correct:
                    per_turn_binary_correct_nt += 1
                per_turn_jaccards_nt.append(jac)

    per_turn_binary_acc = per_turn_binary_correct_all / n_all_pairs if n_all_pairs else 1.0
    per_turn_binary_acc_nt = per_turn_binary_correct_nt / n_nontrivial_pairs if n_nontrivial_pairs else 1.0
    mean_per_turn_jaccard = float(np.mean(per_turn_jaccards_all)) if per_turn_jaccards_all else 1.0
    mean_per_turn_jaccard_nt = float(np.mean(per_turn_jaccards_nt)) if per_turn_jaccards_nt else 1.0

    return {
        "n_variations": len(all_variations),
        "binary_accuracy": round(binary_acc, 4),
        "per_turn_binary_accuracy": round(per_turn_binary_acc, 4),
        "per_turn_binary_accuracy_nontrivial": round(per_turn_binary_acc_nt, 4),
        "ind_name_jaccard": round(mean_name_jaccard, 4),
        "ind_name_jaccard_per_turn_all": round(mean_per_turn_jaccard, 4),
        "ind_name_jaccard_per_turn_nontrivial": round(mean_per_turn_jaccard_nt, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-model evaluation treating opus as ground truth")
    parser.add_argument("base_dir", help="Base dir (e.g. bloom/indicator_results/v2.2)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    R = lambda x: round(float(x), 4)

    all_results = {}

    for behavior_dir in sorted(base.iterdir()):
        if not behavior_dir.is_dir():
            continue
        behavior = behavior_dir.name
        all_results[behavior] = {}

        # Group configs by indicator set
        ind_sets = {}
        for config_dir in sorted(behavior_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            name = config_dir.name  # e.g. "v2.2-finegrain_opus"
            # Split into indicator set and model
            # Known model suffixes (check longest first)
            model = None
            for suffix in ["_glm_flash", "_haiku", "_opus"]:
                if name.endswith(suffix):
                    ind_set = name[:-len(suffix)]
                    model = suffix[1:]  # strip leading _
                    break
            if model is None:
                continue
            if ind_set not in ind_sets:
                ind_sets[ind_set] = {}
            agg_path = config_dir / "rollout_aggregated.json"
            if agg_path.exists():
                ind_sets[ind_set][model] = agg_path

        # For each indicator set, compare haiku/glm_flash against opus
        for ind_set, model_paths in sorted(ind_sets.items()):
            if "opus" not in model_paths:
                continue
            opus_data = load_aggregated(model_paths["opus"])
            for model_name in ["haiku", "glm_flash"]:
                if model_name not in model_paths:
                    continue
                model_data = load_aggregated(model_paths[model_name])
                result = compare_model_to_opus(opus_data, model_data)
                key = f"{ind_set}_{model_name}"
                all_results[behavior][key] = result

    # Print results
    metrics = [
        ("Binary accuracy", "binary_accuracy"),
        ("Per-turn binary accuracy", "per_turn_binary_accuracy"),
        ("Per-turn binary accuracy (non-trivial)", "per_turn_binary_accuracy_nontrivial"),
        ("Ind-Name Jaccard", "ind_name_jaccard"),
        ("Ind-Name Jaccard (per-turn, all)", "ind_name_jaccard_per_turn_all"),
        ("Ind-Name Jaccard (per-turn, non-trivial)", "ind_name_jaccard_per_turn_nontrivial"),
    ]

    for behavior, configs in sorted(all_results.items()):
        if not configs:
            continue
        print(f"\n{'='*70}")
        print(f"  {behavior}")
        print(f"{'='*70}")

        config_names = sorted(configs.keys())
        # Header
        header = f"{'Metric':<35}"
        for cn in config_names:
            header += f" {cn:>20}"
        print(header)
        print("-" * len(header))

        for label, key in metrics:
            row = f"{label:<35}"
            for cn in config_names:
                val = configs[cn].get(key)
                row += f" {val:>20.3f}" if val is not None else f" {'-':>20}"
            print(row)

    # Save results
    out_path = base / "cross_model_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
