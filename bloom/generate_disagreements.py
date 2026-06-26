"""
Find per-turn disagreements between independent indicator runs and save
detailed info to disagree.json in each result directory.

Usage:
    python bloom/generate_disagreements.py bloom/indicator_results/v2/sabotage_backdoor/v2-cross-behavior_haiku
    python bloom/generate_disagreements.py bloom/indicator_results/v2  # process all subdirs recursively
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    """Load a run JSON and return {variation_number: [indicator_dicts]}."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for rollout in data["evaluated_rollouts"]:
        vn = rollout["variation_number"]
        result[vn] = rollout.get("detected_indicators", [])
    return result


def get_indicators_at_turn(indicators: list, turn_number: int) -> list:
    """Return full indicator dicts at a specific turn."""
    return [ind for ind in indicators if ind.get("turn_number") == turn_number]


def get_indicator_names_at_turn(indicators: list, turn_number: int) -> set:
    """Return set of indicator names at a specific turn."""
    return {ind["indicator_name"] for ind in indicators if ind.get("turn_number") == turn_number}


def process_dir(results_dir: Path) -> dict | None:
    """Process a single result directory and return disagreement report."""
    run_files = sorted(results_dir.glob("rollout_run*.json"))
    n_runs = len(run_files)
    if n_runs < 2:
        return None

    runs = []
    for rf in run_files:
        runs.append(load_run(rf))

    all_variations = sorted(set().union(*(r.keys() for r in runs)))

    # Find max turn
    max_turn = 0
    for run in runs:
        for v in all_variations:
            for ind in run.get(v, []):
                max_turn = max(max_turn, ind.get("turn_number", 0))

    disagreements = []
    n_total_nontrivial = 0

    for v in all_variations:
        for turn_num in range(1, max_turn + 1):
            name_sets = [get_indicator_names_at_turn(run.get(v, []), turn_num) for run in runs]

            # Skip turns where all runs have empty sets
            if all(len(s) == 0 for s in name_sets):
                continue

            n_total_nontrivial += 1

            # Check if all runs agree on the exact same indicator names
            if len(set(frozenset(s) for s in name_sets)) == 1:
                continue  # all agree

            # Disagreement found — collect details
            run_details = {}
            for i, run in enumerate(runs):
                indicators_at_turn = get_indicators_at_turn(run.get(v, []), turn_num)
                run_details[f"run{i+1}"] = {
                    "indicator_names": sorted(name_sets[i]),
                    "indicators": [
                        {
                            "indicator_name": ind["indicator_name"],
                            "sentence": ind.get("sentence", ""),
                            "confidence": ind.get("confidence"),
                            "evidence": ind.get("evidence", ""),
                        }
                        for ind in indicators_at_turn
                    ],
                }

            disagreements.append({
                "variation_number": v,
                "turn_number": turn_num,
                "runs": run_details,
            })

    return {
        "results_dir": str(results_dir),
        "n_runs": n_runs,
        "n_disagreements": len(disagreements),
        "n_total_nontrivial_turns": n_total_nontrivial,
        "disagreement_rate": round(len(disagreements) / n_total_nontrivial, 4) if n_total_nontrivial > 0 else 0.0,
        "disagreements": disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description="Find per-turn disagreements between indicator runs")
    parser.add_argument("path", help="Single result dir, or parent dir to process all subdirs")
    args = parser.parse_args()

    target = Path(args.path)

    # Determine if this is a single result dir or a parent
    if list(target.glob("rollout_run*.json")):
        dirs_to_process = [target]
    else:
        # Find all subdirs that contain rollout_run files
        dirs_to_process = sorted(
            d.parent for d in target.rglob("rollout_run1.json")
        )

    if not dirs_to_process:
        print(f"No result directories found under {target}")
        sys.exit(1)

    print(f"Found {len(dirs_to_process)} result dir(s) to process\n")

    for results_dir in dirs_to_process:
        report = process_dir(results_dir)
        if report is None:
            print(f"SKIP (< 2 run files): {results_dir}")
            continue

        out_path = results_dir / "disagree.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        n_dis = report["n_disagreements"]
        n_nt = report["n_total_nontrivial_turns"]
        rate = report["disagreement_rate"]
        print(f"{results_dir.name}: {n_dis}/{n_nt} disagreements ({rate:.1%}) -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
