#!/usr/bin/env python3
"""Split bloom results into dev (first N) and test (remaining) sets.

Handles: ideation.json, rollout.json, judgment.json, transcript_v*r*.json,
         and understanding.json. Re-indexes test set variations to start from 1.

Usage:
    python split_dev_test.py <results_dir> <dev_count> [--test-dir bloom-results-test]

Example:
    python split_dev_test.py bloom-results/self-preservation_glm_4_7_flash 20
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def recompute_summary_stats(judgments):
    """Recompute summary statistics from a list of judgments."""
    scores = [j["behavior_presence"] for j in judgments]
    stats = {
        "average_behavior_presence_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_behavior_presence_score": min(scores) if scores else 0,
        "max_behavior_presence_score": max(scores) if scores else 0,
        "elicitation_rate": round(sum(1 for s in scores if s >= 8) / len(scores), 2) if scores else 0,
        "total_judgments": len(judgments),
    }
    # Collect averages for any additional quality scores
    quality_keys = set()
    for j in judgments:
        for k, v in j.items():
            if k.startswith("average_") or (isinstance(v, (int, float)) and k not in (
                "behavior_presence", "variation_number", "repetition_number", "num_samples"
            )):
                quality_keys.add(k)
    # Look for score fields that aren't variation/rep/behavior_presence
    score_fields = set()
    for j in judgments:
        for k, v in j.items():
            if isinstance(v, (int, float)) and k not in (
                "behavior_presence", "variation_number", "repetition_number", "num_samples"
            ):
                score_fields.add(k)
    for field in sorted(score_fields):
        vals = [j[field] for j in judgments if field in j and isinstance(j[field], (int, float))]
        if vals:
            stats[f"average_{field}"] = round(sum(vals) / len(vals), 2)
    return stats


def split_results(results_dir, dev_count, test_dir_base):
    results_dir = Path(results_dir)
    behavior_dir_name = results_dir.name

    # Derive test directory name: prepend test_
    test_behavior_name = f"test_{behavior_dir_name}"
    test_dir = Path(test_dir_base) / test_behavior_name

    print(f"Source: {results_dir}")
    print(f"Dev (in-place): first {dev_count} variations")
    print(f"Test: {test_dir}")

    # Load all JSON files
    ideation = json.loads((results_dir / "ideation.json").read_text())
    rollout = json.loads((results_dir / "rollout.json").read_text())
    judgment = json.loads((results_dir / "judgment.json").read_text())

    total = len(ideation["variations"])
    test_count = total - dev_count
    print(f"Total variations: {total}, dev: {dev_count}, test: {test_count}")

    if dev_count >= total:
        print("ERROR: dev_count >= total variations, nothing to split")
        return

    # --- Build test set (re-indexed from 1) ---
    os.makedirs(test_dir, exist_ok=True)

    # Copy understanding.json as-is
    understanding_path = results_dir / "understanding.json"
    if understanding_path.exists():
        shutil.copy2(understanding_path, test_dir / "understanding.json")

    # Ideation: variations are positional (no variation_number field)
    test_ideation = dict(ideation)
    test_ideation["variations"] = ideation["variations"][dev_count:]
    test_ideation["total_evals"] = test_count
    test_ideation["num_scenarios"] = test_count  # approximate
    (test_dir / "ideation.json").write_text(json.dumps(test_ideation, indent=2))

    # Rollout: has variation_number field, re-index
    test_rollouts = []
    for r in rollout["rollouts"]:
        if r["variation_number"] > dev_count:
            new_r = dict(r)
            new_r["variation_number"] = r["variation_number"] - dev_count
            test_rollouts.append(new_r)
    test_rollout = dict(rollout)
    test_rollout["rollouts"] = test_rollouts
    test_rollout["total_count"] = len(test_rollouts)
    test_rollout["successful_count"] = len(test_rollouts)
    test_rollout["variations_count"] = test_count
    (test_dir / "rollout.json").write_text(json.dumps(test_rollout, indent=2))

    # Judgment: has variation_number field, re-index + recompute stats
    test_judgments = []
    for j in judgment["judgments"]:
        if j["variation_number"] > dev_count:
            new_j = dict(j)
            new_j["variation_number"] = j["variation_number"] - dev_count
            test_judgments.append(new_j)
    test_judgment = dict(judgment)
    test_judgment["judgments"] = test_judgments
    test_judgment["total_conversations"] = len(test_judgments)
    test_judgment["successful_count"] = len(test_judgments)
    test_judgment["summary_statistics"] = recompute_summary_stats(test_judgments)
    # Keep metajudgment fields as-is
    (test_dir / "judgment.json").write_text(json.dumps(test_judgment, indent=2))

    # Transcript files: rename v{old}r{rep} -> v{new}r{rep}
    for old_var in range(dev_count + 1, total + 1):
        new_var = old_var - dev_count
        # Handle multiple reps
        for rep in range(1, 10):  # assume max 9 reps
            old_name = f"transcript_v{old_var}r{rep}.json"
            new_name = f"transcript_v{new_var}r{rep}.json"
            old_path = results_dir / old_name
            if old_path.exists():
                shutil.copy2(old_path, test_dir / new_name)

    print(f"Test set written: {len(test_judgments)} judgments, {len(test_rollouts)} rollouts")

    # --- Trim dev set in-place ---
    # Ideation
    ideation["variations"] = ideation["variations"][:dev_count]
    ideation["total_evals"] = dev_count
    ideation["num_scenarios"] = dev_count
    (results_dir / "ideation.json").write_text(json.dumps(ideation, indent=2))

    # Rollout
    dev_rollouts = [r for r in rollout["rollouts"] if r["variation_number"] <= dev_count]
    rollout["rollouts"] = dev_rollouts
    rollout["total_count"] = len(dev_rollouts)
    rollout["successful_count"] = len(dev_rollouts)
    rollout["variations_count"] = dev_count
    (results_dir / "rollout.json").write_text(json.dumps(rollout, indent=2))

    # Judgment
    dev_judgments = [j for j in judgment["judgments"] if j["variation_number"] <= dev_count]
    judgment["judgments"] = dev_judgments
    judgment["total_conversations"] = len(dev_judgments)
    judgment["successful_count"] = len(dev_judgments)
    judgment["summary_statistics"] = recompute_summary_stats(dev_judgments)
    (results_dir / "judgment.json").write_text(json.dumps(judgment, indent=2))

    # Remove test transcript files from dev dir
    for old_var in range(dev_count + 1, total + 1):
        for rep in range(1, 10):
            old_path = results_dir / f"transcript_v{old_var}r{rep}.json"
            if old_path.exists():
                old_path.unlink()

    print(f"Dev set trimmed: {len(dev_judgments)} judgments, {len(dev_rollouts)} rollouts")

    # Print summary stats
    print(f"\nDev summary: {judgment['summary_statistics']}")
    print(f"Test summary: {test_judgment['summary_statistics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split bloom results into dev and test sets")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("dev_count", type=int, help="Number of variations for dev set")
    parser.add_argument("--test-dir", default="bloom-results-test", help="Base directory for test results")
    args = parser.parse_args()
    split_results(args.results_dir, args.dev_count, args.test_dir)
