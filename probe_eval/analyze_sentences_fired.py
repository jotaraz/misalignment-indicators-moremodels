"""
Analyze how many sentences fire per turn in TP vs FP.

For each (probe, behavior, rollout, turn), count how many sentences score above
the probe's tuned threshold.  Group turns by TP/FP/FN/TN based on GT and whether
the turn fired (any sentence >= threshold).

Usage:
    python -m probe_eval.analyze_sentences_fired \
        --results-subdir v4_v2_6_combined_v2_span \
        --tuned-thresholds probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json \
        --tuned-threshold-version fpr_0.02 \
        --layer 27
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from probe_eval.common import (
    RESULTS_DIR,
    get_rollout_var_rep,
)
from probe_eval.misalignment_ground_truth import load_misalignment_ground_truth
from probe_eval.sentence_scores import load_per_sentence_scores


def load_thresholds(path: Path, version: str, layer: str) -> dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    return (
        data.get("versions", {})
        .get(version, {})
        .get("per_layer", {})
        .get(layer, {})
        .get("thresholds", {})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-subdir", required=True)
    parser.add_argument("--tuned-thresholds", required=True)
    parser.add_argument("--tuned-threshold-version", default="fpr_0.02")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--include-behaviors", default=None,
                        help="Comma-separated behavior patterns to include (wildcards)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_root = RESULTS_DIR / args.results_subdir
    layer_key = f"layer{args.layer}"

    thresholds = load_thresholds(Path(args.tuned_thresholds),
                                  args.tuned_threshold_version, layer_key)
    print(f"Loaded {len(thresholds)} thresholds")

    # Find all behaviors
    from fnmatch import fnmatch
    behaviors = set()
    for concept_dir in results_root.iterdir():
        if not concept_dir.is_dir():
            continue
        for beh_dir in (concept_dir / "span" / layer_key).glob("*"):
            if beh_dir.is_dir():
                behaviors.add(beh_dir.name)

    # Apply include filter
    if args.include_behaviors:
        patterns = [p.strip() for p in args.include_behaviors.split(",")]
        behaviors = {b for b in behaviors if any(fnmatch(b, p) for p in patterns)}

    behaviors = sorted(behaviors)
    print(f"Behaviors: {len(behaviors)}")

    # Aggregate across all probes: per turn, across all probes, sentence-fires
    # Classification: a turn is TP if GT-pos and ANY probe fires, FP if NOT GT-pos
    # and ANY probe fires, etc.  For "sentences fired", count total sentences
    # firing across all probes in that turn.

    # We'll compute two views:
    # (A) Per-probe per-turn: count sentences firing (just that probe)
    # (B) Combined (OR across probes): count total sentences firing (any probe)

    per_probe_sent_counts = defaultdict(lambda: {"TP": [], "FP": [], "FN": [], "TN": []})
    combined_sent_counts = {"TP": [], "FP": [], "FN": [], "TN": []}

    for behavior in behaviors:
        # Load GT + rollout info from first available probe
        rollout_dir = None
        for concept in thresholds:
            rpath = results_root / concept / "span" / layer_key / behavior / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    rollout_dir = json.load(f).get("rollout_dir")
                if rollout_dir:
                    break
        if not rollout_dir:
            continue

        var_rep_map = get_rollout_var_rep(rollout_dir)
        gt_json = Path(rollout_dir) / "rollout_misalignment_turns.json"
        if not gt_json.exists():
            print(f"  Skip {behavior}: no GT file")
            continue
        gt_map = load_misalignment_ground_truth(gt_json, exclude_ambiguous=True)

        # Build per-turn aggregated data across probes
        # Key: (var, rep, turn) → {gt_pos, combined_sent_fires, per_probe: {probe: sent_fires}}
        turn_data: dict[tuple, dict] = {}

        for concept, thresh in thresholds.items():
            ts_path = results_root / concept / "span" / layer_key / behavior / "token_scores.json"
            if not ts_path.exists():
                continue
            with open(ts_path) as f:
                ts_data = json.load(f)
            for rollout in ts_data.get("per_rollout", []):
                idx = rollout["rollout_index"]
                if idx not in var_rep_map:
                    continue
                var, rep = var_rep_map[idx]
                # Load results.json for this probe/behavior to get per_turn_scores + sentence_scores
                rpath = results_root / concept / "span" / layer_key / behavior / "results.json"
                if not rpath.exists():
                    continue

        # Load per-sentence scores from token_scores.json for each probe
        rollouts_cache: dict[str, list[dict]] = {}
        for concept, thresh in thresholds.items():
            ts_path = results_root / concept / "span" / layer_key / behavior / "token_scores.json"
            per_sent = load_per_sentence_scores(
                ts_path, rollout_dir, rollouts_cache,
                short_sentence_mode="merge", min_words=5,
            )
            if per_sent is None:
                continue
            for idx, turn_to_sents in per_sent.items():
                if idx not in var_rep_map:
                    continue
                var, rep = var_rep_map[idx]
                misaligned_turns = gt_map.get((var, rep), set())
                for turn, sent_scores in turn_to_sents.items():
                    key = (behavior, var, rep, turn)
                    if key not in turn_data:
                        turn_data[key] = {
                            "gt_pos": turn in misaligned_turns,
                            "per_probe": {},
                        }
                    n_fires = sum(1 for s in sent_scores if s > thresh)
                    turn_data[key]["per_probe"][concept] = n_fires

        # Classify turns + count sentences
        for key, info in turn_data.items():
            gt = info["gt_pos"]
            per_probe = info["per_probe"]
            combined_fires = sum(per_probe.values())
            any_fires = combined_fires > 0

            # Combined classification
            if gt and any_fires:
                cls = "TP"
            elif (not gt) and any_fires:
                cls = "FP"
            elif gt and (not any_fires):
                cls = "FN"
            else:
                cls = "TN"
            combined_sent_counts[cls].append(combined_fires)

            # Per-probe classification
            for concept, n_fires in per_probe.items():
                probe_fires = n_fires > 0
                if gt and probe_fires:
                    cls = "TP"
                elif (not gt) and probe_fires:
                    cls = "FP"
                elif gt and (not probe_fires):
                    cls = "FN"
                else:
                    cls = "TN"
                per_probe_sent_counts[concept][cls].append(n_fires)

    # Print results
    print(f"\n{'='*70}")
    print(f"Combined (OR across probes) — total sentences fired per turn")
    print(f"{'='*70}")
    print(f"{'Category':<6} {'N':>6} {'Mean':>7} {'Median':>7} {'Max':>5} {'N sentences>0':>14}")
    for cls in ["TP", "FP", "FN", "TN"]:
        arr = np.array(combined_sent_counts[cls])
        n = len(arr)
        if n == 0:
            print(f"{cls:<6} {0:>6}")
            continue
        nz = int((arr > 0).sum())
        print(f"{cls:<6} {n:>6} {arr.mean():>7.2f} {int(np.median(arr)):>7} {int(arr.max()):>5} {nz:>14}")

    print(f"\n{'='*70}")
    print(f"Per-probe — sentences fired per turn (only where that probe matters)")
    print(f"{'='*70}")
    print(f"{'Probe':<45} {'TP n':>5} {'TP avg':>7} {'FP n':>5} {'FP avg':>7} {'diff':>6}")
    print("-" * 85)
    rows = []
    for concept in sorted(per_probe_sent_counts):
        d = per_probe_sent_counts[concept]
        tp = np.array(d["TP"]) if d["TP"] else np.array([0])
        fp = np.array(d["FP"]) if d["FP"] else np.array([0])
        tp_arr = [x for x in d["TP"] if x > 0]
        fp_arr = [x for x in d["FP"] if x > 0]
        tp_mean = float(np.mean(tp_arr)) if tp_arr else 0.0
        fp_mean = float(np.mean(fp_arr)) if fp_arr else 0.0
        diff = tp_mean - fp_mean
        rows.append((concept, len(tp_arr), tp_mean, len(fp_arr), fp_mean, diff))

    for concept, tp_n, tp_avg, fp_n, fp_avg, diff in sorted(rows, key=lambda r: -r[5]):
        print(f"{concept:<45} {tp_n:>5} {tp_avg:>7.2f} {fp_n:>5} {fp_avg:>7.2f} {diff:>+6.2f}")

    # Save JSON
    if args.output:
        out = {
            "combined": {
                cls: {
                    "counts": combined_sent_counts[cls],
                    "mean": float(np.mean(combined_sent_counts[cls])) if combined_sent_counts[cls] else 0.0,
                    "n": len(combined_sent_counts[cls]),
                }
                for cls in ["TP", "FP", "FN", "TN"]
            },
            "per_probe": {
                concept: {
                    cls: {
                        "mean_nonzero": float(np.mean([x for x in per_probe_sent_counts[concept][cls] if x > 0]))
                        if any(x > 0 for x in per_probe_sent_counts[concept][cls]) else 0.0,
                        "n_fired": sum(1 for x in per_probe_sent_counts[concept][cls] if x > 0),
                        "n_total": len(per_probe_sent_counts[concept][cls]),
                    }
                    for cls in ["TP", "FP", "FN", "TN"]
                }
                for concept in per_probe_sent_counts
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
