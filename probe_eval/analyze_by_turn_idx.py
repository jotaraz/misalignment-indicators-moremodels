"""
Analyze misalignment probe performance broken down by turn index.

Reproduces the combined val_clean_label_f1 metrics (transcript_relaxed and
pf_transcript_full_fires) from misalignment_gt_summary_{dev,test}.json,
but split by turn index.

Uses the same data pipeline as misalignment_ground_truth.py:
  1. Load per-probe results.json files
  2. Build per-turn data points with misalignment GT
  3. Replace probe_score with max(sentence_scores) via to_sentence_max_points
  4. Apply val_clean_label_f1 thresholds (OR across probes)
  5. Compute transcript_relaxed and pf_transcript_full_fires metrics
  6. Group by turn index and report per-turn metrics
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from probe_eval.common import (
    NON_INDICATOR_CONCEPTS,
    get_concept_from_experiment_folder,
    get_n_turns,
    get_rollout_var_rep,
)
from probe_eval.misalignment_ground_truth import (
    build_per_turn_data,
    build_transcript_relaxed_points,
    load_misalignment_ground_truth,
)
from probe_eval.sentence_scores import (
    build_sentence_scores_by_key,
    load_per_sentence_scores,
    to_sentence_max_points,
)


def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _build_turn_map_or(
    per_probe_points: dict[str, list[dict]],
    thresholds: dict[str, float],
) -> dict[tuple, dict]:
    """Build {(var, rep, turn): {gt, fires}} from OR across probes."""
    concepts = sorted(per_probe_points.keys())
    turn_map: dict[tuple, dict] = {}
    for concept in concepts:
        t = thresholds.get(concept, float("inf"))
        for p in per_probe_points[concept]:
            if np.isnan(p["probe_score"]):
                continue
            key = (p["var"], p["rep"], p["turn"])
            if key not in turn_map:
                turn_map[key] = {"gt": p["gt_misaligned"], "fires": False}
            if p["gt_misaligned"]:
                turn_map[key]["gt"] = True
            if p["probe_score"] > t:
                turn_map[key]["fires"] = True
    return turn_map


def compute_metrics_from_turn_map(turn_map: dict[tuple, dict]) -> dict:
    """Standard accuracy/precision/recall/f1 from turn_map."""
    tp = fp = fn = tn = 0
    for info in turn_map.values():
        if info["gt"] and info["fires"]:
            tp += 1
        elif info["fires"]:
            fp += 1
        elif info["gt"]:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    n_pos = tp + fn
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (math.isnan(precision) or math.isnan(recall))
          else float("nan"))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "fpr": fpr,
            "f1": f1, "n": n, "n_pos": n_pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pf_transcript_full_fires_metrics(
    eval_turn_map: dict[tuple, dict],
    full_fire_map: dict[tuple, dict],
) -> dict:
    """Prior-fire relaxation using fires from full_fire_map, evaluating on eval_turn_map."""
    # Build per-transcript fire timeline from full_fire_map
    transcript_fires: dict[tuple, list[tuple[int, bool]]] = defaultdict(list)
    for key, info in full_fire_map.items():
        var, rep, turn = key
        transcript_fires[(var, rep)].append((turn, info["fires"]))

    prior_fire_by_turn: dict[tuple, bool] = {}
    for tkey, turns in transcript_fires.items():
        turns.sort(key=lambda x: x[0])
        any_fire_so_far = False
        for turn_num, fires in turns:
            prior_fire_by_turn[(*tkey, turn_num)] = any_fire_so_far
            if fires:
                any_fire_so_far = True

    tp = fp = fn = tn = 0
    for key, info in eval_turn_map.items():
        gt = info["gt"]
        fires = info["fires"]
        has_prior = prior_fire_by_turn.get(key, False)
        if gt and fires:
            tp += 1
        elif fires:
            fp += 1
        elif gt and has_prior:
            pass  # exclude
        elif gt:
            fn += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    n_pos = tp + fn
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (math.isnan(precision) or math.isnan(recall))
          else float("nan"))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "fpr": fpr,
            "f1": f1, "n": n, "n_pos": n_pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def extract_turn_idx(key: tuple) -> int:
    """Extract original turn index from the (var, rep, turn) key."""
    return key[2]


def filter_turn_map_by_turn(turn_map: dict[tuple, dict], turn_idx: int) -> dict[tuple, dict]:
    """Filter turn_map to only include entries with the given turn index."""
    return {k: v for k, v in turn_map.items() if k[2] == turn_idx}


def discover_results_for_split(
    search_root: Path,
    split: str,  # "dev" or "test"
) -> list[dict]:
    """Find probe results.json files for dev or test split.

    Uses the same behavior filtering as run_v2_4_existing_pipeline.sh:
      TEST_BEHAVIOR_PATTERNS = test_*, bloom_rollout, bloom,
                               sycophancy_answer, sycophancy_are_you_sure,
                               sycophancy_feedback
      dev: --exclude-behaviors TEST_BEHAVIOR_PATTERNS
      test: --include-behaviors TEST_BEHAVIOR_PATTERNS
    """
    from fnmatch import fnmatch

    TEST_PATTERNS = [
        "test_*", "bloom_rollout", "bloom",
        "sycophancy_answer", "sycophancy_are_you_sure", "sycophancy_feedback",
    ]

    found = []
    gt_exists_cache: dict[str, bool] = {}

    for rj in sorted(search_root.rglob("results.json")):
        behavior = rj.parent.name

        # Apply the same include/exclude logic as the pipeline script
        matches_test_pattern = any(fnmatch(behavior, p) for p in TEST_PATTERNS)
        if split == "dev" and matches_test_pattern:
            continue
        if split == "test" and not matches_test_pattern:
            continue

        with open(rj) as f:
            data = json.load(f)

        rollout_dir = data.get("rollout_dir", "")
        if not rollout_dir:
            continue

        is_all_negative = data.get("all_negative", False)
        if not is_all_negative:
            if rollout_dir not in gt_exists_cache:
                gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
                gt_exists_cache[rollout_dir] = gt_path.exists()
            if not gt_exists_cache[rollout_dir]:
                continue

        concept = get_concept_from_experiment_folder(data.get("experiment_folder", ""))
        if concept is None:
            continue
        if concept in NON_INDICATOR_CONCEPTS:
            continue

        # Get layer from path
        # Structure: .../concept/span/layerNN/rollout_name/results.json
        layer = rj.parent.parent.name  # e.g., "layer27"

        found.append({
            "behavior": behavior,
            "concept": concept,
            "layer": layer,
            "rollout_dir": rollout_dir,
            "result_path": rj,
            "data": data,
            "all_negative": is_all_negative,
        })
    return found


def analyze_split(search_root: Path, split: str, layer_name: str = "layer27"):
    """Analyze a single split (dev or test) and print per-turn metrics."""
    results = discover_results_for_split(search_root, split)
    if not results:
        print(f"No results found for split={split}")
        return

    # Filter to the target layer
    results = [r for r in results if r["layer"] == layer_name]

    # Load thresholds from the summary file
    summary_path = search_root / f"misalignment_gt_summary_{split}.json"
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    thresholds = summary["per_layer"][layer_name]["any_probe_fires"]["val_clean_label_f1"]["thresholds"]
    print(f"\n{'='*80}")
    print(f"Split: {split.upper()} | Layer: {layer_name}")
    print(f"{'='*80}")
    print(f"Using val_clean_label_f1 thresholds ({len(thresholds)} probes)")

    # Group results by behavior (rollout_dir)
    by_behavior_rollout: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        # Use rollout_dir as key to group by unique behavior+rollout combination
        by_behavior_rollout[r["rollout_dir"]].append(r)

    # Cache GT and var_rep maps
    gt_cache: dict[str, dict] = {}
    rollouts_full_cache: dict[str, list[dict]] = {}

    # Accumulate per-probe points across behaviors (with behavior-tagged var)
    per_probe_combined: dict[str, list[dict]] = defaultdict(list)

    for rollout_dir, rollout_results in sorted(by_behavior_rollout.items()):
        behavior = rollout_results[0]["behavior"]
        is_all_neg = rollout_results[0].get("all_negative", False)

        if rollout_dir not in gt_cache:
            var_rep_map = get_rollout_var_rep(rollout_dir)
            n_turns_map = get_n_turns(rollout_dir)

            if is_all_neg:
                misalignment_gt = {vr: set() for vr in var_rep_map.values()}
            else:
                gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
                if not gt_path.exists():
                    continue
                misalignment_gt = load_misalignment_ground_truth(gt_path)

            gt_cache[rollout_dir] = {
                "misalignment_gt": misalignment_gt,
                "var_rep_map": var_rep_map,
                "n_turns_map": n_turns_map,
            }

        bdata = gt_cache[rollout_dir]
        misalignment_gt = bdata["misalignment_gt"]
        var_rep_map = bdata["var_rep_map"]
        n_turns_map = bdata["n_turns_map"]

        for r in rollout_results:
            concept = r["concept"]
            points = build_per_turn_data(r["data"], misalignment_gt, n_turns_map, var_rep_map)
            if not points:
                continue

            # Apply sentence-max scoring (same as summary pipeline)
            token_scores_path = r["result_path"].parent / "token_scores.json"
            sentence_data = load_per_sentence_scores(
                token_scores_path, r["data"]["rollout_dir"], rollouts_full_cache,
            )
            sent_by_key: dict[tuple, list[float]] = {}
            if sentence_data is not None:
                sent_by_key = build_sentence_scores_by_key(sentence_data, var_rep_map)

            points = to_sentence_max_points(points, sent_by_key)

            # Tag with behavior prefix
            for p in points:
                if not np.isnan(p["probe_score"]):
                    tagged_p = {**p, "var": f"{behavior}__{p['var']}"}
                    per_probe_combined[concept].append(tagged_p)

    # Build full per_turn turn_map (for fire detection in pf_transcript_full_fires)
    full_turn_map = _build_turn_map_or(dict(per_probe_combined), thresholds)

    # Build transcript_relaxed points per probe
    per_probe_transcript: dict[str, list[dict]] = {}
    for concept, pts in per_probe_combined.items():
        per_probe_transcript[concept] = build_transcript_relaxed_points(pts)

    # Build transcript_relaxed turn_map
    transcript_turn_map = _build_turn_map_or(dict(per_probe_transcript), thresholds)

    # Verify overall metrics match the summary
    overall_transcript = compute_metrics_from_turn_map(transcript_turn_map)
    overall_pf_full = pf_transcript_full_fires_metrics(transcript_turn_map, full_turn_map)

    print(f"\n--- Overall verification ---")
    print(f"transcript_relaxed:       n={overall_transcript['n']:>4}, n_pos={overall_transcript['n_pos']:>3}, "
          f"acc={overall_transcript['accuracy']:.4f}, prec={overall_transcript['precision']:.4f}, "
          f"rec={overall_transcript['recall']:.4f}, fpr={overall_transcript['fpr']:.4f}, f1={overall_transcript['f1']:.4f}")
    print(f"pf_transcript_full_fires: n={overall_pf_full['n']:>4}, n_pos={overall_pf_full['n_pos']:>3}, "
          f"acc={overall_pf_full['accuracy']:.4f}, prec={overall_pf_full['precision']:.4f}, "
          f"rec={overall_pf_full['recall']:.4f}, fpr={overall_pf_full['fpr']:.4f}, f1={overall_pf_full['f1']:.4f}")

    # Get all unique turn indices
    all_turns = sorted(set(k[2] for k in full_turn_map.keys()))
    print(f"\nTurn indices present: {all_turns}")
    print(f"Number of distinct turns: {len(all_turns)}")

    # --- Per-turn-idx metrics ---
    print(f"\n{'='*120}")
    print(f"Per-turn-idx metrics for {split.upper()} | transcript_relaxed")
    print(f"{'='*120}")
    print(f"{'Turn':>5} | {'n':>5} {'n_pos':>5} {'n_neg':>5} | "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} | "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 130)

    transcript_by_turn = {}
    for turn_idx in all_turns:
        filtered = filter_turn_map_by_turn(transcript_turn_map, turn_idx)
        if not filtered:
            continue
        m = compute_metrics_from_turn_map(filtered)
        transcript_by_turn[turn_idx] = m
        print(f"{turn_idx:>5} | {m['n']:>5} {m['n_pos']:>5} {m['n']-m['n_pos']:>5} | "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} | "
              f"{_fmt(m['accuracy']):>7} {_fmt(m['precision']):>7} "
              f"{_fmt(m['recall']):>7} {_fmt(m['fpr']):>7} {_fmt(m['f1']):>7}")

    print(f"\n{'='*120}")
    print(f"Per-turn-idx metrics for {split.upper()} | pf_transcript_full_fires")
    print(f"{'='*120}")
    print(f"{'Turn':>5} | {'n':>5} {'n_pos':>5} {'n_neg':>5} | "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} | "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 130)

    pf_by_turn = {}
    for turn_idx in all_turns:
        filtered_eval = filter_turn_map_by_turn(transcript_turn_map, turn_idx)
        if not filtered_eval:
            continue
        # For pf_transcript_full_fires, we need the full fire map for prior-fire
        # detection, but only evaluate on the current turn's transcript_relaxed points.
        # The prior-fire check looks at ALL turns < current turn in the SAME transcript.
        m = pf_transcript_full_fires_metrics(filtered_eval, full_turn_map)
        pf_by_turn[turn_idx] = m
        print(f"{turn_idx:>5} | {m['n']:>5} {m['n_pos']:>5} {m['n']-m['n_pos']:>5} | "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} | "
              f"{_fmt(m['accuracy']):>7} {_fmt(m['precision']):>7} "
              f"{_fmt(m['recall']):>7} {_fmt(m['fpr']):>7} {_fmt(m['f1']):>7}")

    return {
        "split": split,
        "layer": layer_name,
        "thresholds": thresholds,
        "overall_transcript_relaxed": {k: _nan_to_none(v) for k, v in overall_transcript.items()},
        "overall_pf_transcript_full_fires": {k: _nan_to_none(v) for k, v in overall_pf_full.items()},
        "per_turn_transcript_relaxed": {
            t: {k: _nan_to_none(v) for k, v in m.items()} for t, m in transcript_by_turn.items()
        },
        "per_turn_pf_transcript_full_fires": {
            t: {k: _nan_to_none(v) for k, v in m.items()} for t, m in pf_by_turn.items()
        },
    }


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A"
    return f"{v:.4f}"


def main():
    search_root = Path(__file__).parent / "results" / "v2_4_combined_v3_span"

    all_results = {}
    for split in ["dev", "test"]:
        result = analyze_split(search_root, split, layer_name="layer27")
        if result:
            all_results[split] = result

    # Save results
    out_path = search_root / "per_turn_idx_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
