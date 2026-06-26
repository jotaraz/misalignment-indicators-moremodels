"""
Sweep FPR targets across probe versions to produce comparable operating points.

For each version, builds per-probe points once (dev & test), then sweeps
tune_all_thresholds at multiple FPR targets. Reports prior_fire_transcript_relaxed.

Layer 27 only. No per-concept output.

Usage:
    python -m probe_eval.sweep_fpr_targets
    python -m probe_eval.sweep_fpr_targets --fpr-min 0.005 --fpr-max 0.025 --fpr-step 0.002
    python -m probe_eval.sweep_fpr_targets --versions v3_v2_5_combined_v1_span
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from probe_eval.misalignment_ground_truth import (
    discover_results,
    tune_all_thresholds,
    build_transcript_relaxed_points,
    load_misalignment_ground_truth,
    get_rollout_var_rep,
    get_n_turns,
)
from probe_eval.common import filter_behaviors, get_concept_from_experiment_folder
from probe_eval.metrics import (
    apply_thresholds_or_misalignment as _apply_thresholds_or_per_turn,
    apply_thresholds_or_misalignment_prior_fire as _apply_thresholds_or_pf,
    apply_thresholds_or_transcript_level as _apply_thresholds_or_transcript,
)
from probe_eval.sentence_scores import (
    build_sentence_scores_by_key,
    load_per_sentence_scores,
)

RESULTS_BASE = REPO_ROOT / "probe_eval" / "results"

# Each version entry is (results_subdir, label_mode_filter).
# label_mode_filter: "span", "turn", or None (match any).
DEFAULT_VERSIONS = [
    ("v3_v2_5_combined_v1_span", "span"),            # 18-indicator span
    ("v3_v2_5_combined_v1_span", "turn"),             # 18-indicator turn
    ("v3_v2_5_combined_v1_mechanism_span", None),     # 10-mechanism
    ("v3_v2_5_combined_v1_behavior_span", None),      # 10-behavior
    ("v3_v2_5_combined_v1_7probe_span", None),        # 7-probe
    ("v3_v2_5_combined_v1_1probe_span", None),        # 1-probe
]

VERSION_NAMES = {
    ("v3_v2_5_combined_v1_span", "span"): "18-span",
    ("v3_v2_5_combined_v1_span", "turn"): "18-turn",
    ("v3_v2_5_combined_v1_mechanism_span", None): "10-mech",
    ("v3_v2_5_combined_v1_behavior_span", None): "10-behav",
    ("v3_v2_5_combined_v1_7probe_span", None): "7-probe",
    ("v3_v2_5_combined_v1_1probe_span", None): "1-probe",
}

DEV_EXCLUDE = "test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback,mask,benign_*,hard_negatives_*"
TEST_INCLUDE = "test_*,bloom_rollout,bloom,sycophancy_answer,sycophancy_are_you_sure,sycophancy_feedback,mask"

LAYER = "layer27"


def _nan_to_none(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def to_sentence_max_points(points, sent_by_key):
    """Convert per-turn points to sentence-max score points.

    Mirrors misalignment_ground_truth.to_sentence_max_points.
    """
    from probe_eval.misalignment_ground_truth import to_sentence_max_points as _to_smp
    return _to_smp(points, sent_by_key)


def build_per_probe_points(results_subdir, include_pats=None, exclude_pats=None,
                           label_mode_filter=None):
    """Build per-probe transcript_relaxed points for layer 27.

    Mirrors the combined-metrics section of misalignment_ground_truth.main(),
    but only for one layer and without per-concept / sentence / visualization output.

    Returns:
        per_probe_transcript: dict[concept, list[point]]  — for tuning & eval
        val_thresholds_fpr1: dict[concept, float]  — starting thresholds
    """
    search_root = RESULTS_BASE / results_subdir

    # Only discover results for layer27 (much faster than scanning all layers)
    all_results = []
    for rj in sorted(search_root.glob(f"*/*/{LAYER}/*/results.json")):
        with open(rj) as f:
            data = json.load(f)
        rollout_dir = data.get("rollout_dir", "")
        if not rollout_dir:
            continue
        is_all_negative = data.get("all_negative", False)
        gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
        has_gt = gt_path.exists()
        if has_gt:
            is_all_negative = False
        elif not is_all_negative:
            continue
        behavior = rj.parent.name
        concept = get_concept_from_experiment_folder(data.get("experiment_folder", ""))
        if concept is None:
            continue
        all_results.append({
            "behavior": behavior,
            "concept": concept,
            "experiment_folder": data["experiment_folder"],
            "rollout_dir": rollout_dir,
            "result_path": rj,
            "data": data,
            "all_negative": is_all_negative,
        })

    by_behavior = {}
    for r in all_results:
        by_behavior.setdefault(r["behavior"], []).append(r)
    if include_pats or exclude_pats:
        by_behavior = filter_behaviors(by_behavior, include_pats, exclude_pats)

    # GT cache
    gt_cache = {}
    rollouts_full_cache = {}

    # raw_turn_data[(behavior)][concept] = {"points": [...], "sent_by_key": {...}}
    raw_turn_data = defaultdict(dict)

    for behavior, behavior_results in by_behavior.items():
        rollout_dir = behavior_results[0]["rollout_dir"]
        is_all_negative = behavior_results[0].get("all_negative", False)

        # Load GT
        if rollout_dir not in gt_cache:
            gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
            if gt_path.exists():
                gt_cache[rollout_dir] = {
                    "misalignment_gt": load_misalignment_ground_truth(gt_path),
                    "var_rep_map": get_rollout_var_rep(rollout_dir),
                    "n_turns_map": get_n_turns(rollout_dir),
                }
            elif is_all_negative:
                var_rep_map = get_rollout_var_rep(rollout_dir)
                gt_cache[rollout_dir] = {
                    "misalignment_gt": {vr: set() for vr in var_rep_map.values()},
                    "var_rep_map": var_rep_map,
                    "n_turns_map": get_n_turns(rollout_dir),
                }
            else:
                gt_cache[rollout_dir] = None

        gd = gt_cache.get(rollout_dir)
        if gd is None:
            continue

        misalignment_gt = gd["misalignment_gt"]
        var_rep_map = gd["var_rep_map"]
        n_turns_map = gd["n_turns_map"]

        for r in behavior_results:
            exp_folder = r["experiment_folder"]
            if LAYER not in exp_folder:
                continue
            # Filter by label mode (span/turn) if specified
            if label_mode_filter:
                if f"/{label_mode_filter}/" not in exp_folder:
                    continue

            concept = r["concept"]
            data = r["data"]

            points = []
            for rollout in data.get("per_rollout", []):
                ridx = rollout["rollout_index"]
                vr = var_rep_map.get(ridx)
                if vr is None:
                    continue
                gt_turns = misalignment_gt.get(vr, set())

                for turn_data in rollout.get("per_turn_scores", []):
                    turn_idx = turn_data.get("turn_index", turn_data.get("turn"))
                    sent_scores = turn_data.get("sentence_scores", [])
                    if sent_scores:
                        score = max(s["score"] for s in sent_scores)
                    else:
                        score = turn_data.get("score", turn_data.get("probe_score", float("nan")))

                    points.append({
                        "probe_score": score,
                        "gt_misaligned": turn_idx in gt_turns,
                        "var": vr[0],
                        "rep": vr[1],
                        "turn": turn_idx,
                        "behavior": behavior,
                    })

            # Load sentence scores for sentence-max aggregation
            token_scores_path = r["result_path"].parent / "token_scores.json"
            sent_by_key = {}
            sentence_data = load_per_sentence_scores(
                token_scores_path, rollout_dir, rollouts_full_cache,
                short_sentence_mode="discard", min_words=5,
            )
            if sentence_data is not None:
                sent_by_key = build_sentence_scores_by_key(sentence_data, var_rep_map)

            raw_turn_data[behavior].setdefault(concept, {"points": [], "sent_by_key": {}})
            raw_turn_data[behavior][concept]["points"].extend(points)
            raw_turn_data[behavior][concept]["sent_by_key"].update(sent_by_key)

    # Build per_probe_combined with sentence-max scores (matching main())
    per_probe_combined_sentence = defaultdict(list)
    for behavior, probes_data in raw_turn_data.items():
        for concept, info in probes_data.items():
            sentence_points = to_sentence_max_points(info["points"], info["sent_by_key"])
            for p in sentence_points:
                if np.isnan(p["probe_score"]):
                    continue
                tagged_p = {**p, "var": f"{behavior}__{p['var']}"}
                per_probe_combined_sentence[concept].append(tagged_p)

    per_probe_combined = dict(per_probe_combined_sentence)

    # Build transcript_relaxed
    per_probe_transcript = {}
    for concept, pts in per_probe_combined.items():
        per_probe_transcript[concept] = build_transcript_relaxed_points(pts)

    # per_probe_combined holds sentence-max per-turn points (one per (var, rep, turn)).
    # Used for per_turn and transcript_level metrics downstream.
    per_probe_per_turn = per_probe_combined

    # Extract val thresholds from training metadata.
    # The experiment_folder in results.json points to the probe directory,
    # which contains cfg.yaml and training_meta.json.
    import yaml
    val_thresholds_fpr1 = {}
    val_thresholds_fpr5 = {}
    seen_concepts = set()
    for results_list in by_behavior.values():
        for r in results_list:
            exp_folder = r["experiment_folder"]
            if LAYER not in exp_folder:
                continue
            if label_mode_filter and f"/{label_mode_filter}/" not in exp_folder:
                continue
            concept = get_concept_from_experiment_folder(exp_folder)
            if not concept or concept in seen_concepts:
                continue
            seen_concepts.add(concept)

            meta_path = Path(exp_folder) / "training_meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            cl = meta.get("val_thresholds", {}).get("clean_label", {})
            t1 = cl.get("fpr_1pct", {}).get("threshold")
            t5 = cl.get("fpr_5pct", {}).get("threshold")
            if t1 is not None:
                val_thresholds_fpr1[concept] = t1
            if t5 is not None:
                val_thresholds_fpr5[concept] = t5

    return (per_probe_transcript, per_probe_per_turn,
            val_thresholds_fpr1, val_thresholds_fpr5)


def derive_fpr(m):
    recall = m.get("recall", 0) or 0
    prec = m.get("precision", 0) or 0
    n = m.get("n", 0) or 0
    n_pos = m.get("n_pos", 0) or 0
    n_neg = n - n_pos
    if prec == 0 or n_neg == 0:
        return 0.0
    tp = recall * n_pos
    fp = tp / prec - tp
    return fp / n_neg


def main():
    parser = argparse.ArgumentParser(description="Sweep FPR targets across probe versions")
    parser.add_argument("--fpr-min", type=float, default=0.010)
    parser.add_argument("--fpr-max", type=float, default=0.020)
    parser.add_argument("--fpr-step", type=float, default=0.002)
    parser.add_argument(
        "--fpr-targets", type=str, default=None,
        help="Explicit comma-separated FPR target list (e.g. '0.005,0.01,0.02,0.05,0.1,0.5'). "
             "If given, overrides --fpr-min/--fpr-max/--fpr-step.",
    )
    parser.add_argument("--versions", nargs="+", default=None)
    parser.add_argument("--starting-threshold", type=str, default="fpr1pct",
                        choices=["fpr1pct", "fpr5pct"],
                        help="Starting threshold variant (default: fpr1pct)")
    parser.add_argument(
        "--allow-tune-down", action="store_true",
        help="When a probe's starting FPR is already ≤ target, lower the threshold "
             "to match the target instead of keeping it. Use for AUROC sweeps.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Parse versions: support both tuple entries and plain strings from CLI
    if args.versions:
        # CLI args are plain strings — treat as (subdir, None)
        versions = [(v, None) for v in args.versions]
    else:
        versions = DEFAULT_VERSIONS

    if args.fpr_targets:
        fpr_targets = [round(float(x.strip()), 4) for x in args.fpr_targets.split(",") if x.strip()]
    else:
        fpr_targets = []
        t = args.fpr_min
        while t <= args.fpr_max + 1e-9:
            fpr_targets.append(round(t, 4))
            t += args.fpr_step

    print(f"Versions: {[VERSION_NAMES.get(v, v) for v in versions]}")
    print(f"FPR targets: {fpr_targets}")
    print(f"Layer: {LAYER}\n")

    dev_exclude_pats = DEV_EXCLUDE.split(",")
    test_include_pats = TEST_INCLUDE.split(",")

    all_results = {}

    for version_key in versions:
        subdir, label_mode = version_key
        name = VERSION_NAMES.get(version_key, f"{subdir}/{label_mode or 'any'}")
        print(f"\n{'='*70}")
        print(f"{name} ({subdir}, label_mode={label_mode or 'any'})")
        print(f"{'='*70}")

        # Build points once for dev and test
        print("  Building dev points...")
        dev_transcript, dev_per_turn, thresholds_fpr1, thresholds_fpr5 = build_per_probe_points(
            subdir, exclude_pats=dev_exclude_pats, label_mode_filter=label_mode,
        )
        start_thresholds = thresholds_fpr1 if args.starting_threshold == "fpr1pct" else thresholds_fpr5
        n_dev_pts = sum(len(pts) for pts in dev_transcript.values())
        n_dev_turn_pts = sum(len(pts) for pts in dev_per_turn.values())
        print(f"    {len(dev_transcript)} probes, {n_dev_pts} transcript pts, "
              f"{n_dev_turn_pts} per-turn pts, "
              f"{len(start_thresholds)} starting thresholds ({args.starting_threshold})")

        print("  Building test points...")
        test_transcript, test_per_turn, _, _ = build_per_probe_points(
            subdir, include_pats=test_include_pats, label_mode_filter=label_mode,
        )
        n_test_pts = sum(len(pts) for pts in test_transcript.values())
        n_test_turn_pts = sum(len(pts) for pts in test_per_turn.values())
        print(f"    {len(test_transcript)} probes, {n_test_pts} transcript pts, "
              f"{n_test_turn_pts} per-turn pts")

        if not dev_transcript or not start_thresholds:
            print("  Skipping (no data or thresholds)")
            continue

        def _eval(points_transcript, points_per_turn, thresholds):
            pftr = _apply_thresholds_or_pf(dict(points_transcript), thresholds)
            pftr = {k: _nan_to_none(v) for k, v in pftr.items()}
            trans = _apply_thresholds_or_transcript(dict(points_per_turn), thresholds)
            trans = {k: _nan_to_none(v) for k, v in trans.items()}
            per_turn = _apply_thresholds_or_per_turn(dict(points_per_turn), thresholds)
            per_turn = {k: _nan_to_none(v) for k, v in per_turn.items()}
            return {
                "prior_fire_transcript_relaxed": pftr,
                "derived_fpr": derive_fpr(pftr),
                "transcript_level": trans,
                "transcript_level_fpr": derive_fpr(trans),
                "per_turn": per_turn,
                "per_turn_fpr": derive_fpr(per_turn),
            }

        version_results = []
        for fpr_target in fpr_targets:
            # Tune on dev (transcript-relaxed points, prior-fire metric)
            tuned_thresholds, _ = tune_all_thresholds(
                dict(dev_transcript), start_thresholds, fpr_target=fpr_target,
                allow_tune_down=args.allow_tune_down,
            )

            dev_m = _eval(dev_transcript, dev_per_turn, tuned_thresholds)
            test_m = _eval(test_transcript, test_per_turn, tuned_thresholds)

            pftr_d = dev_m["prior_fire_transcript_relaxed"]
            pftr_t = test_m["prior_fire_transcript_relaxed"]
            tr_t = test_m["transcript_level"]
            pt_t = test_m["per_turn"]
            print(f"  fpr_tgt={fpr_target:.3f} | "
                  f"dev_pftr: TPR={pftr_d.get('recall',0):.4f} FPR={dev_m['derived_fpr']:.4f} | "
                  f"test_pftr: TPR={pftr_t.get('recall',0):.4f} FPR={test_m['derived_fpr']:.4f} "
                  f"F1={pftr_t.get('f1',0):.4f} | "
                  f"test_trans: TPR={tr_t.get('recall',0):.4f} FPR={test_m['transcript_level_fpr']:.4f} "
                  f"F1={tr_t.get('f1',0):.4f} | "
                  f"test_turn: TPR={pt_t.get('recall',0):.4f} FPR={test_m['per_turn_fpr']:.4f} "
                  f"F1={pt_t.get('f1',0):.4f}")

            version_results.append({
                "fpr_target": fpr_target,
                "dev": dev_m,
                "test": test_m,
            })

        result_key = f"{subdir}__{label_mode or 'any'}"
        all_results[result_key] = {
            "display_name": name,
            "n_probes": len(start_thresholds),
            "results_subdir": subdir,
            "label_mode": label_mode,
            "results": version_results,
        }

    # Summary table — test metrics
    print(f"\n\n{'='*120}")
    print("Summary — Test metrics, layer 27 "
          "(pftr = prior-fire transcript_relaxed, trans = transcript_level, turn = per_turn)")
    print(f"{'='*120}")
    print(f"{'Version':12s} | {'#P':>3s} | {'FPR_tgt':>8s} | "
          f"{'pftr_TPR':>8s} {'pftr_FPR':>8s} {'pftr_F1':>8s} | "
          f"{'trn_TPR':>8s} {'trn_FPR':>8s} {'trn_F1':>8s} | "
          f"{'tur_TPR':>8s} {'tur_FPR':>8s} {'tur_F1':>8s}")
    print("-" * 120)

    for version_key in versions:
        subdir, label_mode = version_key
        result_key = f"{subdir}__{label_mode or 'any'}"
        vdata = all_results.get(result_key)
        if not vdata:
            continue
        for r in vdata["results"]:
            pftr = r["test"]["prior_fire_transcript_relaxed"]
            trans = r["test"]["transcript_level"]
            turn = r["test"]["per_turn"]
            print(
                f"{vdata['display_name']:12s} | {vdata['n_probes']:>3d} | "
                f"{r['fpr_target']:>8.3f} | "
                f"{(pftr.get('recall') or 0):>8.4f} "
                f"{r['test']['derived_fpr']:>8.4f} "
                f"{(pftr.get('f1') or 0):>8.4f} | "
                f"{(trans.get('recall') or 0):>8.4f} "
                f"{r['test']['transcript_level_fpr']:>8.4f} "
                f"{(trans.get('f1') or 0):>8.4f} | "
                f"{(turn.get('recall') or 0):>8.4f} "
                f"{r['test']['per_turn_fpr']:>8.4f} "
                f"{(turn.get('f1') or 0):>8.4f}"
            )
        print()

    # Save
    output_path = Path(args.output) if args.output else RESULTS_BASE / "fpr_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
