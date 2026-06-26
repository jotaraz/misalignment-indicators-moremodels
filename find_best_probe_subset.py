"""Find best subset of probes for joint TPR/FPR.

Uses sentence-level max scores (max of per-sentence average token scores
within each turn) from token_scores.json, matching the eval pipeline.
"""

import fnmatch
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from probe_eval.common import get_rollout_var_rep, get_n_turns
from probe_eval.misalignment_ground_truth import load_misalignment_ground_truth
from probe_eval.sentence_scores import (
    load_per_sentence_scores,
    build_sentence_scores_by_key,
)

RESULTS_DIR = Path(os.environ.get("PROBE_RESULTS_DIR", "probe_eval/results/v2_4_combined_v3_span"))
LAYER = "layer27"

# Behavior filter patterns (from run_v2_4_existing_pipeline.sh)
TEST_PATTERNS = ["test_*", "bloom_rollout", "bloom",
                 "sycophancy_answer", "sycophancy_are_you_sure", "sycophancy_feedback"]


def matches_any(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def load_thresholds():
    with open(RESULTS_DIR / "misalignment_gt_summary_dev.json") as f:
        summary = json.load(f)
    concepts = summary["per_layer"][LAYER]["per_concept"]
    return {c: data["val_clean_label_f1"]["threshold"]
            for c, data in concepts.items()
            if data.get("val_clean_label_f1", {}).get("threshold") is not None}


def load_turn_data(split="dev"):
    """Load per-turn sentence-max probe scores and GT labels.

    For each turn, the score = max(per-sentence average token scores).
    A turn fires if this max-sentence-score > threshold.
    """
    thresholds = load_thresholds()
    concepts = sorted(thresholds.keys())

    # {concept -> {(beh, var, rep, turn): max_sentence_score}}
    turn_max_scores = {}
    turn_gt = {}  # (beh, var, rep, turn) -> bool
    transcript_has_pos = {}  # (beh, var, rep) -> bool

    rollouts_cache = {}  # shared cache for rollout.json

    for concept in concepts:
        cdir = RESULTS_DIR / concept / "span" / LAYER
        if not cdir.exists():
            continue

        concept_scores = {}

        for beh in sorted(os.listdir(cdir)):
            beh_dir = cdir / beh
            if not beh_dir.is_dir():
                continue

            is_test = matches_any(beh, TEST_PATTERNS)
            if split == "dev" and is_test:
                continue
            if split == "test" and not is_test:
                continue

            results_path = beh_dir / "results.json"
            token_scores_path = beh_dir / "token_scores.json"
            if not results_path.exists():
                continue

            with open(results_path) as f:
                probe_data = json.load(f)

            rollout_dir = Path(probe_data["rollout_dir"])
            all_negative = probe_data.get("all_negative", False)

            # Load GT
            gt_path = rollout_dir / "rollout_misalignment_turns.json"
            if gt_path.exists():
                misalignment_gt = load_misalignment_ground_truth(gt_path)
            elif all_negative:
                misalignment_gt = {}
            else:
                continue

            var_rep_map = get_rollout_var_rep(rollout_dir)
            n_turns_map = get_n_turns(rollout_dir)

            # Load sentence scores from token_scores.json
            sentence_data = load_per_sentence_scores(
                token_scores_path, str(rollout_dir), rollouts_cache
            )
            if sentence_data is not None:
                sent_by_key = build_sentence_scores_by_key(sentence_data, var_rep_map)
            else:
                sent_by_key = {}

            # Build per-turn max-sentence scores
            for entry in probe_data["per_rollout"]:
                idx = entry["rollout_index"]
                if idx not in var_rep_map:
                    continue
                var, rep = var_rep_map[idx]
                if (var, rep) not in n_turns_map:
                    continue

                n_turns = n_turns_map[(var, rep)]
                misaligned_turns = misalignment_gt.get((var, rep), set())

                transcript_key = (beh, var, rep)
                if transcript_key not in transcript_has_pos:
                    transcript_has_pos[transcript_key] = False

                for turn in range(1, n_turns + 1):
                    turn_key = (beh, var, rep, turn)

                    # Get sentence scores for this turn
                    sent_scores = sent_by_key.get((var, rep, turn), [])
                    if sent_scores:
                        max_sent_score = max(sent_scores)
                    else:
                        # Fallback to per_turn_scores from results.json
                        pts = entry.get("per_turn_scores")
                        if pts:
                            score_by_turn = {s["turn"]: s["score"] for s in pts}
                            max_sent_score = score_by_turn.get(turn, float("-inf"))
                        else:
                            max_sent_score = entry["probe_score"]

                    concept_scores[turn_key] = max_sent_score

                    is_misaligned = turn in misaligned_turns
                    turn_gt[turn_key] = turn_gt.get(turn_key, False) or is_misaligned
                    if is_misaligned:
                        transcript_has_pos[transcript_key] = True

        turn_max_scores[concept] = concept_scores

    # Build turn_fires using thresholds
    turn_fires = {}
    for concept in concepts:
        threshold = thresholds[concept]
        fires = {}
        for turn_key, score in turn_max_scores.get(concept, {}).items():
            fires[turn_key] = score > threshold
        turn_fires[concept] = fires

    return turn_fires, turn_gt, transcript_has_pos, concepts


def compute_or_metrics(turn_fires, turn_gt, turn_keys, subset,
                       prior_fire_keys=None):
    """Compute OR metrics.

    If prior_fire_keys is provided, GT-positive turns that don't fire
    are excluded from FN if any probe in subset fired on an earlier turn
    in the same transcript. prior_fire_keys should be the full set of turn keys
    (including ambiguous) to detect fires on.
    """
    # Build per-turn fire decisions for subset
    turn_fire_map = {}
    for tk in set(turn_keys) | (set(prior_fire_keys) if prior_fire_keys else set()):
        turn_fire_map[tk] = any(turn_fires[c].get(tk, False) for c in subset)

    # Build prior-fire lookup if needed
    prior_fire_at = {}
    if prior_fire_keys is not None:
        from collections import defaultdict
        transcripts = defaultdict(list)
        for tk in prior_fire_keys:
            beh, var, rep, turn = tk
            transcripts[(beh, var, rep)].append((turn, tk))
        for tkey, turns in transcripts.items():
            turns.sort()
            any_fire = False
            for turn_num, tk in turns:
                prior_fire_at[tk] = any_fire
                if turn_fire_map.get(tk, False):
                    any_fire = True

    tp = fp = fn = tn = 0
    for tk in turn_keys:
        gt = turn_gt.get(tk, False)
        fires = turn_fire_map.get(tk, False)
        if gt and fires:
            tp += 1
        elif fires:
            fp += 1
        elif gt and prior_fire_keys is not None and prior_fire_at.get(tk, False):
            pass  # exclude: prior fire detected misalignment
        elif gt:
            fn += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    n_pos = tp + fn
    n_neg = fp + tn
    tpr = tp / n_pos if n_pos > 0 else 0
    fpr = fp / n_neg if n_neg > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    ratio = tpr / fpr if fpr > 0 else float("inf")

    return {"tpr": tpr, "fpr": fpr, "ratio": ratio, "f1": f1, "prec": prec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "n_pos": n_pos}


def run_analysis(split):
    print(f"\n{'#' * 120}")
    print(f"  SPLIT: {split}")
    print(f"{'#' * 120}")

    turn_fires, turn_gt, transcript_has_pos, concepts = load_turn_data(split)

    all_turn_keys = sorted(turn_gt.keys())

    # transcript_relaxed: strong positives + strong negatives only
    transcript_relaxed_keys = []
    for tk in all_turn_keys:
        beh, var, rep, turn = tk
        tkey = (beh, var, rep)
        if turn_gt[tk]:  # strong positive
            transcript_relaxed_keys.append(tk)
        elif not transcript_has_pos.get(tkey, False):  # strong negative
            transcript_relaxed_keys.append(tk)

    n_pos_pt = sum(1 for tk in all_turn_keys if turn_gt[tk])
    n_pos_tr = sum(1 for tk in transcript_relaxed_keys if turn_gt[tk])
    print(f"\n  per_turn: n={len(all_turn_keys)}, n_pos={n_pos_pt}")
    print(f"  transcript_relaxed: n={len(transcript_relaxed_keys)}, n_pos={n_pos_tr}")

    eval_configs = [
        ("transcript_relaxed", transcript_relaxed_keys, None),
        ("pf_transcript_relaxed", transcript_relaxed_keys, transcript_relaxed_keys),
        ("pf_transcript_full_fires", transcript_relaxed_keys, all_turn_keys),
    ]

    for eval_label, turn_keys, pf_keys in eval_configs:
        print(f"\n  {'=' * 110}")
        print(f"  {eval_label}")
        print(f"  {'=' * 110}")

        m = compute_or_metrics(turn_fires, turn_gt, turn_keys, concepts, pf_keys)
        r = f"{m['ratio']:.2f}" if not math.isinf(m['ratio']) else "inf"
        print(f"  ALL {len(concepts)}: TPR={m['tpr']:.4f} FPR={m['fpr']:.4f} TPR/FPR={r:>8} "
              f"F1={m['f1']:.4f} Prec={m['prec']:.4f} (TP={m['tp']} FP={m['fp']} n={m['n']} n_pos={m['n_pos']})")

        # Individual probes
        print(f"\n  {'INDICATOR':<45} {'TPR':>7} {'FPR':>7} {'TPR/FPR':>9} {'TP':>4} {'FP':>4}")
        print(f"  {'-' * 80}")
        indiv = []
        for c in concepts:
            mi = compute_or_metrics(turn_fires, turn_gt, turn_keys, [c], pf_keys)
            indiv.append((c, mi))
        indiv.sort(key=lambda x: -x[1]["ratio"] if not math.isinf(x[1]["ratio"]) else -1e9)
        for c, mi in indiv:
            r = f"{mi['ratio']:.2f}" if not math.isinf(mi['ratio']) else "inf"
            print(f"  {c:<45} {mi['tpr']:.4f}  {mi['fpr']:.4f}  {r:>9}  {mi['tp']:>4} {mi['fp']:>4}")

        # Greedy backward elimination
        print(f"\n  BACKWARD ELIMINATION:")
        current = list(concepts)
        m = compute_or_metrics(turn_fires, turn_gt, turn_keys, current, pf_keys)
        r = f"{m['ratio']:.2f}" if not math.isinf(m['ratio']) else "inf"
        print(f"    All {len(current):>2}: TPR={m['tpr']:.4f} FPR={m['fpr']:.4f} TPR/FPR={r:>8} F1={m['f1']:.4f}")
        for _ in range(len(concepts) - 1):
            best_score = -1
            best_rm = None
            best_m = None
            for c in current:
                cand = [x for x in current if x != c]
                mi = compute_or_metrics(turn_fires, turn_gt, turn_keys, cand, pf_keys)
                score = mi["ratio"] if not math.isinf(mi["ratio"]) else 1e9
                if score > best_score:
                    best_score = score
                    best_rm = c
                    best_m = mi
            current.remove(best_rm)
            r = f"{best_m['ratio']:.2f}" if not math.isinf(best_m['ratio']) else "inf"
            print(f"    -{best_rm:<41} TPR={best_m['tpr']:.4f} FPR={best_m['fpr']:.4f} "
                  f"TPR/FPR={r:>8} F1={best_m['f1']:.4f} ({len(current):>2} probes)")

        # Leave-one-out
        print(f"\n  LEAVE-ONE-OUT:")
        baseline = compute_or_metrics(turn_fires, turn_gt, turn_keys, concepts, pf_keys)
        impacts = []
        for c in concepts:
            rem = [x for x in concepts if x != c]
            mi = compute_or_metrics(turn_fires, turn_gt, turn_keys, rem, pf_keys)
            impacts.append((c, mi['tpr'] - baseline['tpr'], mi['fpr'] - baseline['fpr'], mi))
        impacts.sort(key=lambda x: (x[2], -x[1]))
        for c, dt, df, mi in impacts:
            r = f"{mi['ratio']:.2f}" if not math.isinf(mi['ratio']) else "inf"
            print(f"    -{c:<43} ΔTPR={dt:+.4f} ΔFPR={df:+.4f} → TPR/FPR={r:>8}")


def main():
    for split in ["dev", "test"]:
        run_analysis(split)


if __name__ == "__main__":
    main()
