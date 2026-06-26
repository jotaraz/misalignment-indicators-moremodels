"""
Split probe firing into reasoning-only and response-only partitions.

The probes are trained on reasoning tokens (inside ``<think>...</think>``).
This script asks: if we apply the same trained probes to
  (a) the reasoning portion of each turn, and
  (b) the response portion after ``</think>``,
what detection performance do we get on each partition separately?

Uses existing token_scores.json (which already contains per-token scores for
all detection-mask-covered tokens) and the tuned-threshold JSON. For each
rollout × turn we locate the ``<think>`` / ``</think>`` token boundaries
within the scored-token stream, split tokens into reasoning / response
buckets, sentence-split each bucket independently, then apply thresholds
and compute OR-across-probes TPR/FPR against the transcript-level GT.

Usage (single behavior, layer 27, tuned fpr_0.02):
    python -m probe_eval.reasoning_vs_output_metrics \\
        --results-subdir v4_v2_6_combined_v2_span \\
        --tuned-thresholds probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds_discard.json \\
        --tuned-threshold-version fpr_0.02 \\
        --layer 27 \\
        --include-behaviors "test_*"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from probe_eval.common import (
    NON_INDICATOR_CONCEPTS,
    RESULTS_DIR,
    add_behavior_filter_args,
    filter_behaviors,
    get_n_turns,
    get_rollout_var_rep,
)
from probe_eval.misalignment_ground_truth import (
    discover_results,
    load_misalignment_ground_truth,
    load_unreliable_rollouts,
)
from probe_eval.sentence_scores import (
    _get_assistant_turns_for_rollout,
    _split_tokens_into_sentences,
)


# Reasoning-block tag pairs. Different pipelines use different markers:
#   <think>...</think>                             (bloom default)
#   <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING> (agentic-misalignment)
REASONING_TAG_PAIRS = [
    ("<think>", "</think>"),
    ("<SCRATCHPAD_REASONING>", "</SCRATCHPAD_REASONING>"),
]


def _find_think_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    """Return token-index ranges that lie *between* reasoning-block tags.

    Supports both ``<think>`` and ``<SCRATCHPAD_REASONING>`` tag families, and
    multi-token splits (e.g. tokenizer breaks ``<SCRATCHPAD_REASONING>`` into
    several sub-word tokens). Detection is done on the joined text with a
    char-to-token map, then mapped back to token indices.
    """
    if not tokens:
        return []
    # Build char positions for each token
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)
        ends.append(pos)
    full = "".join(tokens)

    def char_to_token(ch: int, round_up: bool = False) -> int:
        """Map a char position to the nearest token index."""
        # Binary search on ``starts``.
        lo, hi = 0, len(tokens)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= ch < ends[mid]:
                return mid + 1 if round_up else mid
            if starts[mid] < ch:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # Greedy match of open/close pairs in the joined text.
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(full):
        # Find the earliest opening tag at or after cursor.
        best_open = None  # (char_pos, close_tag, open_tag_len)
        for open_t, close_t in REASONING_TAG_PAIRS:
            idx = full.find(open_t, cursor)
            if idx != -1 and (best_open is None or idx < best_open[0]):
                best_open = (idx, close_t, len(open_t))
        if best_open is None:
            break
        open_pos, close_t, open_len = best_open
        # Find matching close
        close_pos = full.find(close_t, open_pos + open_len)
        content_start_char = open_pos + open_len
        if close_pos == -1:
            content_end_char = len(full)
        else:
            content_end_char = close_pos

        if content_end_char > content_start_char:
            # Map char range back to token index range (inclusive/exclusive).
            start_tok = char_to_token(content_start_char, round_up=False)
            end_tok = char_to_token(content_end_char - 1, round_up=False) + 1
            if end_tok > start_tok:
                ranges.append((start_tok, end_tok))

        cursor = content_end_char + (len(close_t) if close_pos != -1 else 0)
    return ranges


def _partition_turn(
    tokens: list[str], scores: list[float | None],
) -> tuple[tuple[list[str], list[float | None]], tuple[list[str], list[float | None]]]:
    """Partition one turn's scored tokens into (reasoning, response) buckets.

    Boundary tokens (``<think>`` / ``</think>``) themselves go into reasoning's
    bucket so char offsets line up, but they are short and will be merged by
    the usual short-sentence-merge rule.
    """
    ranges = _find_think_ranges(tokens)
    r_tokens: list[str] = []
    r_scores: list[float | None] = []
    o_tokens: list[str] = []
    o_scores: list[float | None] = []
    in_reasoning = [False] * len(tokens)
    for s, e in ranges:
        for i in range(s, e):
            in_reasoning[i] = True
    for i, tok in enumerate(tokens):
        if in_reasoning[i]:
            r_tokens.append(tok)
            r_scores.append(scores[i])
        else:
            # Tag-boundary tokens are part of neither bucket.
            # With multi-subword tags we can't cleanly detect every piece;
            # rely on the reasoning-range exclusion handling most of it and
            # drop any token containing a simple <think> / </think> literal.
            if "<think>" in tok or "</think>" in tok:
                continue
            o_tokens.append(tok)
            o_scores.append(scores[i])
    return (r_tokens, r_scores), (o_tokens, o_scores)


def compute_partition_fires(
    search_root: Path,
    behavior: str,
    layer: str,
    thresholds: dict[str, float],
    short_sentence_mode: str = "discard",
    min_sentence_words: int = 5,
    rollouts_cache: dict[str, list[dict]] | None = None,
) -> dict[str, Any] | None:
    """Return per-turn fire decisions split by partition for one behavior.

    Output structure:
        {
            "rollout_dir": str,
            "turn_has_reasoning": set of (var, rep, turn),
            "turn_has_response":  set of (var, rep, turn),
            "fires_reasoning":    set of (var, rep, turn)  (any probe fired in reasoning),
            "fires_response":     set of (var, rep, turn)  (any probe fired in response),
            "scored_turns":       set of (var, rep, turn),
        }
    """
    rollouts_cache = rollouts_cache if rollouts_cache is not None else {}
    rollout_dir: str | None = None
    fires_r: set[tuple[int, int, int]] = set()
    fires_o: set[tuple[int, int, int]] = set()
    has_r: set[tuple[int, int, int]] = set()
    has_o: set[tuple[int, int, int]] = set()
    scored: set[tuple[int, int, int]] = set()

    any_probe = False
    for concept, thresh in thresholds.items():
        probe_beh_dir = search_root / concept / "span" / layer / behavior
        ts_path = probe_beh_dir / "token_scores.json"
        res_path = probe_beh_dir / "results.json"
        if not ts_path.exists() or not res_path.exists():
            continue
        any_probe = True
        if rollout_dir is None:
            rollout_dir = json.load(open(res_path))["rollout_dir"]
        ts_data = json.load(open(ts_path))

        # Reload rollouts once
        rollout_dir_str = str(rollout_dir)
        if rollout_dir_str not in rollouts_cache:
            rollouts_cache[rollout_dir_str] = json.load(
                open(Path(rollout_dir) / "rollout.json")
            )["rollouts"]
        rollouts = rollouts_cache[rollout_dir_str]
        var_rep_map = get_rollout_var_rep(rollout_dir)

        for ts_entry in ts_data["per_rollout"]:
            idx = ts_entry["rollout_index"]
            if idx >= len(rollouts) or idx not in var_rep_map:
                continue
            tokens = ts_entry["tokens"]
            scores = ts_entry["scores"]

            # Scored segments (consecutive non-None runs)
            segments: list[tuple[int, int]] = []
            seg_s: int | None = None
            for i, s in enumerate(scores):
                if s is not None:
                    if seg_s is None:
                        seg_s = i
                else:
                    if seg_s is not None:
                        segments.append((seg_s, i))
                        seg_s = None
            if seg_s is not None:
                segments.append((seg_s, len(scores)))

            assistant_turns = _get_assistant_turns_for_rollout(rollouts[idx])
            turn_token_lists: dict[int, list[str]] = defaultdict(list)
            turn_score_lists: dict[int, list[float | None]] = defaultdict(list)
            for seg_idx, (s0, s1) in enumerate(segments):
                turn = assistant_turns[seg_idx] if seg_idx < len(assistant_turns) else 0
                turn_token_lists[turn].extend(tokens[s0:s1])
                turn_score_lists[turn].extend(scores[s0:s1])

            var, rep = var_rep_map[idx]
            for turn, tok_list in turn_token_lists.items():
                sc_list = turn_score_lists[turn]
                (r_tok, r_sc), (o_tok, o_sc) = _partition_turn(tok_list, sc_list)

                key = (var, rep, turn)
                scored.add(key)
                if r_tok:
                    has_r.add(key)
                if o_tok:
                    has_o.add(key)

                if r_tok:
                    sent_scores = _split_tokens_into_sentences(
                        r_tok,
                        # None scores shouldn't occur within the kept ranges,
                        # but just in case fall back to 0.
                        [s if s is not None else 0.0 for s in r_sc],
                        short_sentence_mode=short_sentence_mode,
                        min_words=min_sentence_words,
                    )
                    if any(s > thresh for s in sent_scores):
                        fires_r.add(key)

                if o_tok:
                    sent_scores = _split_tokens_into_sentences(
                        o_tok,
                        [s if s is not None else 0.0 for s in o_sc],
                        short_sentence_mode=short_sentence_mode,
                        min_words=min_sentence_words,
                    )
                    if any(s > thresh for s in sent_scores):
                        fires_o.add(key)

    if not any_probe:
        return None
    return {
        "rollout_dir": rollout_dir,
        "fires_reasoning": fires_r,
        "fires_response": fires_o,
        "turn_has_reasoning": has_r,
        "turn_has_response": has_o,
        "scored_turns": scored,
    }


def metrics_for_partition(
    fires: set[tuple[int, int, int]],
    turn_has: set[tuple[int, int, int]],
    gt: dict[tuple[int, int], set[int]],
    ambiguous: set[tuple[int, int]],
    n_turns_map: dict[tuple[int, int], int],
) -> dict[str, float]:
    """Turn-level TPR/FPR restricted to turns that have this partition at all."""
    tp = fp = tn = fn = 0
    n_trans = n_pos_trans = 0
    tp_trans = 0
    for (var, rep), mis_turns in gt.items():
        if (var, rep) in ambiguous:
            continue
        n_turns = n_turns_map.get((var, rep), 0)
        any_fire_on_pos = False
        is_pos = bool(mis_turns)
        n_trans += 1
        if is_pos:
            n_pos_trans += 1
        for t in range(1, n_turns + 1):
            key = (var, rep, t)
            if key not in turn_has:
                continue
            fired = key in fires
            gt_pos = t in mis_turns
            if fired and gt_pos:
                tp += 1
                any_fire_on_pos = True
            elif fired:
                fp += 1
                any_fire_on_pos = any_fire_on_pos  # still can be True later
            elif gt_pos:
                fn += 1
            else:
                tn += 1
        if is_pos:
            # transcript-level TP = any turn fire on a GT-positive transcript
            if any_fire_on_pos:
                tp_trans += 1
            else:
                # Check if any fire happened on ANY turn (even non-positive turns):
                # for transcript TPR we also count those as successful flagging.
                for t in range(1, n_turns + 1):
                    key = (var, rep, t)
                    if key in fires:
                        tp_trans += 1
                        break
    n = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    trans_tpr = tp_trans / n_pos_trans if n_pos_trans > 0 else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr,
        "turn_n": n, "turn_n_pos": tp + fn,
        "transcript_tpr": trans_tpr,
        "n_pos_trans": n_pos_trans,
        "tp_trans": tp_trans,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-subdir", required=True)
    parser.add_argument("--tuned-thresholds", required=True)
    parser.add_argument("--tuned-threshold-version", default="fpr_0.02")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--short-sentence-mode", default="discard",
                        choices=["merge", "discard"])
    parser.add_argument("--min-sentence-words", type=int, default=5)
    parser.add_argument("--include-all-negative", action="store_true")
    parser.add_argument("--exclude-ambiguous", action="store_true", default=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    add_behavior_filter_args(parser)
    args = parser.parse_args()

    layer = f"layer{args.layer}"
    search_root = RESULTS_DIR / args.results_subdir

    tt = json.load(open(args.tuned_thresholds))
    thresholds = (
        tt["versions"][args.tuned_threshold_version]
        .get("per_layer", {}).get(layer, {}).get("thresholds", {})
    )
    if not thresholds:
        print(f"No thresholds for {layer} in version {args.tuned_threshold_version}")
        return
    print(f"Loaded {len(thresholds)} thresholds ({args.tuned_threshold_version}, {layer})")

    results = discover_results(
        search_root, exclude_concepts=NON_INDICATOR_CONCEPTS,
        include_all_negative=args.include_all_negative,
    )
    by_behavior: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_behavior[r["behavior"]].append(r)
    include_pats = args.include_behaviors.split(",") if args.include_behaviors else None
    exclude_pats = args.exclude_behaviors.split(",") if args.exclude_behaviors else None
    if include_pats or exclude_pats:
        by_behavior = filter_behaviors(by_behavior, include_pats, exclude_pats)
    behaviors = sorted(by_behavior.keys())
    print(f"Behaviors: {len(behaviors)}")

    rollouts_cache: dict[str, list[dict]] = {}
    per_behavior_rows: list[dict] = []

    for behavior in behaviors:
        fires = compute_partition_fires(
            search_root, behavior, layer, thresholds,
            short_sentence_mode=args.short_sentence_mode,
            min_sentence_words=args.min_sentence_words,
            rollouts_cache=rollouts_cache,
        )
        if fires is None:
            print(f"  [skip] {behavior}: no probe results")
            continue
        rollout_dir = fires["rollout_dir"]
        gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
        if gt_path.exists():
            gt = load_misalignment_ground_truth(
                gt_path, exclude_ambiguous=args.exclude_ambiguous,
            )
            ambiguous = load_unreliable_rollouts(gt_path)
        else:
            gt = {vr: set() for vr in get_rollout_var_rep(rollout_dir).values()}
            ambiguous = set()
        n_turns_map = get_n_turns(rollout_dir)

        reasoning_m = metrics_for_partition(
            fires["fires_reasoning"], fires["turn_has_reasoning"],
            gt, ambiguous, n_turns_map,
        )
        response_m = metrics_for_partition(
            fires["fires_response"], fires["turn_has_response"],
            gt, ambiguous, n_turns_map,
        )

        row = {
            "behavior": behavior,
            "reasoning": reasoning_m,
            "response": response_m,
            "n_turns_with_reasoning": len(fires["turn_has_reasoning"]),
            "n_turns_with_response": len(fires["turn_has_response"]),
        }
        per_behavior_rows.append(row)

    # Print table
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) and v == v else "  N/A"
    print(f"\n{'Behavior':<55} "
          f"{'R-TPR':>6} {'R-FPR':>6} {'R-T.TPR':>8} "
          f"{'O-TPR':>6} {'O-FPR':>6} {'O-T.TPR':>8} "
          f"{'n_R':>5} {'n_O':>5}")
    print("-" * 130)
    for r in per_behavior_rows:
        print(f"{r['behavior']:<55} "
              f"{fmt(r['reasoning']['tpr']):>6} {fmt(r['reasoning']['fpr']):>6} "
              f"{fmt(r['reasoning']['transcript_tpr']):>8} "
              f"{fmt(r['response']['tpr']):>6} {fmt(r['response']['fpr']):>6} "
              f"{fmt(r['response']['transcript_tpr']):>8} "
              f"{r['n_turns_with_reasoning']:>5} {r['n_turns_with_response']:>5}")

    # Aggregate totals across behaviors
    agg_r = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tp_trans": 0, "n_pos_trans": 0}
    agg_o = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tp_trans": 0, "n_pos_trans": 0}
    for r in per_behavior_rows:
        for k in agg_r:
            agg_r[k] += r["reasoning"][k]
            agg_o[k] += r["response"][k]
    def _ratio(a, b): return (a / b) if b > 0 else float("nan")
    agg_row = {
        "reasoning": {
            "tpr": _ratio(agg_r["tp"], agg_r["tp"] + agg_r["fn"]),
            "fpr": _ratio(agg_r["fp"], agg_r["fp"] + agg_r["tn"]),
            "transcript_tpr": _ratio(agg_r["tp_trans"], agg_r["n_pos_trans"]),
            **agg_r,
        },
        "response": {
            "tpr": _ratio(agg_o["tp"], agg_o["tp"] + agg_o["fn"]),
            "fpr": _ratio(agg_o["fp"], agg_o["fp"] + agg_o["tn"]),
            "transcript_tpr": _ratio(agg_o["tp_trans"], agg_o["n_pos_trans"]),
            **agg_o,
        },
    }
    print("-" * 130)
    print(f"{'TOTAL':<55} "
          f"{fmt(agg_row['reasoning']['tpr']):>6} {fmt(agg_row['reasoning']['fpr']):>6} "
          f"{fmt(agg_row['reasoning']['transcript_tpr']):>8} "
          f"{fmt(agg_row['response']['tpr']):>6} {fmt(agg_row['response']['fpr']):>6} "
          f"{fmt(agg_row['response']['transcript_tpr']):>8}")

    if args.output:
        out = {
            "config": {
                "results_subdir": args.results_subdir,
                "layer": args.layer,
                "tuned_threshold_version": args.tuned_threshold_version,
                "short_sentence_mode": args.short_sentence_mode,
                "min_sentence_words": args.min_sentence_words,
            },
            "per_behavior": per_behavior_rows,
            "total": agg_row,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
