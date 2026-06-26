"""
Evaluate probes against indicator-level GT from the combined misalignment turn judge.

Ground-truth source: ``rollout_misalignment_turns.json``'s ``indicators_turns``
field (populated by the misalignment judge when ``indicator_judge`` config is
enabled).  Each rollout / turn can list multiple present indicators; each
indicator carries a list of evidence spans.

Per probe (mapped to one indicator via ``_slug(indicator.name)``):
  * Turn-level positives: turns where this indicator appears in
    ``present_indicators``.
  * Turn-level negatives: turns that have NO indicator at all AND are not
    listed as misaligned (including all turns of fully-clean transcripts).
  * Sentence-level positives: sentences overlapping this indicator's spans.
  * Sentence-level negatives: sentences in turn-level-negative turns.
  * Turns that have some other indicator OR are misaligned without this
    indicator are excluded (grey zone).

Metrics: per-probe AUROC and PR-AUC at both turn and sentence granularities.

Usage:
    python -m probe_eval.indicator_detection_gt --results-subdir v4_v2_6_combined_v2_span
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from probe_eval.common import (
    NON_INDICATOR_CONCEPTS,
    RESULTS_DIR,
    add_behavior_filter_args,
    filter_behaviors,
    get_concept_from_experiment_folder,
    get_n_turns,
    get_rollout_var_rep,
    nan_to_none as _nan_to_none,
    save_json as _save_json,
)
from probe_eval.metrics import per_turn_binary_auroc as _per_turn_binary_auroc_generic
from probe_eval.misalignment_ground_truth import (
    load_misalignment_ground_truth,
    load_unreliable_rollouts,
)
from probe_eval.sentence_scores import (
    build_sentence_scores_by_key,
    build_sentence_texts_by_key,
    load_per_sentence_scores,
    load_per_sentence_texts,
)


# ---------------------------------------------------------------------------
# Concept-slug helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.lower()).strip("_")
    return s


def build_slug_to_indicator_name() -> dict[str, str]:
    """Map probe-folder slug -> canonical indicator name."""
    from indicators.misalignment_indicators_v2_6 import MISALIGNMENT_INDICATORS_V2
    return {_slug(ind.name): ind.name for ind in MISALIGNMENT_INDICATORS_V2}


# ---------------------------------------------------------------------------
# GT loading: indicators_turns
# ---------------------------------------------------------------------------

def load_indicators_turns(
    gt_path: Path,
    exclude_ambiguous: bool = True,
) -> dict[tuple[int, int], dict[int, dict[str, list[str]]]]:
    """Load per-turn indicator-name → spans from rollout_misalignment_turns.json.

    Returns ``{(var, rep): {turn_index: {indicator_name: [span_text, ...]}}}``.
    """
    with open(gt_path) as f:
        data = json.load(f)

    ambiguous: set[tuple[int, int]] = set()
    if exclude_ambiguous:
        ambiguous = load_unreliable_rollouts(gt_path)

    result: dict[tuple[int, int], dict[int, dict[str, list[str]]]] = {}
    for rollout in data["evaluated_rollouts"]:
        key = (rollout["variation_number"], rollout["repetition_number"])
        if key in ambiguous:
            continue
        per_turn: dict[int, dict[str, list[str]]] = {}
        for turn_entry in rollout.get("indicators_turns", []):
            turn_idx = turn_entry["turn_index"]
            present = turn_entry.get("present_indicators", []) or []
            ind_map: dict[str, list[str]] = {}
            for ind in present:
                name = ind.get("indicator_name")
                if not name:
                    continue
                spans = ind.get("spans", []) or []
                # Normalize dict-style spans (safety; evidence sometimes dicts)
                span_texts = [s.get("text", "") if isinstance(s, dict) else str(s)
                              for s in spans]
                ind_map[name] = [s for s in span_texts if s]
            per_turn[turn_idx] = ind_map
        result[key] = per_turn
    return result


# ---------------------------------------------------------------------------
# Point building (turn + sentence level)
# ---------------------------------------------------------------------------

def build_turn_points(
    probe_data: dict,
    indicators_gt: dict[tuple[int, int], dict[int, dict[str, list[str]]]],
    misalignment_gt: dict[tuple[int, int], set[int]],
    n_turns_map: dict[tuple[int, int], int],
    var_rep_map: dict[int, tuple[int, int]],
    target_indicator_name: str,
    sent_by_key: dict[tuple, list[float]] | None = None,
) -> list[dict]:
    """Build per-turn points for one probe (scored against one indicator).

    Each entry: ``{var, rep, turn, probe_score, gt_pos}``.

    *gt_pos* = True  → turn has the target indicator.
    *gt_pos* = False → turn has no indicator AND is not in misaligned_turns.
    Other turns (grey zone) are excluded.

    ``sent_by_key`` (optional): when provided, per-turn ``probe_score`` is
    max of sentence scores (consistent with the rest of the pipeline);
    otherwise falls back to ``per_turn_scores`` or global ``probe_score``.
    """
    points: list[dict] = []
    for entry in probe_data.get("per_rollout", []):
        idx = entry["rollout_index"]
        if idx not in var_rep_map:
            continue
        var, rep = var_rep_map[idx]
        key = (var, rep)
        if key not in n_turns_map:
            continue
        n_turns = n_turns_map[key]
        turn_indicators = indicators_gt.get(key, {})
        misaligned_turns = misalignment_gt.get(key, set())

        per_turn_scores = entry.get("per_turn_scores")
        score_by_turn = (
            {s["turn"]: s["score"] for s in per_turn_scores}
            if per_turn_scores else None
        )

        for turn in range(1, n_turns + 1):
            inds = turn_indicators.get(turn, {})
            has_target = target_indicator_name in inds
            any_indicator = bool(inds)
            is_misaligned = turn in misaligned_turns

            if has_target:
                gt_pos = True
            elif (not any_indicator) and (not is_misaligned):
                gt_pos = False
            else:
                continue  # grey zone: other indicator / misaligned w/o this ind

            # Determine per-turn score
            if sent_by_key is not None:
                ss = sent_by_key.get((var, rep, turn))
                if ss:
                    score = max(ss)
                else:
                    score = float("nan")
            elif score_by_turn is not None:
                if turn not in score_by_turn:
                    continue
                score = score_by_turn[turn]
            else:
                score = entry.get("probe_score", float("nan"))

            points.append({
                "var": var,
                "rep": rep,
                "turn": turn,
                "probe_score": score,
                "gt_pos": gt_pos,
                "any_indicator": any_indicator,
                "is_misaligned": is_misaligned,
            })
    return points


def _find_char_ranges(full_text: str, spans: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, str):
            continue
        idx = full_text.find(span)
        if idx >= 0:
            ranges.append((idx, idx + len(span)))
            continue
        stripped = span.strip()
        if stripped:
            idx = full_text.find(stripped)
            if idx >= 0:
                ranges.append((idx, idx + len(stripped)))
    return ranges


def build_sentence_points(
    sent_scores_by_key: dict[tuple, list[float]],
    sent_texts_by_key: dict[tuple, list[tuple[str, int, int]]],
    indicators_gt: dict[tuple[int, int], dict[int, dict[str, list[str]]]],
    misalignment_gt: dict[tuple[int, int], set[int]],
    target_indicator_name: str,
) -> list[dict]:
    """Build per-sentence points for one probe.

    Positive sentence = sentence in a turn with the target indicator AND
    overlapping that indicator's evidence spans.
    Negative sentence = every sentence inside a turn-level negative turn
    (no indicator AND not misaligned).
    Sentences in grey-zone turns are excluded entirely.
    """
    points: list[dict] = []
    for key, scores in sent_scores_by_key.items():
        var, rep, turn = key
        texts = sent_texts_by_key.get(key, [])
        if not texts or len(texts) != len(scores):
            continue
        inds = indicators_gt.get((var, rep), {}).get(turn, {})
        has_target = target_indicator_name in inds
        any_indicator = bool(inds)
        is_misaligned = turn in misalignment_gt.get((var, rep), set())

        if has_target:
            # Positive turn: find overlap with spans
            full_text = "".join(t for t, _, _ in texts)
            target_spans = inds.get(target_indicator_name, [])
            ranges = _find_char_ranges(full_text, target_spans)
            for i, (score, (_, cs, ce)) in enumerate(zip(scores, texts)):
                overlap = any(cs < r_end and ce > r_start for r_start, r_end in ranges)
                if not overlap:
                    continue  # negative sentence inside a positive turn → excluded
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    continue
                points.append({
                    "var": var, "rep": rep, "turn": turn, "sent_idx": i,
                    "probe_score": float(score),
                    "gt_pos": True,
                })
        elif (not any_indicator) and (not is_misaligned):
            # Negative turn — all sentences negative
            for i, score in enumerate(scores):
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    continue
                points.append({
                    "var": var, "rep": rep, "turn": turn, "sent_idx": i,
                    "probe_score": float(score),
                    "gt_pos": False,
                })
        # else: grey zone → skip entirely
    return points


# ---------------------------------------------------------------------------
# Discovery (same rule as misalignment GT: require rollout_misalignment_turns.json)
# ---------------------------------------------------------------------------

def discover_results(
    search_root: Path,
    slug_to_indicator: dict[str, str],
    exclude_concepts: set[str] | None = None,
) -> list[dict]:
    exclude = exclude_concepts or set()
    found: list[dict] = []
    gt_exists_cache: dict[str, bool] = {}

    for rj in sorted(search_root.rglob("results.json")):
        with open(rj) as f:
            data = json.load(f)
        rollout_dir = data.get("rollout_dir", "")
        if not rollout_dir:
            continue

        if rollout_dir not in gt_exists_cache:
            gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
            gt_exists_cache[rollout_dir] = gt_path.exists()
        if not gt_exists_cache[rollout_dir]:
            continue

        behavior = rj.parent.name
        concept = get_concept_from_experiment_folder(data.get("experiment_folder", ""))
        if concept is None or concept in exclude:
            continue
        if concept not in slug_to_indicator:
            # Probe concept isn't a known indicator — skip
            continue

        found.append({
            "probe_id": str(rj.parent.relative_to(search_root)),
            "behavior": behavior,
            "concept": concept,
            "indicator_name": slug_to_indicator[concept],
            "experiment_folder": data.get("experiment_folder", ""),
            "rollout_dir": rollout_dir,
            "result_path": rj,
            "data": data,
        })
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auroc(points: list[dict]) -> dict:
    return _per_turn_binary_auroc_generic(points, gt_field="gt_pos")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate probes against indicator GT (from misalignment judge)."
    )
    parser.add_argument("--results-subdir", required=True)
    parser.add_argument(
        "--include-preconditions", action="store_true",
        help="Include precondition/behavioral concept probes (excluded by default).",
    )
    parser.add_argument(
        "--include-ambiguous", action="store_true",
        help="Include rollouts flagged ambiguous (excluded by default).",
    )
    parser.add_argument(
        "--short-sentence-mode", default="merge", choices=["merge", "discard"],
    )
    parser.add_argument("--min-sentence-words", type=int, default=5)
    add_behavior_filter_args(parser)
    args = parser.parse_args()

    search_root = RESULTS_DIR / args.results_subdir
    if not search_root.exists():
        print(f"Results directory not found: {search_root}")
        return

    slug_to_indicator = build_slug_to_indicator_name()
    exclude = None if args.include_preconditions else set(NON_INDICATOR_CONCEPTS)
    results = discover_results(search_root, slug_to_indicator, exclude_concepts=exclude)
    if not results:
        print("No probe results with indicator GT found.")
        return

    # Group by behavior
    by_behavior: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_behavior[r["behavior"]].append(r)

    include_pats = args.include_behaviors.split(",") if args.include_behaviors else None
    exclude_pats = args.exclude_behaviors.split(",") if args.exclude_behaviors else None
    if include_pats or exclude_pats:
        by_behavior = filter_behaviors(by_behavior, include_pats, exclude_pats)
        if not by_behavior:
            print("No behaviors after filtering.")
            return

    exclude_ambiguous = not args.include_ambiguous

    # Caches
    indicators_cache: dict[str, dict] = {}
    misalignment_cache: dict[str, dict] = {}
    rollouts_full_cache: dict[str, list[dict]] = {}
    sent_text_cache: dict[str, dict[tuple, list[tuple[str, int, int]]]] = {}

    # Aggregates: {(layer, concept): [points_turn], [points_sent]}
    turn_points_agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sent_points_agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    # Also a per-concept behavior breakdown if useful
    per_behavior_counts: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for behavior, behavior_results in sorted(by_behavior.items()):
        rollout_dir = behavior_results[0]["rollout_dir"]
        gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
        if rollout_dir not in indicators_cache:
            indicators_cache[rollout_dir] = load_indicators_turns(
                gt_path, exclude_ambiguous=exclude_ambiguous,
            )
            misalignment_cache[rollout_dir] = load_misalignment_ground_truth(
                gt_path, exclude_ambiguous=exclude_ambiguous,
            )
        indicators_gt = indicators_cache[rollout_dir]
        misalignment_gt = misalignment_cache[rollout_dir]
        var_rep_map = get_rollout_var_rep(rollout_dir)
        n_turns_map = get_n_turns(rollout_dir)

        # Summary: how many turns per label class in this behavior
        n_turns_pos_any = 0
        n_turns_all_clean = 0
        for (var, rep), per_turn in indicators_gt.items():
            n_turns_pos_any += sum(1 for t, inds in per_turn.items() if inds)
        for (var, rep), misaligned in misalignment_gt.items():
            all_turns = n_turns_map.get((var, rep), 0)
            per_t = indicators_gt.get((var, rep), {})
            for t in range(1, all_turns + 1):
                if not per_t.get(t) and t not in misaligned:
                    n_turns_all_clean += 1

        print(f"\n{'='*70}")
        print(f"Behavior: {behavior}")
        print(f"  Rollouts: {len(misalignment_gt)}, "
              f"turns w/ any indicator: {n_turns_pos_any}, "
              f"turns clean (neg pool): {n_turns_all_clean}")
        print(f"{'='*70}")

        for r in behavior_results:
            concept = r["concept"]
            target_indicator_name = r["indicator_name"]
            layer = r["result_path"].parent.parent.name

            # Per-sentence scoring for this probe
            ts_path = r["result_path"].parent / "token_scores.json"
            sent_scores_by_rollout = load_per_sentence_scores(
                ts_path, rollout_dir, rollouts_full_cache,
                short_sentence_mode=args.short_sentence_mode,
                min_words=args.min_sentence_words,
            )
            sent_scores_by_key: dict[tuple, list[float]] = {}
            if sent_scores_by_rollout is not None:
                sent_scores_by_key = build_sentence_scores_by_key(
                    sent_scores_by_rollout, var_rep_map,
                )

            # Sentence texts: cache per (behavior, rollout_dir) — any probe works
            if rollout_dir not in sent_text_cache:
                text_data = load_per_sentence_texts(
                    ts_path, rollout_dir, rollouts_full_cache,
                )
                if text_data is not None:
                    sent_text_cache[rollout_dir] = build_sentence_texts_by_key(
                        text_data, var_rep_map,
                    )
                else:
                    sent_text_cache[rollout_dir] = {}

            # Turn-level points (sentence-max scores)
            turn_points = build_turn_points(
                r["data"], indicators_gt, misalignment_gt, n_turns_map,
                var_rep_map, target_indicator_name,
                sent_by_key=sent_scores_by_key or None,
            )

            # Sentence-level points
            sent_points = build_sentence_points(
                sent_scores_by_key,
                sent_text_cache.get(rollout_dir, {}),
                indicators_gt, misalignment_gt, target_indicator_name,
            )

            # Tag var so we can combine behaviors without key collisions
            for p in turn_points:
                p["var"] = f"{behavior}__{p['var']}"
            for p in sent_points:
                p["var"] = f"{behavior}__{p['var']}"

            turn_points_agg[(layer, concept)].extend(turn_points)
            sent_points_agg[(layer, concept)].extend(sent_points)

            n_pos_t = sum(1 for p in turn_points if p["gt_pos"])
            n_neg_t = sum(1 for p in turn_points if not p["gt_pos"])
            n_pos_s = sum(1 for p in sent_points if p["gt_pos"])
            n_neg_s = sum(1 for p in sent_points if not p["gt_pos"])
            per_behavior_counts[(layer, concept)][behavior] = {
                "turn_pos": n_pos_t, "turn_neg": n_neg_t,
                "sent_pos": n_pos_s, "sent_neg": n_neg_s,
            }

    # Compute per-probe metrics at each layer
    per_layer: dict[str, dict] = {}
    for (layer, concept), turn_pts in sorted(turn_points_agg.items()):
        sent_pts = sent_points_agg.get((layer, concept), [])
        turn_metric = _auroc(turn_pts) if turn_pts else {}
        sent_metric = _auroc(sent_pts) if sent_pts else {}
        layer_data = per_layer.setdefault(layer, {"per_probe": {}})
        layer_data["per_probe"][concept] = {
            "indicator_name": build_slug_to_indicator_name().get(concept, concept),
            "turn_level": {
                "auroc": _nan_to_none(turn_metric.get("auroc", float("nan"))),
                "pr_auc": _nan_to_none(turn_metric.get("pr_auc", float("nan"))),
                "n": turn_metric.get("n", 0),
                "n_pos": turn_metric.get("n_pos", 0),
            },
            "sentence_level": {
                "auroc": _nan_to_none(sent_metric.get("auroc", float("nan"))),
                "pr_auc": _nan_to_none(sent_metric.get("pr_auc", float("nan"))),
                "n": sent_metric.get("n", 0),
                "n_pos": sent_metric.get("n_pos", 0),
            },
            "per_behavior_counts": per_behavior_counts.get((layer, concept), {}),
        }

    # Print summary
    for layer in sorted(per_layer):
        print(f"\n{'='*100}")
        print(f"Layer {layer}: per-probe indicator-detection AUROC / PR-AUC")
        print(f"{'='*100}")
        print(f"  {'Probe (concept)':<45} "
              f"{'Turn AUROC':>10} {'Turn PR':>8} {'Turn n(+/tot)':>15}  "
              f"{'Sent AUROC':>10} {'Sent PR':>8} {'Sent n(+/tot)':>15}")
        print(f"  {'-'*130}")
        for concept, info in sorted(per_layer[layer]["per_probe"].items()):
            t = info["turn_level"]
            s = info["sentence_level"]
            def _f(v):
                return f"{v:.3f}" if v is not None else "  N/A"
            print(
                f"  {concept:<45} "
                f"{_f(t['auroc']):>10} {_f(t['pr_auc']):>8} "
                f"{t['n_pos']:>5}/{t['n']:<9}  "
                f"{_f(s['auroc']):>10} {_f(s['pr_auc']):>8} "
                f"{s['n_pos']:>5}/{s['n']:<9}"
            )

    # Save
    out = {"per_layer": per_layer}
    out_path = search_root / f"indicator_detection_gt_summary{args.output_suffix}.json"
    _save_json(out_path, out)


if __name__ == "__main__":
    main()
