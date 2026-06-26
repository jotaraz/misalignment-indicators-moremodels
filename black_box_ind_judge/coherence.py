"""Coherence analysis between the misalignment judge and the indicator judge.

Treats the misalignment judge's `misaligned_turns` (per-rollout list of turns
flagged as misaligned) as the reference, and the indicator judge's
`indicators_turns` (per-rollout list of turns with at least one present
indicator) as the predictor.

Reports three views:
  * Turn-level: does "any indicator fired on turn X" agree with "turn X is
    misaligned"? Full P/R/F1.
  * Transcript-level: does "any turn in the rollout has an indicator" agree
    with "any turn is misaligned"? Full P/R/F1 across rollouts.
  * Per-indicator turn-level: for each indicator name, precision (of its
    firings that land on misaligned turns) and coverage (share of all
    misaligned turns where this indicator fires). Per-indicator recall is
    NOT reported because each indicator is only meant to cover a sub-pattern
    of misalignment — a miss doesn't mean the indicator is wrong.

Usage:
    python -m black_box_ind_judge.coherence FILE [FILE ...]
    python -m black_box_ind_judge.coherence --json-out report.json FILE ...

Each FILE is a `_misalignment_turns.json` produced by the combined pipeline
(misalignment judge + indicator judge merged into the same file).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r, _f1(p, r)


def analyze_file(path: Path) -> dict[str, Any]:
    """Compute turn-, transcript-, and per-indicator coherence for one file."""
    data = json.load(open(path))

    per_indicator_tp: dict[str, int] = defaultdict(int)
    per_indicator_fp: dict[str, int] = defaultdict(int)
    per_indicator_positive: dict[str, int] = defaultdict(int)

    any_tp = any_fp = any_fn = 0
    total_misalign_turns = 0
    total_indicator_turns = 0

    trans_tp = trans_fp = trans_fn = trans_tn = 0

    for entry in data.get("evaluated_rollouts", []):
        misalign_set: set[int] = set(
            t["turn_index"]
            for t in entry.get("misaligned_turns", [])
            if "turn_index" in t
        )

        per_indicator_turns: dict[str, set[int]] = defaultdict(set)
        for it in entry.get("indicators_turns", []) or []:
            t_idx = it.get("turn_index")
            if t_idx is None:
                continue
            for ind in it.get("present_indicators", []):
                name = ind.get("indicator_name")
                if name:
                    per_indicator_turns[name].add(t_idx)

        indicator_set_any: set[int] = (
            set().union(*per_indicator_turns.values())
            if per_indicator_turns
            else set()
        )

        total_misalign_turns += len(misalign_set)
        total_indicator_turns += len(indicator_set_any)

        any_tp += len(misalign_set & indicator_set_any)
        any_fp += len(indicator_set_any - misalign_set)
        any_fn += len(misalign_set - indicator_set_any)

        for name, tset in per_indicator_turns.items():
            per_indicator_positive[name] += len(tset)
            per_indicator_tp[name] += len(tset & misalign_set)
            per_indicator_fp[name] += len(tset - misalign_set)

        has_misalign = len(misalign_set) > 0
        has_indicator = len(indicator_set_any) > 0
        if has_misalign and has_indicator:
            trans_tp += 1
        elif not has_misalign and has_indicator:
            trans_fp += 1
        elif has_misalign and not has_indicator:
            trans_fn += 1
        else:
            trans_tn += 1

    any_p, any_r, any_f1 = _prf(any_tp, any_fp, any_fn)
    trans_p, trans_r, trans_f1 = _prf(trans_tp, trans_fp, trans_fn)

    per_indicator: dict[str, dict[str, Any]] = {}
    for name in sorted(per_indicator_positive):
        tp = per_indicator_tp[name]
        fp = per_indicator_fp[name]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        coverage = tp / total_misalign_turns if total_misalign_turns > 0 else 0.0
        per_indicator[name] = {
            "tp": tp,
            "fp": fp,
            "positive_turns": per_indicator_positive[name],
            "precision": precision,
            "coverage": coverage,
        }

    return {
        "file": str(path),
        "n_rollouts": len(data.get("evaluated_rollouts", [])),
        "turn_level_any": {
            "tp": any_tp,
            "fp": any_fp,
            "fn": any_fn,
            "precision": any_p,
            "recall": any_r,
            "f1": any_f1,
            "total_misalign_turns": total_misalign_turns,
            "total_indicator_turns": total_indicator_turns,
        },
        "transcript_level": {
            "tp": trans_tp,
            "fp": trans_fp,
            "fn": trans_fn,
            "tn": trans_tn,
            "precision": trans_p,
            "recall": trans_r,
            "f1": trans_f1,
        },
        "per_indicator": per_indicator,
    }


def aggregate(results: list[dict]) -> dict[str, Any]:
    """Sum counts across files, then recompute derived metrics."""
    agg: dict[str, Any] = {
        "files": [r["file"] for r in results],
        "n_rollouts": sum(r["n_rollouts"] for r in results),
        "turn_level_any": {
            k: 0 for k in ["tp", "fp", "fn", "total_misalign_turns", "total_indicator_turns"]
        },
        "transcript_level": {k: 0 for k in ["tp", "fp", "fn", "tn"]},
        "per_indicator": defaultdict(lambda: {"tp": 0, "fp": 0, "positive_turns": 0}),
    }

    for r in results:
        for k in agg["turn_level_any"]:
            agg["turn_level_any"][k] += r["turn_level_any"][k]
        for k in agg["transcript_level"]:
            agg["transcript_level"][k] += r["transcript_level"][k]
        for name, stats in r["per_indicator"].items():
            for k in ["tp", "fp", "positive_turns"]:
                agg["per_indicator"][name][k] += stats[k]

    tp, fp, fn = agg["turn_level_any"]["tp"], agg["turn_level_any"]["fp"], agg["turn_level_any"]["fn"]
    p, rr, f = _prf(tp, fp, fn)
    agg["turn_level_any"]["precision"] = p
    agg["turn_level_any"]["recall"] = rr
    agg["turn_level_any"]["f1"] = f

    tp, fp, fn = agg["transcript_level"]["tp"], agg["transcript_level"]["fp"], agg["transcript_level"]["fn"]
    p, rr, f = _prf(tp, fp, fn)
    agg["transcript_level"]["precision"] = p
    agg["transcript_level"]["recall"] = rr
    agg["transcript_level"]["f1"] = f

    total_misalign = agg["turn_level_any"]["total_misalign_turns"]
    for name, stats in agg["per_indicator"].items():
        tpn, fpn = stats["tp"], stats["fp"]
        stats["precision"] = tpn / (tpn + fpn) if (tpn + fpn) > 0 else 0.0
        stats["coverage"] = tpn / total_misalign if total_misalign > 0 else 0.0
    agg["per_indicator"] = dict(agg["per_indicator"])
    return agg


def format_report(agg: dict, per_file_results: list[dict] | None = None) -> str:
    """Human-readable markdown-style report."""
    lines: list[str] = []
    lines.append(
        f"\n=== Aggregate across {len(agg['files'])} file(s), "
        f"{agg['n_rollouts']} rollouts ===\n"
    )

    t = agg["turn_level_any"]
    lines.append("## Turn-level (any-indicator predictor vs misaligned_turns reference)")
    lines.append(
        f"  misaligned turns: {t['total_misalign_turns']}   "
        f"indicator-flagged turns: {t['total_indicator_turns']}"
    )
    lines.append(f"  TP={t['tp']}  FP={t['fp']}  FN={t['fn']}")
    lines.append(
        f"  precision={t['precision']:.3f}  recall={t['recall']:.3f}  F1={t['f1']:.3f}"
    )

    t = agg["transcript_level"]
    lines.append("\n## Transcript-level (rollout has ANY flagged turn)")
    lines.append(
        f"  TP={t['tp']}  FP={t['fp']}  FN={t['fn']}  TN={t['tn']}"
    )
    lines.append(
        f"  precision={t['precision']:.3f}  recall={t['recall']:.3f}  F1={t['f1']:.3f}"
    )

    lines.append("\n## Per-indicator turn-level (precision + coverage; per-indicator recall not reported)")
    header = (
        f"  {'indicator':<45} {'+turns':>7} {'TP':>5} {'FP':>5} "
        f"{'prec':>6} {'cov':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, s in sorted(agg["per_indicator"].items(), key=lambda x: -x[1]["positive_turns"]):
        lines.append(
            f"  {name[:45]:<45} {s['positive_turns']:>7} "
            f"{s['tp']:>5} {s['fp']:>5} "
            f"{s['precision']:>6.3f} {s['coverage']:>6.3f}"
        )

    if per_file_results:
        lines.append("\n## Per-file summary")
        header = f"  {'file':<60} {'rollouts':>8} {'turn_F1':>8} {'trans_F1':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for r in per_file_results:
            name = Path(r["file"]).parent.name
            lines.append(
                f"  {name[:60]:<60} {r['n_rollouts']:>8} "
                f"{r['turn_level_any']['f1']:>8.3f} {r['transcript_level']['f1']:>8.3f}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coherence between misalignment and indicator judges",
    )
    parser.add_argument("files", nargs="+", help="_misalignment_turns.json files")
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to save the full report as JSON",
    )
    args = parser.parse_args()

    paths = [Path(f) for f in args.files]
    results = [analyze_file(p) for p in paths]
    agg = aggregate(results)
    print(format_report(agg, per_file_results=results))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"aggregate": agg, "per_file": results}, f, indent=2)
        print(f"\nJSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
