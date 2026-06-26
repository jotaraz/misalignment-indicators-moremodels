"""Compare two versions of the indicator-turn judge.

For each rollout (matched by variation_number + repetition_number), compares
`indicators_turns` between a NEW run and an OLD/backup run of the same
`_misalignment_turns.json` file. Reports:

  * Turn-level ANY-indicator agreement (did at least one indicator fire?).
  * Per-indicator turn-level: for each indicator, how many turns flipped
    (both, only-new, only-old) and where the flip-over most (on-target vs
    off-target behaviors if `relevant_behaviors` available).

Usage:
    python -m black_box_ind_judge.compare_indicator_versions \
        <new.json>:<old.json> [<new.json>:<old.json> ...]

    # Or let the tool auto-resolve the old path from a suffix pattern:
    python -m black_box_ind_judge.compare_indicator_versions \
        --old-suffix _no_prior \
        <new.json> [<new.json> ...]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _key(r: dict) -> tuple:
    return (r.get("variation_number"), r.get("repetition_number"))


def _load_relevant_behaviors(indicator_set: str = "v2_6") -> dict[str, set[str]]:
    """Return indicator_name -> set of relevant_behaviors (empty if not found)."""
    try:
        import sys
        from pathlib import Path as _P
        repo_root = _P(__file__).parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        module_name = f"indicators.misalignment_indicators_{indicator_set}"
        module = __import__(module_name, fromlist=["MISALIGNMENT_INDICATORS_V2"])
        return {
            ind.name: set(getattr(ind, "relevant_behaviors", []))
            for ind in module.MISALIGNMENT_INDICATORS_V2
        }
    except Exception:
        return {}


def compare_pair(new_path: Path, old_path: Path) -> dict[str, Any]:
    new_data = json.load(open(new_path))
    old_data = json.load(open(old_path))
    old_map = {_key(r): r for r in old_data.get("evaluated_rollouts", [])}

    any_both = any_only_new = any_only_old = 0
    total_new_any = total_old_any = 0

    trans_both = trans_only_new = trans_only_old = trans_both_clean = 0

    per_ind_both: dict[str, int] = defaultdict(int)
    per_ind_only_new: dict[str, int] = defaultdict(int)
    per_ind_only_old: dict[str, int] = defaultdict(int)

    unmatched = 0

    for r_new in new_data.get("evaluated_rollouts", []):
        r_old = old_map.get(_key(r_new))
        if r_old is None:
            unmatched += 1
            continue

        # Build per-turn per-indicator sets for each side
        def _extract(entry: dict) -> tuple[set[int], dict[str, set[int]]]:
            any_set: set[int] = set()
            per_ind: dict[str, set[int]] = defaultdict(set)
            for it in entry.get("indicators_turns", []) or []:
                t_idx = it.get("turn_index")
                if t_idx is None:
                    continue
                for ind in it.get("present_indicators", []) or []:
                    name = ind.get("indicator_name")
                    if name:
                        per_ind[name].add(t_idx)
                        any_set.add(t_idx)
            return any_set, per_ind

        new_any, new_per = _extract(r_new)
        old_any, old_per = _extract(r_old)

        total_new_any += len(new_any)
        total_old_any += len(old_any)
        any_both += len(new_any & old_any)
        any_only_new += len(new_any - old_any)
        any_only_old += len(old_any - new_any)

        has_new_trans = len(new_any) > 0
        has_old_trans = len(old_any) > 0
        if has_new_trans and has_old_trans:
            trans_both += 1
        elif has_new_trans:
            trans_only_new += 1
        elif has_old_trans:
            trans_only_old += 1
        else:
            trans_both_clean += 1

        all_names = set(new_per) | set(old_per)
        for name in all_names:
            ns = new_per.get(name, set())
            os = old_per.get(name, set())
            per_ind_both[name] += len(ns & os)
            per_ind_only_new[name] += len(ns - os)
            per_ind_only_old[name] += len(os - ns)

    union = any_both + any_only_new + any_only_old
    jaccard = any_both / union if union > 0 else 0.0

    trans_union = trans_both + trans_only_new + trans_only_old
    trans_jaccard = trans_both / trans_union if trans_union > 0 else 0.0

    # Extract the test_behavior from metadata if present
    new_behavior = (
        new_data.get("metadata", {}).get("indicator_judge", {}).get("test_behavior", "unknown")
    )
    old_behavior = (
        old_data.get("metadata", {}).get("indicator_judge", {}).get("test_behavior", "unknown")
    )

    return {
        "file_new": str(new_path),
        "file_old": str(old_path),
        "test_behavior": new_behavior,
        "old_test_behavior": old_behavior,
        "n_rollouts": len(new_data.get("evaluated_rollouts", [])),
        "unmatched_rollouts": unmatched,
        "any_indicator": {
            "both": any_both,
            "only_new": any_only_new,
            "only_old": any_only_old,
            "total_new": total_new_any,
            "total_old": total_old_any,
            "jaccard": jaccard,
        },
        "transcript_level": {
            "both_flag": trans_both,
            "only_new_flag": trans_only_new,
            "only_old_flag": trans_only_old,
            "both_clean": trans_both_clean,
            "jaccard": trans_jaccard,
        },
        "per_indicator": {
            name: {
                "both": per_ind_both[name],
                "only_new": per_ind_only_new[name],
                "only_old": per_ind_only_old[name],
            }
            for name in sorted(set(per_ind_both) | set(per_ind_only_new) | set(per_ind_only_old))
        },
    }


def aggregate(results: list[dict]) -> dict[str, Any]:
    agg: dict[str, Any] = {
        "n_rollouts": sum(r["n_rollouts"] for r in results),
        "unmatched_rollouts": sum(r["unmatched_rollouts"] for r in results),
        "any_indicator": {
            k: 0 for k in ["both", "only_new", "only_old", "total_new", "total_old"]
        },
        "transcript_level": {
            k: 0 for k in ["both_flag", "only_new_flag", "only_old_flag", "both_clean"]
        },
        "per_indicator": defaultdict(lambda: {"both": 0, "only_new": 0, "only_old": 0}),
    }
    for r in results:
        for k in agg["any_indicator"]:
            agg["any_indicator"][k] += r["any_indicator"][k]
        for k in agg["transcript_level"]:
            agg["transcript_level"][k] += r["transcript_level"][k]
        for name, stats in r["per_indicator"].items():
            for k in ["both", "only_new", "only_old"]:
                agg["per_indicator"][name][k] += stats[k]
    a = agg["any_indicator"]
    union = a["both"] + a["only_new"] + a["only_old"]
    a["jaccard"] = a["both"] / union if union > 0 else 0.0
    tr = agg["transcript_level"]
    trans_union = tr["both_flag"] + tr["only_new_flag"] + tr["only_old_flag"]
    tr["jaccard"] = tr["both_flag"] / trans_union if trans_union > 0 else 0.0
    agg["per_indicator"] = dict(agg["per_indicator"])
    return agg


def format_report(agg: dict, per_file: list[dict], rel: dict[str, set[str]]) -> str:
    lines: list[str] = []
    lines.append(
        f"\n=== Indicator judge: NEW vs OLD — {len(per_file)} file pair(s), "
        f"{agg['n_rollouts']} rollouts ==="
    )
    if agg["unmatched_rollouts"]:
        lines.append(f"(unmatched rollouts: {agg['unmatched_rollouts']})")

    a = agg["any_indicator"]
    lines.append("\n## Any-indicator turn-level agreement (did any indicator fire on this turn?)")
    lines.append(
        f"  both: {a['both']}   only NEW: {a['only_new']}   only OLD: {a['only_old']}"
    )
    lines.append(
        f"  total NEW flags: {a['total_new']}   total OLD flags: {a['total_old']}"
    )
    lines.append(f"  Jaccard: {a['jaccard']:.3f}")

    tr = agg["transcript_level"]
    lines.append("\n## Any-indicator transcript-level agreement (did any indicator fire in this rollout?)")
    lines.append(
        f"  both positive: {tr['both_flag']}   "
        f"only NEW positive: {tr['only_new_flag']}   "
        f"only OLD positive: {tr['only_old_flag']}   "
        f"both clean: {tr['both_clean']}"
    )
    lines.append(f"  Jaccard: {tr['jaccard']:.3f}")

    lines.append("\n## Per-indicator flip counts (how the new prior changed firings)")
    header = (
        f"  {'indicator':<45} {'both':>6} {'+new':>6} {'-old':>6} "
        f"{'net':>6} {'total_old':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name, s in sorted(
        agg["per_indicator"].items(),
        key=lambda x: -(x[1]["both"] + x[1]["only_old"]),
    ):
        total_old_here = s["both"] + s["only_old"]
        net = s["only_new"] - s["only_old"]
        lines.append(
            f"  {name[:45]:<45} {s['both']:>6} {s['only_new']:>6} {s['only_old']:>6} "
            f"{net:>+6d} {total_old_here:>10}"
        )

    lines.append("\n## Per-file summary")
    header = (
        f"  {'file':<55} {'behavior':<20} {'turn_jacc':>10} "
        f"{'trans_jacc':>11} {'+new':>6} {'-old':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in per_file:
        name = Path(r["file_new"]).parent.name
        a = r["any_indicator"]
        t = r["transcript_level"]
        lines.append(
            f"  {name[:55]:<55} {r['test_behavior'][:20]:<20} "
            f"{a['jaccard']:>10.3f} {t['jaccard']:>11.3f} "
            f"{a['only_new']:>6} {a['only_old']:>6}"
        )
    return "\n".join(lines)


def _resolve_pair(spec: str, old_suffix: str) -> tuple[Path, Path]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return Path(a), Path(b)
    new_path = Path(spec)
    old_path = new_path.with_name(new_path.stem + old_suffix + new_path.suffix)
    return new_path, old_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two versions of the indicator-turn judge",
    )
    parser.add_argument(
        "pairs",
        nargs="+",
        help="Either <new.json> (with --old-suffix) or <new.json>:<old.json>.",
    )
    parser.add_argument(
        "--old-suffix", type=str, default="_no_prior",
        help="Suffix (before .json) to derive old path from new path.",
    )
    parser.add_argument(
        "--indicator-set", type=str, default="v2_6",
        help="Indicator set for relevant-behaviors lookup.",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to save the full report as JSON",
    )
    args = parser.parse_args()

    rel = _load_relevant_behaviors(args.indicator_set)

    results: list[dict] = []
    for spec in args.pairs:
        new_path, old_path = _resolve_pair(spec, args.old_suffix)
        if not new_path.exists():
            print(f"skip (new missing): {new_path}")
            continue
        if not old_path.exists():
            print(f"skip (old missing): {old_path}")
            continue
        results.append(compare_pair(new_path, old_path))

    if not results:
        print("no comparable pairs found")
        return

    agg = aggregate(results)
    print(format_report(agg, per_file=results, rel=rel))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"aggregate": agg, "per_file": results}, f, indent=2)
        print(f"\nJSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
