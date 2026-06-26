"""Compare two versions of the misalignment-turn judge.

For each rollout (matched by variation_number + repetition_number), reports
agreement at two levels:
  * Turn-level: set agreement on which turns are flagged misaligned
    (Jaccard + precision/recall of NEW vs. OLD treated as reference).
  * Transcript-level: did both versions flag ANY turn in the rollout?

Usage:
    # Pair a "_misalignment_turns.json" with its sibling "_v3.json" automatically.
    python -m black_box_ind_judge.compare_misalign_versions \
        bloom/bloom-results/*/rollout_misalignment_turns.json

    # Or pass explicit pairs with `new_path:old_path` separator.
    python -m black_box_ind_judge.compare_misalign_versions \
        path/new.json:path/old.json ...
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _key(r: dict) -> tuple:
    return (r.get("variation_number"), r.get("repetition_number"))


def compare_pair(new_path: Path, old_path: Path) -> dict[str, Any]:
    """Compare one new-vs-old judge file pair."""
    new_data = json.load(open(new_path))
    old_data = json.load(open(old_path))
    old_map = {_key(r): r for r in old_data.get("evaluated_rollouts", [])}

    both = only_new = only_old = 0
    total_new = total_old = 0
    trans_both = trans_only_new = trans_only_old = trans_both_clean = 0
    unmatched = 0

    for r_new in new_data.get("evaluated_rollouts", []):
        r_old = old_map.get(_key(r_new))
        if r_old is None:
            unmatched += 1
            continue

        new_set = set(
            t["turn_index"]
            for t in r_new.get("misaligned_turns", [])
            if "turn_index" in t
        )
        old_set = set(
            t["turn_index"]
            for t in r_old.get("misaligned_turns", [])
            if "turn_index" in t
        )

        total_new += len(new_set)
        total_old += len(old_set)
        both += len(new_set & old_set)
        only_new += len(new_set - old_set)
        only_old += len(old_set - new_set)

        has_new = len(new_set) > 0
        has_old = len(old_set) > 0
        if has_new and has_old:
            trans_both += 1
        elif has_new:
            trans_only_new += 1
        elif has_old:
            trans_only_old += 1
        else:
            trans_both_clean += 1

    union = both + only_new + only_old
    jaccard = both / union if union > 0 else 0.0
    # Treat OLD as reference; new's "precision" = share of new flags also in old.
    precision = both / (both + only_new) if (both + only_new) > 0 else 0.0
    recall = both / (both + only_old) if (both + only_old) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    trans_union = trans_both + trans_only_new + trans_only_old
    trans_jaccard = trans_both / trans_union if trans_union > 0 else 0.0

    return {
        "file_new": str(new_path),
        "file_old": str(old_path),
        "n_rollouts": len(new_data.get("evaluated_rollouts", [])),
        "unmatched_rollouts": unmatched,
        "turn_level": {
            "both": both,
            "only_new": only_new,
            "only_old": only_old,
            "total_new": total_new,
            "total_old": total_old,
            "jaccard": jaccard,
            "precision_new_vs_old": precision,
            "recall_new_vs_old": recall,
            "f1": f1,
        },
        "transcript_level": {
            "both_flag": trans_both,
            "only_new_flag": trans_only_new,
            "only_old_flag": trans_only_old,
            "both_clean": trans_both_clean,
            "jaccard": trans_jaccard,
        },
    }


def aggregate(results: list[dict]) -> dict[str, Any]:
    agg = {
        "n_rollouts": sum(r["n_rollouts"] for r in results),
        "unmatched_rollouts": sum(r["unmatched_rollouts"] for r in results),
        "turn_level": {
            k: 0 for k in ["both", "only_new", "only_old", "total_new", "total_old"]
        },
        "transcript_level": {
            k: 0 for k in ["both_flag", "only_new_flag", "only_old_flag", "both_clean"]
        },
    }
    for r in results:
        for k in agg["turn_level"]:
            agg["turn_level"][k] += r["turn_level"][k]
        for k in agg["transcript_level"]:
            agg["transcript_level"][k] += r["transcript_level"][k]

    t = agg["turn_level"]
    union = t["both"] + t["only_new"] + t["only_old"]
    t["jaccard"] = t["both"] / union if union > 0 else 0.0
    t["precision_new_vs_old"] = (
        t["both"] / (t["both"] + t["only_new"]) if (t["both"] + t["only_new"]) > 0 else 0.0
    )
    t["recall_new_vs_old"] = (
        t["both"] / (t["both"] + t["only_old"]) if (t["both"] + t["only_old"]) > 0 else 0.0
    )
    p, r = t["precision_new_vs_old"], t["recall_new_vs_old"]
    t["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    tr = agg["transcript_level"]
    trans_union = tr["both_flag"] + tr["only_new_flag"] + tr["only_old_flag"]
    tr["jaccard"] = tr["both_flag"] / trans_union if trans_union > 0 else 0.0

    return agg


def format_report(agg: dict, per_file: list[dict]) -> str:
    lines: list[str] = []
    lines.append(
        f"\n=== Aggregate across {len(per_file)} file pair(s), "
        f"{agg['n_rollouts']} rollouts ==="
    )
    if agg["unmatched_rollouts"]:
        lines.append(f"(unmatched rollouts: {agg['unmatched_rollouts']})")

    t = agg["turn_level"]
    lines.append("\n## Turn-level set agreement")
    lines.append(
        f"  both flagged (∩): {t['both']}   "
        f"only NEW: {t['only_new']}   only OLD: {t['only_old']}"
    )
    lines.append(
        f"  total NEW flags: {t['total_new']}   total OLD flags: {t['total_old']}"
    )
    lines.append(
        f"  Jaccard: {t['jaccard']:.3f}   "
        f"P(new|old): {t['precision_new_vs_old']:.3f}   "
        f"R(new|old): {t['recall_new_vs_old']:.3f}   F1: {t['f1']:.3f}"
    )

    tr = agg["transcript_level"]
    lines.append("\n## Transcript-level agreement (any flagged turn)")
    lines.append(
        f"  both positive: {tr['both_flag']}   "
        f"only NEW positive: {tr['only_new_flag']}   "
        f"only OLD positive: {tr['only_old_flag']}   "
        f"both clean: {tr['both_clean']}"
    )
    lines.append(f"  Jaccard: {tr['jaccard']:.3f}")

    lines.append("\n## Per-file")
    header = (
        f"  {'file':<50} {'rollouts':>8} {'jacc_turn':>10} "
        f"{'only_new':>9} {'only_old':>9} {'jacc_trans':>11}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in per_file:
        name = Path(r["file_new"]).parent.name
        t = r["turn_level"]
        tr = r["transcript_level"]
        lines.append(
            f"  {name[:50]:<50} {r['n_rollouts']:>8} "
            f"{t['jaccard']:>10.3f} {t['only_new']:>9} {t['only_old']:>9} "
            f"{tr['jaccard']:>11.3f}"
        )
    return "\n".join(lines)


def _resolve_pair(spec: str) -> tuple[Path, Path]:
    """Parse 'new:old' or 'new' (with v3 sibling auto-resolved)."""
    if ":" in spec:
        a, b = spec.split(":", 1)
        return Path(a), Path(b)
    new_path = Path(spec)
    old_path = new_path.with_name(new_path.stem + "_v3" + new_path.suffix)
    return new_path, old_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two versions of the misalignment-turn judge",
    )
    parser.add_argument(
        "pairs",
        nargs="+",
        help=(
            "Either <new.json> (old = <new>_v3.json sibling) or "
            "<new.json>:<old.json> explicit pair."
        ),
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to save the full report as JSON",
    )
    args = parser.parse_args()

    results: list[dict] = []
    for spec in args.pairs:
        new_path, old_path = _resolve_pair(spec)
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
    print(format_report(agg, per_file=results))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"aggregate": agg, "per_file": results}, f, indent=2)
        print(f"\nJSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
