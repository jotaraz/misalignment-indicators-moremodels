"""Union two indicator-judge runs into the primary output file.

Given a primary `_misalignment_turns.json` (typically the latest run) and
one or more secondary files (prior runs of the indicator judge), produces
a union over `indicators_turns`:

  * For each (variation_number, repetition_number, turn_index), merges the
    `present_indicators` lists per indicator name: spans are unioned and
    deduplicated (by exact text, case-sensitive).
  * An indicator flagged in only one run is kept with its spans.
  * `misaligned_turns`, `summary`, and top-level metadata are preserved
    from the PRIMARY file (the other judges' outputs aren't touched).

This lets you run the indicator judge twice with different taxonomies
(e.g. definition-only vs definition+examples) and consolidate the
complementary spans without losing either run's coverage.

Usage:
    python -m black_box_ind_judge.union_indicator_judge \
        --primary path/to/rollout_misalignment_turns.json \
        --secondary path/to/rollout_misalignment_turns_no_examples.json \
        [--secondary path/to/other.json ...] \
        [--dry-run]

The primary file is overwritten in place (make a backup first if unsure).
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _key(r: dict) -> tuple:
    return (r.get("variation_number"), r.get("repetition_number"))


def _union_present_indicators(lists: list[list[dict]]) -> list[dict]:
    """Union a list-of-present_indicators across runs.

    Returns a single list with one entry per indicator_name, spans
    deduplicated by exact text and sorted.
    """
    by_name: dict[str, set[str]] = {}
    for pi in lists:
        for ind in pi or []:
            name = ind.get("indicator_name")
            if not name:
                continue
            spans_raw = ind.get("spans") or []
            bucket = by_name.setdefault(name, set())
            for s in spans_raw:
                if isinstance(s, str) and s.strip():
                    bucket.add(s)
    return [
        {"indicator_name": name, "spans": sorted(by_name[name])}
        for name in sorted(by_name)
    ]


def _union_indicators_turns(turn_entry_lists: list[list[dict]]) -> list[dict]:
    """Union `indicators_turns` lists across runs. Entries with no present
    indicator after union are dropped (same mirror-of-misaligned-turns
    convention as the indicator judge's writer)."""
    by_turn: dict[int, list[list[dict]]] = {}
    for it_list in turn_entry_lists:
        for it in it_list or []:
            t_idx = it.get("turn_index")
            if t_idx is None:
                continue
            by_turn.setdefault(t_idx, []).append(it.get("present_indicators", []) or [])

    merged: list[dict] = []
    for t_idx in sorted(by_turn):
        unioned = _union_present_indicators(by_turn[t_idx])
        if unioned:
            merged.append({"turn_index": t_idx, "present_indicators": unioned})
    return merged


def union_one(primary_path: Path, secondary_paths: list[Path]) -> dict[str, Any]:
    primary = json.load(open(primary_path))
    secondaries = [json.load(open(p)) for p in secondary_paths]

    # Index secondaries by (var, rep) → rollout-entry for quick lookup
    sec_maps: list[dict[tuple, dict]] = []
    for s in secondaries:
        sec_maps.append({_key(r): r for r in s.get("evaluated_rollouts", [])})

    stats = {
        "primary_turns": 0,
        "primary_ind_presences": 0,
        "primary_spans": 0,
        "final_turns": 0,
        "final_ind_presences": 0,
        "final_spans": 0,
    }

    for r_prim in primary.get("evaluated_rollouts", []):
        k = _key(r_prim)
        prim_it = r_prim.get("indicators_turns", []) or []
        # Pre-union stats
        stats["primary_turns"] += sum(1 for it in prim_it if it.get("present_indicators"))
        for it in prim_it:
            for ind in it.get("present_indicators", []) or []:
                stats["primary_ind_presences"] += 1
                stats["primary_spans"] += len(ind.get("spans") or [])

        collected = [prim_it]
        for sm in sec_maps:
            r_sec = sm.get(k)
            if r_sec is None:
                continue
            collected.append(r_sec.get("indicators_turns", []) or [])

        unioned = _union_indicators_turns(collected)
        r_prim["indicators_turns"] = unioned
        stats["final_turns"] += len(unioned)
        for it in unioned:
            for ind in it.get("present_indicators", []):
                stats["final_ind_presences"] += 1
                stats["final_spans"] += len(ind.get("spans") or [])

    # Stamp union provenance in metadata
    primary.setdefault("metadata", {}).setdefault("indicator_judge", {})
    primary["metadata"]["indicator_judge"]["unioned_with"] = [str(p) for p in secondary_paths]

    return primary, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Union indicator-judge spans from multiple runs into the primary file",
    )
    parser.add_argument("--primary", type=str, required=True)
    parser.add_argument(
        "--secondary", type=str, action="append", default=[],
        help="Path to secondary file; repeatable.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print stats without writing.",
    )
    args = parser.parse_args()

    if not args.secondary:
        raise SystemExit("At least one --secondary required")

    primary_path = Path(args.primary)
    secondary_paths = [Path(p) for p in args.secondary]
    for p in [primary_path, *secondary_paths]:
        if not p.exists():
            raise SystemExit(f"missing file: {p}")

    merged, stats = union_one(primary_path, secondary_paths)

    print(f"Primary: {primary_path}")
    for p in secondary_paths:
        print(f"Secondary: {p}")
    print()
    print(
        f"  primary:   turns={stats['primary_turns']:>4}  "
        f"presences={stats['primary_ind_presences']:>4}  "
        f"spans={stats['primary_spans']:>5}"
    )
    print(
        f"  unioned:   turns={stats['final_turns']:>4}  "
        f"presences={stats['final_ind_presences']:>4}  "
        f"spans={stats['final_spans']:>5}"
    )
    d_turns = stats["final_turns"] - stats["primary_turns"]
    d_pres = stats["final_ind_presences"] - stats["primary_ind_presences"]
    d_spans = stats["final_spans"] - stats["primary_spans"]
    print(
        f"  delta:     turns={d_turns:>+4}  "
        f"presences={d_pres:>+4}  spans={d_spans:>+5}"
    )

    if args.dry_run:
        print("(dry run — no write)")
        return

    with open(primary_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote union → {primary_path}")


if __name__ == "__main__":
    main()
