"""Remove toxic/harmful dialogues from existing neutral FP scoring results.

Reads toxic_indices.json (indices into dialogues_filtered.json) and removes
matching entries from false_positives_*.json files, saving back in-place.

Usage:
    python -m probe.neutral.filter_toxic_fps \
        --fp-files probe/data/neutral/false_positives_v2_4_span_filtered.json:0 \
                   probe/data/neutral/false_positives_v2_4_combined_v3_span_dev.json:0 \
                   probe/data/neutral/false_positives_v2_4_combined_v3_span_test.json:1500

Each --fp-files entry is "path:offset" where offset is the dialogues_filtered.json
offset used when scoring.  dialogue_index in the FP file is local (0-based within
the slice), so the global index is offset + dialogue_index.
"""

import argparse
import json
from pathlib import Path


def filter_fp_file(fp_path: Path, offset: int, toxic_set: set[int]) -> dict:
    with open(fp_path) as f:
        data = json.load(f)

    triggered = data.get("triggered_dialogues", [])
    before = len(triggered)

    # Filter: global index = offset + dialogue_index
    filtered = [
        t for t in triggered
        if (offset + t["dialogue_index"]) not in toxic_set
    ]
    removed = before - len(filtered)

    # Update per_probe_counts
    per_probe = data.get("per_probe_counts", {})
    if removed > 0 and per_probe:
        # Recount from filtered triggered dialogues
        for slug in per_probe:
            per_probe[slug]["n_triggered"] = sum(
                1 for t in filtered if slug in t.get("triggered_probes", {})
            )

    data["triggered_dialogues"] = filtered
    data["n_dialogues_triggered"] = len(filtered)

    with open(fp_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"path": str(fp_path), "before": before, "after": len(filtered), "removed": removed}


def main():
    parser = argparse.ArgumentParser(
        description="Remove toxic dialogues from neutral FP scoring results"
    )
    parser.add_argument(
        "--toxic-indices", type=str,
        default="probe/data/neutral/toxic_indices.json",
    )
    parser.add_argument(
        "--fp-files", nargs="+", required=True,
        help="path:offset pairs (e.g. file.json:0 file.json:1500)",
    )
    args = parser.parse_args()

    with open(args.toxic_indices) as f:
        toxic_set = set(json.load(f)["toxic_indices"])
    print(f"Loaded {len(toxic_set)} toxic indices")

    for entry in args.fp_files:
        parts = entry.rsplit(":", 1)
        fp_path = Path(parts[0])
        offset = int(parts[1]) if len(parts) > 1 else 0

        if not fp_path.exists():
            print(f"  SKIP (not found): {fp_path}")
            continue

        result = filter_fp_file(fp_path, offset, toxic_set)
        print(f"  {fp_path.name} (offset={offset}): {result['before']} → {result['after']} ({result['removed']} removed)")


if __name__ == "__main__":
    main()
