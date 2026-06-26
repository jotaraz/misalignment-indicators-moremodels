#!/usr/bin/env python3
"""Merge expanded dev transcripts into existing bloom-results directories.

For each transcript in --src, renumber variation from N to N + offset,
copy into --dst. Also merges rollout.json and rollout_misalignment_turns.json.

Usage:
    python merge_dev_expansion.py --src bloom-results-expansion/dir --dst bloom-results/dir
    python merge_dev_expansion.py --src ... --dst ... --offset 10 --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def merge_behavior(src: Path, dst: Path, offset: int = 10, dry_run: bool = False) -> bool:
    """Merge new transcripts from src into dst, renumbering variations.

    Returns True if merge was successful.
    """
    print(f"  Source:      {src}")
    print(f"  Destination: {dst}")
    print(f"  Offset:      +{offset}")
    if dry_run:
        print("  Mode:        DRY RUN (no files will be modified)")

    # ── 1. Copy and rename transcript files ──────────────────────────
    transcript_re = re.compile(r"^transcript_v(\d+)r(\d+)\.json$")
    src_transcripts = sorted(src.glob("transcript_v*r*.json"))

    if not src_transcripts:
        print("  WARNING: No transcript files found in source")
        return False

    print(f"  Transcripts: {len(src_transcripts)} files to merge")

    copied = 0
    for src_file in src_transcripts:
        match = transcript_re.match(src_file.name)
        if not match:
            print(f"    WARNING: Unexpected filename: {src_file.name} — skipping")
            continue

        old_v = int(match.group(1))
        rep = int(match.group(2))
        new_v = old_v + offset
        new_name = f"transcript_v{new_v}r{rep}.json"
        dst_file = dst / new_name

        if dst_file.exists():
            print(f"    SKIP (exists): {new_name}")
            continue

        if not dry_run:
            shutil.copy2(src_file, dst_file)
        print(f"    {src_file.name} -> {new_name}")
        copied += 1

    print(f"  Copied {copied} transcript files")

    # ── 2. Merge rollout.json ────────────────────────────────────────
    src_rollout = src / "rollout.json"
    dst_rollout = dst / "rollout.json"

    if src_rollout.exists() and dst_rollout.exists():
        with open(src_rollout) as f:
            src_data = json.load(f)
        with open(dst_rollout) as f:
            dst_data = json.load(f)

        new_rollouts = src_data.get("rollouts", [])
        for rollout in new_rollouts:
            if "variation_number" in rollout:
                rollout["variation_number"] += offset

        existing_count = len(dst_data.get("rollouts", []))

        if not dry_run:
            # Backup
            shutil.copy2(dst_rollout, dst / "rollout.json.bak")
            # Merge
            dst_data.setdefault("rollouts", []).extend(new_rollouts)
            dst_data["successful_count"] = len(dst_data["rollouts"])
            dst_data["total_count"] = len(dst_data["rollouts"])
            dst_data["variations_count"] = len(
                {r.get("variation_number", 0) for r in dst_data["rollouts"]}
            )
            with open(dst_rollout, "w") as f:
                json.dump(dst_data, f, indent=2, ensure_ascii=False)

        print(f"  rollout.json: {existing_count} + {len(new_rollouts)} = {existing_count + len(new_rollouts)} rollouts")
    elif src_rollout.exists():
        print("  WARNING: No rollout.json in destination — skipping rollout merge")
    else:
        print("  WARNING: No rollout.json in source — skipping rollout merge")

    # ── 3. Merge rollout_misalignment_turns.json (if present) ────────
    src_mt = src / "rollout_misalignment_turns.json"
    dst_mt = dst / "rollout_misalignment_turns.json"

    if src_mt.exists():
        with open(src_mt) as f:
            src_mt_data = json.load(f)

        new_evals = src_mt_data.get("evaluated_rollouts", [])
        for ev in new_evals:
            if "variation_number" in ev:
                ev["variation_number"] += offset

        if dst_mt.exists():
            with open(dst_mt) as f:
                dst_mt_data = json.load(f)

            existing_eval_count = len(dst_mt_data.get("evaluated_rollouts", []))

            if not dry_run:
                shutil.copy2(dst_mt, dst / "rollout_misalignment_turns.json.bak")
                dst_mt_data.setdefault("evaluated_rollouts", []).extend(new_evals)
                with open(dst_mt, "w") as f:
                    json.dump(dst_mt_data, f, indent=2, ensure_ascii=False)

            print(
                f"  rollout_misalignment_turns.json: "
                f"{existing_eval_count} + {len(new_evals)} = {existing_eval_count + len(new_evals)} evaluated rollouts"
            )
        else:
            # Create new file with renumbered entries
            if not dry_run:
                with open(dst_mt, "w") as f:
                    json.dump(src_mt_data, f, indent=2, ensure_ascii=False)
            print(f"  rollout_misalignment_turns.json: created with {len(new_evals)} evaluated rollouts")

    print("  Done!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge expanded dev transcripts into existing bloom-results"
    )
    parser.add_argument("--src", required=True, type=Path, help="Source dir (expansion output)")
    parser.add_argument("--dst", required=True, type=Path, help="Destination dir (existing bloom-results)")
    parser.add_argument("--offset", type=int, default=10, help="Variation number offset (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing any files")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: Source directory not found: {args.src}")
        return 1
    if not args.dst.is_dir():
        print(f"ERROR: Destination directory not found: {args.dst}")
        return 1

    ok = merge_behavior(args.src, args.dst, args.offset, args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
