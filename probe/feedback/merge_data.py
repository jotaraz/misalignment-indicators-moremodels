"""
Stage 4: Merge feedback-generated training data with existing training data.

For each indicator, appends new transcripts from the feedback loop to the
existing training data, producing a combined dataset in a NEW directory.
The original training data is never modified.

Usage:
    python -m probe.feedback.merge_data \
        --original-dir probe/data/v2_4 \
        --feedback-dir probe/data/v2_4_feedback \
        --output-dir probe/data/v2_4_combined

Output is always written to --output-dir (default: {original_dir}_combined).
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def merge_indicator_data(
    original_path: Path,
    feedback_path: Path,
) -> dict:
    """Merge original and feedback transcripts for one indicator.

    Returns the merged data dict (same format as original).
    """
    with open(original_path) as f:
        original = json.load(f)

    with open(feedback_path) as f:
        feedback = json.load(f)

    original_transcripts = original.get("transcripts", [])
    feedback_transcripts = feedback.get("transcripts", [])

    # Tag feedback transcripts so they can be identified later
    for t in feedback_transcripts:
        t["source"] = "feedback"

    combined = original_transcripts + feedback_transcripts
    random.shuffle(combined)

    merged = {
        "indicator_name": original.get("indicator_name", feedback.get("indicator_name", "")),
        "indicator_definition": original.get("indicator_definition",
                                              feedback.get("indicator_definition", "")),
        "indicator_category": original.get("indicator_category",
                                            feedback.get("indicator_category", "")),
        "generation_model": original.get("generation_model", ""),
        "transcripts": combined,
        "merge_info": {
            "original_count": len(original_transcripts),
            "feedback_count": len(feedback_transcripts),
            "total_count": len(combined),
            "original_positive": sum(
                1 for t in original_transcripts if t.get("label") == "deceptive"
            ),
            "original_negative": sum(
                1 for t in original_transcripts if t.get("label") == "honest"
            ),
            "feedback_positive": sum(
                1 for t in feedback_transcripts if t.get("label") == "deceptive"
            ),
            "feedback_negative": sum(
                1 for t in feedback_transcripts if t.get("label") == "honest"
            ),
        },
    }

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge feedback data with existing training data")
    parser.add_argument(
        "--original-dir", type=str, required=True,
        help="Directory containing original training data (e.g., probe/data/v2_4)",
    )
    parser.add_argument(
        "--feedback-dir", type=str, required=True,
        help="Directory containing feedback-generated data (e.g., probe/data/v2_4_feedback)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for merged data (default: {original_dir}_combined)",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    feedback_dir = Path(args.feedback_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{original_dir}_combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_dir.exists():
        print(f"ERROR: original dir not found: {original_dir}")
        sys.exit(1)
    if not feedback_dir.exists():
        print(f"ERROR: feedback dir not found: {feedback_dir}")
        sys.exit(1)

    # Find all feedback files
    feedback_files = sorted(feedback_dir.glob("*.json"))
    if not feedback_files:
        print(f"No feedback files found in {feedback_dir}")
        sys.exit(1)

    print(f"Original data:  {original_dir}")
    print(f"Feedback data:  {feedback_dir}")
    print(f"Output:         {output_dir}")
    print(f"Feedback files: {len(feedback_files)}")
    print()

    # Also copy original files that have no feedback counterpart
    original_files = {f.name: f for f in original_dir.glob("*.json")}

    merged_count = 0
    copied_count = 0

    for fb_file in feedback_files:
        orig_file = original_dir / fb_file.name
        out_file = output_dir / fb_file.name

        if orig_file.exists():
            merged = merge_indicator_data(orig_file, fb_file)
            info = merged["merge_info"]

            with open(out_file, "w") as f:
                json.dump(merged, f, indent=2)

            print(f"  MERGED {fb_file.stem}: "
                  f"{info['original_count']} original + {info['feedback_count']} feedback "
                  f"= {info['total_count']} total "
                  f"(+{info['feedback_positive']} pos, +{info['feedback_negative']} neg)")
            merged_count += 1
        else:
            # No original — just copy feedback as-is
            with open(fb_file) as f:
                data = json.load(f)
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
            n = len(data.get("transcripts", []))
            print(f"  NEW    {fb_file.stem}: {n} transcripts (no original to merge with)")
            copied_count += 1

        # Remove from originals set so we know which are unmerged
        original_files.pop(fb_file.name, None)

    # Copy remaining original files that had no feedback
    for name, orig_file in sorted(original_files.items()):
        out_file = output_dir / name
        with open(orig_file) as f:
            data = json.load(f)
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        n = len(data.get("transcripts", []))
        print(f"  COPY   {orig_file.stem}: {n} transcripts (no feedback)")
        copied_count += 1

    print(f"\n{'='*60}")
    print(f"Merge complete: {merged_count} merged, {copied_count} copied")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
