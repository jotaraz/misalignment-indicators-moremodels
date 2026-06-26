"""
Add neutral dialogues to probe training data as additional negative examples.

Takes the last N entries from dialogues_filtered_v3.json, converts them to the
probe transcript format, and appends them to each indicator's training data.
Saves to a new directory to avoid modifying the original data.
"""

import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
NEUTRAL_PATH = REPO_ROOT / "probe" / "data" / "neutral" / "dialogues_filtered_v3.json"
SRC_DIR = REPO_ROOT / "probe" / "data" / "v3_v2_5"
DST_DIR = REPO_ROOT / "probe" / "data" / "v3_v2_5_with_neutral"
N_NEUTRAL = 100


def neutral_to_transcript(neutral_entry: dict, index: int) -> dict:
    """Convert a neutral dialogue to probe transcript format."""
    uid = hashlib.md5(
        neutral_entry["assistant_response"][:200].encode()
    ).hexdigest()[:8]

    messages = [
        {"role": "system", "content": neutral_entry["system_prompt"]},
        {"role": "user", "content": neutral_entry["user_prompt"]},
        {"role": "assistant", "content": neutral_entry["assistant_response"]},
    ]

    per_turn_labels = [
        {"turn_index": 2, "has_indicator": False},
    ]

    return {
        "scenario_description": "Neutral dialogue with no misalignment indicators.",
        "messages": messages,
        "per_turn_labels": per_turn_labels,
        "id": f"neutral_{index}_{uid}",
        "label": "honest",
        "negative_type": "neutral",
    }


def main():
    # Load neutral data
    with open(NEUTRAL_PATH) as f:
        all_neutral = json.load(f)
    neutral_subset = all_neutral[-N_NEUTRAL:]
    print(f"Selected last {len(neutral_subset)} neutral dialogues (from {len(all_neutral)} total)")

    # Convert to transcript format
    neutral_transcripts = [
        neutral_to_transcript(entry, i) for i, entry in enumerate(neutral_subset)
    ]

    # Process each indicator
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(SRC_DIR.glob("*.json")):
        with open(src_file) as f:
            data = json.load(f)

        orig_count = len(data["transcripts"])
        data["transcripts"].extend(neutral_transcripts)
        new_count = len(data["transcripts"])

        dst_file = DST_DIR / src_file.name
        with open(dst_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  {src_file.name}: {orig_count} -> {new_count} transcripts")

    print(f"\nSaved to {DST_DIR}")


if __name__ == "__main__":
    main()
