"""
Convert DeceptionBench inference outputs to rollout.json format for the indicator judge.

Can be imported as a library (used by run_inference.py) or run standalone.

Usage (standalone):
    python convert_to_rollout.py --model glm-4.7-flash
    python convert_to_rollout.py --model glm-4.7-flash --type Sycophantic_Misrepresentation
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROLLOUT_DIR = Path(__file__).parent / "rollouts"

MAX_DESCRIPTION_LEN = 200


def _make_event(role: str, content, view=None):
    """Create a single transcript event."""
    if view is None:
        view = ["target", "combined"]
    return {
        "type": "transcript_event",
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "edit": {
            "operation": "add",
            "message": {
                "role": role,
                "id": str(uuid.uuid4()),
                "content": content,
            },
        },
    }


def convert_entry(entry: dict, index: int) -> dict:
    """Convert a single DeceptionBench result entry to a rollout dict."""
    system_prompt = entry.get("system_prompt", "")
    outer_prompt = entry.get("outer_prompt", "")
    inner_prompt = entry.get("inner_prompt", "")
    fmt = entry.get("format", "")
    deception_type = entry.get("type", "unknown")
    model_output = entry.get("output", "")
    reasoning = entry.get("reasoning")
    model_id = entry.get("model_id", "")
    entry_id = entry.get("id", "")

    # Build variation_description
    desc = outer_prompt
    if len(desc) > MAX_DESCRIPTION_LEN:
        desc = desc[:MAX_DESCRIPTION_LEN] + "..."
    variation_description = f"[{deception_type}] {desc}"

    # Build transcript events
    events = []

    # System message
    if system_prompt:
        events.append(_make_event("system", system_prompt))

    # User message: only the outer_prompt (what the user actually says).
    # inner_prompt and format are evaluation scaffolding, not part of the
    # real conversation — including them triggers false positives in probes.
    user_content = outer_prompt
    events.append(_make_event("user", user_content))

    # Assistant message
    full_output = model_output or ""
    if reasoning:
        assistant_content = [
            {"type": "reasoning", "reasoning": reasoning},
            {"type": "text", "text": full_output},
        ]
    else:
        assistant_content = full_output

    events.append(_make_event("assistant", assistant_content))

    return {
        "variation_number": index,
        "variation_description": variation_description,
        "repetition_number": 0,
        "modality": "deception-bench",
        "transcript": {
            "metadata": {
                "target_model": model_id,
                "deception_type": deception_type,
                "entry_id": entry_id,
                "system_prompt": system_prompt,
                "inner_prompt": inner_prompt,
                "outer_prompt": outer_prompt,
            },
            "events": events,
        },
    }


def convert_results_to_rollout(
    entries: list[dict], model: str, deception_type: str | None = None
) -> dict:
    """Convert in-memory inference results to rollout format.

    Args:
        entries: List of result dicts.
        model: Model display name.
        deception_type: Optional filter label for metadata.

    Returns:
        Rollout dict ready for JSON serialization.
    """
    rollouts = [convert_entry(entry, i) for i, entry in enumerate(entries)]
    return {
        "metadata": {
            "modality": "deception-bench",
            "evaluator": "n/a",
            "target": model,
            "dataset": "DeceptionBench",
            "deception_type": deception_type or "all",
            "num_rollouts": len(rollouts),
        },
        "rollouts": rollouts,
    }


def save_rollout(rollout_data: dict, output_path: Path):
    """Save rollout data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rollout_data, f, indent=2)
    print(f"Saved rollout ({rollout_data['metadata']['num_rollouts']} entries) -> {output_path}")


def convert_from_jsonl(model: str, deception_type: str | None = None) -> Path | None:
    """Convert a JSONL results file to rollout.json. Returns output path."""
    suffix = f"_{deception_type}" if deception_type else ""
    input_path = ROLLOUT_DIR / model / f"deception_bench{suffix}.jsonl"
    if not input_path.exists():
        print(f"Skipping: {input_path} not found")
        return None

    with open(input_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    rollout_data = convert_results_to_rollout(entries, model, deception_type)
    output_path = ROLLOUT_DIR / model / f"deception_bench{suffix}_rollout.json"
    save_rollout(rollout_data, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeceptionBench JSONL outputs to rollout.json for the indicator judge"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (subdirectory under rollouts/)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Deception type suffix (e.g., Sycophantic_Misrepresentation). "
        "If omitted, converts the main deception_bench.jsonl file.",
    )
    args = parser.parse_args()

    convert_from_jsonl(args.model, args.type)


if __name__ == "__main__":
    main()
