"""
Convert benign-misalignment inference outputs to rollout.json format.

Can be imported as a library (used by run_inference.py) or run standalone.

Usage (standalone):
    python convert_to_rollout.py --category story --model glm-4.7-flash
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROLLOUT_DIR = Path(__file__).parent / "rollouts"


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
    """Convert a single benign-misalignment result to a rollout dict."""
    system_prompt = entry.get("system_prompt", "")
    user_messages = entry.get("user_messages", [])
    assistant_responses = entry.get("assistant_responses", [])
    reasoning_responses = entry.get("reasoning_responses", [])
    category = entry.get("category", "unknown")
    indicator_slug = entry.get("indicator_slug", "unknown")
    indicator_name = entry.get("indicator_name", "")
    scenario_desc = entry.get("scenario_description", "")
    model_id = entry.get("model_id", "")

    # Variation description
    desc = f"[{category}/{indicator_slug}] {scenario_desc}"
    if len(desc) > 200:
        desc = desc[:200] + "..."

    # Build transcript events
    events = []

    if system_prompt:
        events.append(_make_event("system", system_prompt))

    # Interleave user/assistant turns
    n_turns = max(len(user_messages), len(assistant_responses))
    for turn_idx in range(n_turns):
        # User message
        if turn_idx < len(user_messages):
            events.append(_make_event("user", user_messages[turn_idx]))

        # Assistant message
        if turn_idx < len(assistant_responses):
            output = assistant_responses[turn_idx] or ""
            reasoning = (
                reasoning_responses[turn_idx]
                if turn_idx < len(reasoning_responses)
                else ""
            )

            if reasoning:
                assistant_content = [
                    {"type": "reasoning", "reasoning": reasoning},
                    {"type": "text", "text": output},
                ]
            else:
                assistant_content = output

            events.append(_make_event("assistant", assistant_content))

    return {
        "variation_number": index + 1,  # 1-indexed
        "variation_description": desc,
        "repetition_number": 0,
        "modality": f"benign-misalignment-{category}",
        "transcript": {
            "metadata": {
                "target_model": model_id,
                "category": category,
                "indicator_name": indicator_name,
                "indicator_slug": indicator_slug,
                "scenario_description": scenario_desc,
            },
            "events": events,
        },
    }


def convert_results_to_rollout(
    entries: list[dict], model: str, category: str,
) -> dict:
    """Convert inference results to rollout format."""
    rollouts = [convert_entry(entry, i) for i, entry in enumerate(entries)]
    return {
        "metadata": {
            "modality": f"benign-misalignment-{category}",
            "evaluator": "n/a",
            "target": model,
            "dataset": "benign-misalignment",
            "category": category,
            "num_rollouts": len(rollouts),
        },
        "rollouts": rollouts,
    }


def save_rollout(rollout_data: dict, output_path: Path):
    """Save rollout data to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rollout_data, f, indent=2, ensure_ascii=False)
    print(f"Saved rollout ({rollout_data['metadata']['num_rollouts']} entries) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert benign-misalignment JSONL to rollout.json"
    )
    parser.add_argument("--model", type=str, default="glm-4.7-flash")
    parser.add_argument(
        "--category", type=str, required=True,
        choices=["story", "roleplay", "user_instructed"],
    )
    args = parser.parse_args()

    jsonl_path = ROLLOUT_DIR / args.model / f"{args.category}.jsonl"
    if not jsonl_path.exists():
        print(f"JSONL not found: {jsonl_path}")
        return

    with open(jsonl_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    rollout_data = convert_results_to_rollout(entries, args.model, args.category)
    rollout_path = ROLLOUT_DIR / args.model / f"{args.category}_rollout.json"
    save_rollout(rollout_data, rollout_path)


if __name__ == "__main__":
    main()
