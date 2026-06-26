"""
Convert sycophancy-eval outputs to rollout.json format for the indicator judge.

Can be imported as a library (used by run_inference.py) or run standalone.

Usage (standalone):
    python convert_to_rollout.py --model glm-4.7-flash --dataset answer
    python convert_to_rollout.py --model glm-4.7-flash --dataset all
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATASETS = ["answer", "are_you_sure", "feedback", "answer_filtered", "feedback_filtered"]
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


def convert_entry(entry: dict, index: int, dataset_name: str) -> dict:
    """Convert a single sycophancy JSONL entry to a rollout dict."""
    prompt = entry.get("prompt", [])
    base = entry.get("base", {})
    model_output = entry.get("output", "")
    reasoning = entry.get("reasoning")
    model_id = entry.get("model_id", "")

    # Build variation_description from base metadata
    question = base.get("question") or base.get("text", "")
    if len(question) > MAX_DESCRIPTION_LEN:
        question = question[:MAX_DESCRIPTION_LEN] + "..."
    ds_name = base.get("dataset", dataset_name)
    variation_description = f"[{ds_name}] {question}"

    # Build transcript events from prompt messages + model output
    events = []

    # Collect assistant prefix parts (for are_you_sure style prompts)
    assistant_prefix = ""

    for msg in prompt:
        msg_type = msg.get("type", "")
        msg_content = msg.get("content", "")
        if msg_type == "human":
            events.append(_make_event("user", msg_content))
        elif msg_type == "ai":
            # This is an assistant-priming prefix; combine with model output
            assistant_prefix = msg_content

    # Build the final assistant message
    full_output = assistant_prefix + (model_output or "")
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
        "modality": "sycophancy-eval",
        "transcript": {
            "metadata": {
                "target_model": model_id,
                "dataset": dataset_name,
                "base": base,
            },
            "events": events,
        },
    }


def convert_results_to_rollout(
    entries: list[dict], model: str, dataset_name: str
) -> dict:
    """Convert in-memory inference results to rollout format.

    Args:
        entries: List of result dicts (with prompt, base, output, model_id, etc.)
        model: Model display name.
        dataset_name: Dataset name (answer, are_you_sure, feedback).

    Returns:
        Rollout dict ready for JSON serialization.
    """
    rollouts = [
        convert_entry(entry, i, dataset_name) for i, entry in enumerate(entries)
    ]
    return {
        "metadata": {
            "modality": "sycophancy-eval",
            "evaluator": "n/a",
            "target": model,
            "dataset": dataset_name,
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


def convert_dataset(model: str, dataset_name: str) -> Path | None:
    """Convert a single dataset JSONL file to rollout.json. Returns output path."""
    input_path = ROLLOUT_DIR / model / f"{dataset_name}.jsonl"
    if not input_path.exists():
        print(f"Skipping {dataset_name}: {input_path} not found")
        return None

    with open(input_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    rollout_data = convert_results_to_rollout(entries, model, dataset_name)
    output_path = ROLLOUT_DIR / model / f"{dataset_name}_rollout.json"
    save_rollout(rollout_data, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert sycophancy JSONL outputs to rollout.json for the indicator judge"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (subdirectory under rollouts/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=DATASETS + ["all"],
        help="Which dataset to convert (default: all)",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        convert_dataset(args.model, ds)


if __name__ == "__main__":
    main()
