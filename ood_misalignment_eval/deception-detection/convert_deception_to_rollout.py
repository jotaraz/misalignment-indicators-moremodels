"""Convert deception-detection rollout format to sycophancy-eval rollout format.

The sycophancy-eval format is the standard input for black_box_ind_judge.

Usage:
    python scripts/convert_deception_to_rollout.py \
        --input deception-detection/data/rollouts/insider_trading__onpolicy__glm-9b-flash.json \
        --output deception-detection/rollouts/glm-9b-flash/insider_trading__onpolicy_rollout.json
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def make_event(role: str, content: str) -> dict:
    """Create a transcript event in the sycophancy-eval format."""
    return {
        "type": "transcript_event",
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "view": ["target", "combined"],
        "edit": {
            "operation": "add",
            "message": {
                "role": role,
                "id": str(uuid.uuid4()),
                "content": content,
            },
        },
    }


def convert_rollout(source: dict, variation_number: int, repetition_number: int,
                    modality: str, model_name: str, dataset_name: str) -> dict:
    """Convert a single deception-detection rollout to sycophancy-eval format."""
    input_messages = source.get("input_messages", [])
    output_messages = source.get("output_messages", [])
    metadata = source.get("metadata", {})

    # Build events from input messages (skip empty assistant prefixes)
    events = []
    for msg in input_messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "assistant" and not content.strip():
            continue
        events.append(make_event(role, content))

    # Add the output message for this repetition
    if repetition_number < len(output_messages):
        out_msg = output_messages[repetition_number]
        events.append(make_event(out_msg["role"], out_msg["content"]))

    # Build variation_description from scenario or first user message
    description = metadata.get("scenario", "") or metadata.get("question", "")
    if not description:
        for msg in input_messages:
            if msg["role"] == "user":
                description = msg.get("content", "")
                break
    # Truncate for readability
    if len(description) > 200:
        description = description[:200] + "..."

    return {
        "variation_number": variation_number,
        "variation_description": f"[{dataset_name}] {description}",
        "repetition_number": repetition_number,
        "modality": modality,
        "transcript": {
            "metadata": {
                "target_model": model_name,
                "dataset": dataset_name,
                "base": {
                    "dataset": dataset_name,
                    **metadata,
                    "expect_deception": source.get("expect_deception"),
                    "graded_deception": source.get("graded_deception"),
                    "grader_reasoning": source.get("grader_reasoning"),
                    "deception_percentage": source.get("deception_percentage"),
                },
            },
            "events": events,
        },
    }


def convert_file(input_path: str, output_path: str) -> None:
    """Convert a deception-detection rollout file to sycophancy-eval format."""
    with open(input_path) as f:
        data = json.load(f)

    base_name = data.get("base_name", "unknown")
    variant = data.get("variant", "")
    model = data.get("model", "unknown")
    dataset_name = f"{base_name}__{variant}" if variant else base_name
    modality = "deception-detection"

    source_rollouts = data.get("rollouts", [])
    if isinstance(source_rollouts, str):
        print(f"Warning: rollouts field is '{source_rollouts}', skipping")
        return

    converted_rollouts = []
    var_num = 0
    for rollout in source_rollouts:
        num_outputs = len(rollout.get("output_messages", []))
        # Create one entry per output (repetition)
        for rep in range(max(num_outputs, 1)):
            converted_rollouts.append(
                convert_rollout(rollout, var_num, rep, modality, model, dataset_name)
            )
        var_num += 1

    result = {
        "metadata": {
            "modality": modality,
            "evaluator": "n/a",
            "target": model,
            "dataset": dataset_name,
            "num_rollouts": len(converted_rollouts),
            "source_file": str(input_path),
        },
        "rollouts": converted_rollouts,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Converted {len(converted_rollouts)} rollouts -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert deception-detection rollouts to sycophancy-eval format"
    )
    parser.add_argument("--input", required=True, help="Input deception-detection rollout JSON")
    parser.add_argument("--output", required=True, help="Output sycophancy-eval format JSON")
    args = parser.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
