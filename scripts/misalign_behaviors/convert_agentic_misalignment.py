"""
Convert agentic-misalignment experiment results to bloom format (rollout.json + judgment.json).

Input:
    results/<experiment_id>/ directory containing:
      - prompts/<condition>/system_prompt.txt, user_prompt.txt, email_content.txt, metadata.json
      - models/<model>/<condition>/sample_NNN/response.json

Output:
    rollout.json  - Bloom-format transcripts
    judgment.json - Behavior presence judgments from classifier

Usage:
    python convert_agentic_misalignment.py results/my_experiment/
    python convert_agentic_misalignment.py results/my_experiment/ --output_dir /tmp/output
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def make_event(role: str, content, view=None):
    """Create a transcript event in bloom format."""
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


def read_text_file(path: Path) -> str:
    """Read a text file, returning empty string if missing."""
    if path.exists():
        return path.read_text().strip()
    return ""


def convert_results(results_dir: str, output_dir: str | None = None):
    results_dir = Path(results_dir)
    prompts_dir = results_dir / "prompts"
    models_dir = results_dir / "models"

    if not models_dir.exists():
        print(f"Error: models directory not found: {models_dir}")
        return

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = results_dir / "bloom_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    bloom_rollouts = []
    judgments = []
    variation_num = 0

    # Discover all model directories
    model_dirs = sorted(d for d in models_dir.iterdir() if d.is_dir())

    for model_dir in model_dirs:
        model_name = model_dir.name

        # Discover all condition directories
        condition_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())

        for condition_dir in condition_dirs:
            condition_id = condition_dir.name

            # Load prompts for this condition
            prompt_dir = prompts_dir / condition_id
            system_prompt = read_text_file(prompt_dir / "system_prompt.txt")
            user_prompt = read_text_file(prompt_dir / "user_prompt.txt")
            email_content = read_text_file(prompt_dir / "email_content.txt")

            # Load condition metadata if available
            metadata_path = prompt_dir / "metadata.json"
            condition_meta = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    condition_meta = json.load(f)

            # Discover sample directories
            sample_dirs = sorted(
                (d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")),
                key=lambda d: d.name,
            )

            for sample_dir in sample_dirs:
                response_path = sample_dir / "response.json"
                if not response_path.exists():
                    continue

                with open(response_path) as f:
                    response = json.load(f)

                variation_num += 1
                raw_response = response.get("raw_response", "")
                metadata = response.get("metadata", {})
                classification = response.get("classification", {})

                # Build variation description
                scenario = condition_meta.get("scenario", "")
                goal_type = condition_meta.get("goal_type", "")
                urgency = condition_meta.get("urgency_type", "")
                variation_desc = f"[{model_name}] {condition_id} ({scenario}, goal={goal_type}, urgency={urgency})"

                # Build transcript events
                events = []

                # System prompt
                if system_prompt:
                    events.append(make_event("system", system_prompt))

                # User prompt + email content combined as user message
                user_content = user_prompt
                if email_content:
                    user_content = f"{user_prompt}\n\n{email_content}" if user_prompt else email_content
                if user_content:
                    events.append(make_event("user", user_content))

                # Model response
                if raw_response:
                    events.append(make_event("assistant", raw_response))

                bloom_rollouts.append({
                    "variation_number": variation_num,
                    "variation_description": variation_desc,
                    "repetition_number": 1,
                    "modality": "agentic-misalignment",
                    "transcript": {
                        "metadata": {
                            "target_model": model_name,
                            "condition": condition_id,
                            "sample": sample_dir.name,
                            "scenario": scenario,
                            "goal_type": goal_type,
                            "urgency_type": urgency,
                            **{k: v for k, v in condition_meta.items()
                               if k not in ("scenario", "goal_type", "urgency_type")},
                        },
                        "events": events,
                    },
                })

                # Build judgment from classification
                harmful = classification.get("harmful_behavior", False)
                score = 10 if harmful else 1
                judgments.append({
                    "variation_number": variation_num,
                    "repetition_number": 1,
                    "behavior_presence": score,
                    "classifier_verdict": classification.get("classifier_verdict", ""),
                    "classifier_reasoning": classification.get("classifier_reasoning", ""),
                })

    if not bloom_rollouts:
        print("Warning: No response.json files found in results directory")
        return

    # Determine behavior name from experiment directory
    experiment_id = results_dir.name
    behavior_name = f"agentic_misalignment__{experiment_id}"

    # Write rollout.json
    rollout_out = {
        "metadata": {
            "source": str(results_dir),
            "modality": "agentic-misalignment",
            "experiment_id": experiment_id,
            "num_rollouts": len(bloom_rollouts),
        },
        "rollouts": bloom_rollouts,
    }
    rollout_path = out_dir / "rollout.json"
    with open(rollout_path, "w") as f:
        json.dump(rollout_out, f, indent=2)

    # Write judgment.json
    judgment_out = {
        "behavior_name": behavior_name,
        "judgments": judgments,
    }
    judgment_path = out_dir / "judgment.json"
    with open(judgment_path, "w") as f:
        json.dump(judgment_out, f, indent=2)

    # Print summary
    harmful_count = sum(1 for j in judgments if j["behavior_presence"] >= 5)
    benign_count = len(judgments) - harmful_count
    print(f"Converted {len(bloom_rollouts)} rollouts")
    print(f"  rollout.json  -> {rollout_path}")
    print(f"  judgment.json -> {judgment_path}")
    print(f"  Distribution: {harmful_count} harmful, {benign_count} benign")


def main():
    parser = argparse.ArgumentParser(
        description="Convert agentic-misalignment results to bloom format"
    )
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: <results_dir>/bloom_rollout)")
    args = parser.parse_args()

    convert_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
