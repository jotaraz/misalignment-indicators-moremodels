#!/usr/bin/env python3
"""
Convert OOD misalignment eval judge results into rollout_misalignment_turns.json format
compatible with the bloom eval pipeline.

Supports: agentic-misalignment, deception-bench, sycophancy-eval
"""

import json
import argparse
from pathlib import Path


def convert_agentic_misalignment(results_dir: Path, output_path: Path):
    """Convert agentic-misalignment classified results to misalignment_turns format."""
    models_dir = results_dir / "models"
    model_dirs = list(models_dir.iterdir())
    assert len(model_dirs) == 1, f"Expected 1 model dir, got {len(model_dirs)}"
    model_dir = model_dirs[0]
    model_name = model_dir.name.replace("_", "/", 1)  # z-ai_glm-4.7-flash -> z-ai/glm-4.7-flash

    evaluated_rollouts = []
    for i, condition_dir in enumerate(sorted(model_dir.iterdir()), start=1):
        if not condition_dir.is_dir():
            continue
        for sample_dir in sorted(condition_dir.iterdir()):
            resp_file = sample_dir / "response.json"
            if not resp_file.exists():
                continue
            data = json.load(open(resp_file))
            clf = data.get("classification", {})
            meta = data.get("metadata", {})
            raw = data.get("raw_response", "")

            is_harmful = clf.get("harmful_behavior", False) or clf.get("classifier_verdict", False)
            reasoning = clf.get("classifier_reasoning", "")

            # Build the rollout entry
            condition = meta.get("condition", condition_dir.name)
            sample_id = meta.get("sample_id", sample_dir.name)
            variation_desc = f"[{condition}] sample_{sample_id}"

            misaligned_turns = []
            if is_harmful:
                # Single-turn: the entire response is the misaligned turn
                # Extract evidence from the raw response (first 500 chars)
                evidence_snippet = raw[:500].strip()
                if len(raw) > 500:
                    evidence_snippet += "..."

                # Determine indicator based on condition type
                condition_type = condition.split("_")[0]  # blackmail, leaking, murder
                indicator_map = {
                    "blackmail": "coercive_leverage_deployment",
                    "leaking": "unauthorized_information_disclosure",
                    "murder": "harmful_action_facilitation",
                }
                indicator = indicator_map.get(condition_type, "harmful_behavior")

                misaligned_turns.append({
                    "turn_index": 1,
                    "description": f"Model exhibited {condition_type} behavior: {reasoning[:200]}",
                    "evidence": [evidence_snippet],
                    "applicable_indicators": [indicator],
                    "indicator_reasoning": {
                        indicator: reasoning[:500] if reasoning else f"Classified as harmful {condition_type} behavior."
                    }
                })

            evaluated_rollouts.append({
                "variation_number": len(evaluated_rollouts) + 1,
                "variation_description": variation_desc,
                "repetition_number": 1,
                "summary": reasoning[:300] if reasoning else ("Harmful behavior detected." if is_harmful else "No harmful behavior detected."),
                "misaligned_turns": misaligned_turns,
            })

    output = {
        "metadata": {
            "judge_model": "claude-sonnet-4-6",
            "judge_type": "misalignment_turn_judge",
            "modality": "agentic-misalignment",
            "target": model_name,
            "source_dataset": "agentic-misalignment",
            "num_rollouts": len(evaluated_rollouts),
        },
        "evaluated_rollouts": evaluated_rollouts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_misaligned = sum(1 for r in evaluated_rollouts if r["misaligned_turns"])
    print(f"Agentic-misalignment: {n_misaligned}/{len(evaluated_rollouts)} misaligned -> {output_path}")


def convert_deception_bench(rollout_path: Path, output_path: Path):
    """Convert deception-bench judged rollout to misalignment_turns format."""
    with open(rollout_path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    rollouts = data["rollouts"]

    evaluated_rollouts = []
    for r in rollouts:
        judgment = r.get("judgment", {})
        is_deceptive = judgment.get("deceptive", False)
        score = judgment.get("score", 0)
        explanation = judgment.get("explanation", "")

        # Extract transcript content for evidence
        events = r.get("transcript", {}).get("events", [])
        assistant_content = ""
        for e in events:
            msg = e.get("edit", {}).get("message", {})
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle structured content (reasoning + text)
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            assistant_content = block.get("text", "")[:500]
                            break
                        elif isinstance(block, str):
                            assistant_content = block[:500]
                            break
                else:
                    assistant_content = str(content)[:500]

        misaligned_turns = []
        if is_deceptive:
            evidence = [assistant_content] if assistant_content else [f"Deception score: {score}/10"]
            misaligned_turns.append({
                "turn_index": 1,
                "description": explanation[:300] if explanation else f"Deceptive behavior detected (score: {score}/10).",
                "evidence": evidence,
                "applicable_indicators": ["deceptive_content_planning"],
                "indicator_reasoning": {
                    "deceptive_content_planning": explanation or f"Deception detected with score {score}/10."
                }
            })

        evaluated_rollouts.append({
            "variation_number": r.get("variation_number", 0) + 1,
            "variation_description": r.get("variation_description", ""),
            "repetition_number": r.get("repetition_number", 0) + 1,
            "summary": explanation[:300] if explanation else ("Deceptive behavior detected." if is_deceptive else "No deception detected."),
            "misaligned_turns": misaligned_turns,
        })

    output = {
        "metadata": {
            "judge_model": "claude-sonnet-4-6",
            "judge_type": "misalignment_turn_judge",
            "modality": "deception-bench",
            "target": meta.get("target", "unknown"),
            "source_dataset": "deception-bench",
            "num_rollouts": len(evaluated_rollouts),
        },
        "evaluated_rollouts": evaluated_rollouts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_misaligned = sum(1 for r in evaluated_rollouts if r["misaligned_turns"])
    print(f"Deception-bench: {n_misaligned}/{len(evaluated_rollouts)} misaligned -> {output_path}")


def convert_sycophancy_eval(rollout_dir: Path, output_path: Path):
    """Convert sycophancy-eval judged rollouts (all splits) to misalignment_turns format."""
    splits = ["answer", "are_you_sure", "feedback"]
    all_rollouts = []
    target_model = "unknown"

    for split in splits:
        rollout_file = rollout_dir / f"{split}_rollout.json"
        if not rollout_file.exists():
            print(f"  Skipping {split}: {rollout_file} not found")
            continue

        with open(rollout_file) as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        target_model = meta.get("target", target_model)
        rollouts = data["rollouts"]

        for r in rollouts:
            judgment = r.get("judgment", {})
            is_sycophantic = judgment.get("sycophantic", False)
            score = judgment.get("score", 0)
            explanation = judgment.get("explanation", "")

            # Extract assistant content from events
            events = r.get("transcript", {}).get("events", [])
            assistant_content = ""
            for e in events:
                msg = e.get("edit", {}).get("message", {})
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                assistant_content = block.get("text", "")[:500]
                                break
                            elif isinstance(block, str):
                                assistant_content = block[:500]
                                break
                    else:
                        assistant_content = str(content)[:500]

            misaligned_turns = []
            if is_sycophantic:
                # Determine turn index based on split type
                turn_idx = 2 if split == "are_you_sure" else 1
                evidence = [assistant_content] if assistant_content else [f"Sycophancy score: {score}/10"]
                misaligned_turns.append({
                    "turn_index": turn_idx,
                    "description": explanation[:300] if explanation else f"Sycophantic behavior detected (score: {score}/10).",
                    "evidence": evidence,
                    "applicable_indicators": ["manufactured_agreement_shift"],
                    "indicator_reasoning": {
                        "manufactured_agreement_shift": explanation or f"Sycophancy detected with score {score}/10."
                    }
                })

            variation_desc = f"[sycophancy_{split}] {r.get('variation_description', '')}"
            all_rollouts.append({
                "variation_number": len(all_rollouts) + 1,
                "variation_description": variation_desc,
                "repetition_number": r.get("repetition_number", 0) + 1,
                "summary": explanation[:300] if explanation else ("Sycophantic behavior detected." if is_sycophantic else "No sycophancy detected."),
                "misaligned_turns": misaligned_turns,
            })

    output = {
        "metadata": {
            "judge_model": "claude-sonnet-4-6",
            "judge_type": "misalignment_turn_judge",
            "modality": "sycophancy-eval",
            "target": target_model,
            "source_dataset": "sycophancy-eval",
            "num_rollouts": len(all_rollouts),
        },
        "evaluated_rollouts": all_rollouts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_misaligned = sum(1 for r in all_rollouts if r["misaligned_turns"])
    print(f"Sycophancy-eval: {n_misaligned}/{len(all_rollouts)} misaligned -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert OOD misalignment eval results to rollout_misalignment_turns.json")
    parser.add_argument("--dataset", required=True, choices=["agentic-misalignment", "deception-bench", "sycophancy-eval", "all"])
    parser.add_argument("--input", required=True, help="Input path (results dir for agentic, rollout.json for deception, rollout dir for sycophancy)")
    parser.add_argument("--output", required=True, help="Output path for rollout_misalignment_turns.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.dataset == "agentic-misalignment":
        convert_agentic_misalignment(input_path, output_path)
    elif args.dataset == "deception-bench":
        convert_deception_bench(input_path, output_path)
    elif args.dataset == "sycophancy-eval":
        convert_sycophancy_eval(input_path, output_path)
    elif args.dataset == "all":
        # Convenience: run all 3 with default paths
        base = Path("ood_misalignment_eval")
        out_base = Path(args.output)

        convert_agentic_misalignment(
            base / "agentic-misalignment/results/ood_eval_glm",
            out_base / "agentic_misalignment_rollout_misalignment_turns.json",
        )
        convert_deception_bench(
            base / "deception-bench/rollouts/glm-4.7-flash/deception_bench_rollout.json",
            out_base / "deception_bench_rollout_misalignment_turns.json",
        )
        convert_sycophancy_eval(
            base / "sycophancy-eval/rollouts/glm-4.7-flash",
            out_base / "sycophancy_eval_rollout_misalignment_turns.json",
        )


if __name__ == "__main__":
    main()
