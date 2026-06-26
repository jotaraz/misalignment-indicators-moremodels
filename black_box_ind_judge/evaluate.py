"""Top-level evaluation functions for processing files."""

import asyncio
import json
from pathlib import Path
from typing import Any

from .judge import MisalignmentJudge
from .conversation import (
    extract_conversation_from_rollout,
    format_conversation_for_judge,
)

# Default output directory for indicator results
DEFAULT_OUTPUT_DIR = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/indicator_results")


async def filter_only_from_aggregated(
    aggregated_file: Path | str,
    rollout_file: Path | str,
    output_file: Path | str,
    provider: str = "anthropic",
    model: str | None = None,
    use_general_indicators: bool = True,
    thinking_budget: int = 10000,
    max_concurrent: int = 5,
    indicator_set: str | None = None,
    no_non_examples: bool = False,
) -> dict[str, Any]:
    """
    Filter aggregated indicators from an existing file (skip detection, run only filtering).

    Args:
        aggregated_file: Path to the aggregated JSON file (with unfiltered_indicators)
        rollout_file: Path to the original rollout file (for conversation context)
        output_file: Path to save filtered results
        provider: LLM provider
        model: Model name
        use_general_indicators: Use general indicators
        thinking_budget: Budget for extended thinking
        max_concurrent: Max concurrent API calls

    Returns:
        Dictionary with filtered results
    """
    aggregated_file = Path(aggregated_file)
    rollout_file = Path(rollout_file)
    output_file = Path(output_file)

    # Load aggregated data
    with open(aggregated_file) as f:
        aggregated_data = json.load(f)

    # Load original rollouts for conversation context
    with open(rollout_file) as f:
        rollout_data = json.load(f)

    rollouts = rollout_data.get("rollouts", rollout_data if isinstance(rollout_data, list) else [])
    evaluated_rollouts = aggregated_data.get("evaluated_rollouts", [])

    # Create rollout lookup by (variation_number, repetition_number)
    rollout_lookup = {}
    for r in rollouts:
        key = (r.get("variation_number"), r.get("repetition_number"))
        rollout_lookup[key] = r

    # Initialize judge
    judge = MisalignmentJudge(
        provider=provider,
        model=model,
        use_general_indicators=use_general_indicators,
        thinking_budget=thinking_budget,
        indicator_set=indicator_set,
        no_non_examples=no_non_examples,
    )

    # Process each evaluated rollout
    semaphore = asyncio.Semaphore(max_concurrent)

    async def filter_one(eval_rollout: dict) -> dict:
        async with semaphore:
            key = (eval_rollout.get("variation_number"), eval_rollout.get("repetition_number"))
            original_rollout = rollout_lookup.get(key)

            # Build aggregated structure for filter
            unfiltered = eval_rollout.get("unfiltered_indicators", [])
            num_runs = eval_rollout.get("num_runs", 3)

            # Skip if no unfiltered indicators
            if not unfiltered:
                print(f"Skipping rollout {key} - no unfiltered indicators")
                return {
                    "variation_number": eval_rollout.get("variation_number"),
                    "variation_description": eval_rollout.get("variation_description"),
                    "repetition_number": eval_rollout.get("repetition_number"),
                    "detected_indicators": [],
                    "rejected_indicators": [],
                    "filtering_summary": "No unfiltered indicators to process.",
                }

            if original_rollout is None:
                print(f"Warning: Could not find original rollout for {key}")
                return {
                    "variation_number": eval_rollout.get("variation_number"),
                    "variation_description": eval_rollout.get("variation_description"),
                    "repetition_number": eval_rollout.get("repetition_number"),
                    "detected_indicators": [],
                    "rejected_indicators": [],
                    "filtering_summary": "Could not find original rollout for context.",
                }

            # Get formatted conversation
            conversation_turns = extract_conversation_from_rollout(original_rollout)
            formatted_conversation = format_conversation_for_judge(conversation_turns)

            aggregated = {
                "num_runs": num_runs,
                "aggregated_indicators": unfiltered,
            }

            # Run filtering
            print(f"Filtering rollout {key} ({len(unfiltered)} indicators)...")
            try:
                filtered_result = await judge.filter_aggregated_indicators(
                    aggregated,
                    formatted_conversation,
                    mode="rollout",
                )
            except Exception as e:
                print(f"Error filtering rollout {key}: {e}")
                return {
                    "variation_number": eval_rollout.get("variation_number"),
                    "variation_description": eval_rollout.get("variation_description"),
                    "repetition_number": eval_rollout.get("repetition_number"),
                    "detected_indicators": [],
                    "rejected_indicators": [],
                    "filtering_summary": f"Error during filtering: {e}",
                }

            # Check for errors in result
            if "error" in filtered_result:
                print(f"Filter error for rollout {key}: {filtered_result['error']}")
                return {
                    "variation_number": eval_rollout.get("variation_number"),
                    "variation_description": eval_rollout.get("variation_description"),
                    "repetition_number": eval_rollout.get("repetition_number"),
                    "detected_indicators": [],
                    "rejected_indicators": [],
                    "filtering_summary": f"Filter error: {filtered_result['error']}",
                }

            return {
                "variation_number": eval_rollout.get("variation_number"),
                "variation_description": eval_rollout.get("variation_description"),
                "repetition_number": eval_rollout.get("repetition_number"),
                "detected_indicators": filtered_result.get("valid_indicators", []),
                "rejected_indicators": filtered_result.get("rejected_indicators", []),
                "filtering_summary": filtered_result.get("filtering_summary", ""),
            }

    # Run all filtering tasks
    tasks = [filter_one(er) for er in evaluated_rollouts]
    results = await asyncio.gather(*tasks)

    # Build output
    output_data = {
        "metadata": aggregated_data.get("metadata", {}),
        "evaluated_rollouts": results,
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Filtered results saved to {output_file}")

    return output_data


async def evaluate_response_file(
    response_file: Path | str,
    output_file: Path | str | None = None,
    provider: str = "openai",
    model: str | None = None,
    response_key: str = "response",
    max_concurrent: int = 5,
    use_general_indicators: bool = False,
    thinking_budget: int = 10000,
    indicator_set: str | None = None,
    no_non_examples: bool = False,
) -> list[dict[str, Any]]:
    """
    Evaluate responses from a file for misalignment indicators.

    Args:
        response_file: Path to JSON/JSONL file containing responses
        output_file: Path to save results (optional)
        provider: LLM provider ("openai" or "anthropic")
        model: Model name to use
        response_key: Key in the JSON objects containing the response text
        max_concurrent: Maximum concurrent API calls
        use_general_indicators: If True, use general high-level indicators
        thinking_budget: Budget tokens for extended thinking (Anthropic only). 0 to disable.

    Returns:
        List of evaluation results
    """
    response_file = Path(response_file)

    # Load responses
    if response_file.suffix == ".jsonl":
        with open(response_file) as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif response_file.suffix == ".json":
        with open(response_file) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
    else:
        raise ValueError(f"Unsupported file format: {response_file.suffix}. Use .json or .jsonl")

    # Extract responses
    responses = []
    for item in data:
        if isinstance(item, str):
            responses.append(item)
        elif isinstance(item, dict):
            if response_key in item:
                responses.append(item[response_key])
            else:
                # Try common keys
                for key in ["response", "content", "text", "output", "answer"]:
                    if key in item:
                        responses.append(item[key])
                        break
                else:
                    raise ValueError(f"Could not find response in item: {item.keys()}")
        else:
            raise ValueError(f"Unexpected item type: {type(item)}")

    # Initialize judge and evaluate
    judge = MisalignmentJudge(
        provider=provider,
        model=model,
        use_general_indicators=use_general_indicators,
        thinking_budget=thinking_budget,
        indicator_set=indicator_set,
        no_non_examples=no_non_examples,
    )
    results = await judge.judge_batch(responses, max_concurrent=max_concurrent)

    # Combine with original data
    combined_results = []
    for original, result in zip(data, results):
        combined = {
            "original": original,
            "evaluation": result,
        }
        combined_results.append(combined)

    # Save results if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(combined_results, f, indent=2)
        print(f"Results saved to {output_file}")

    return combined_results


async def evaluate_rollout_file(
    rollout_file: Path | str,
    output_file: Path | str | None = None,
    provider: str = "openai",
    model: str | None = None,
    max_concurrent: int = 5,
    use_general_indicators: bool = False,
    thinking_budget: int = 10000,
    aggregate_k: int = 1,
    aggregate_temperature: float = 0.7,
    per_turn: bool = False,
    use_spans: bool = False,
    neutral: bool = False,
    no_future_context: bool = False,
    indicator_set: str | None = None,
    no_non_examples: bool = False,
) -> dict[str, Any]:
    """
    Evaluate rollouts from a file for misalignment indicators.

    Args:
        rollout_file: Path to JSON file containing rollouts (with 'rollouts' key)
        output_file: Path to save results (defaults to indicator_results directory)
        provider: LLM provider ("openai" or "anthropic")
        model: Model name to use
        max_concurrent: Maximum concurrent API calls
        use_general_indicators: If True, use general high-level indicators
        thinking_budget: Budget tokens for extended thinking (Anthropic only). 0 to disable.
        aggregate_k: Number of detection runs to aggregate (1 = no aggregation)
        aggregate_temperature: Temperature for aggregation runs (higher = more variation)
        per_turn: If True, evaluate each turn separately instead of the full conversation at once
        use_spans: If True, use span-based per-turn prompt instead of sentence-based
        neutral: If True, use neutrally-framed per-turn prompt (no "misalignment" language)
        no_future_context: If True, per-turn judging only includes target turn and prior turns

    Returns:
        Dictionary with metadata and evaluated_rollouts
    """
    rollout_file = Path(rollout_file)

    # Default output file to indicator_results directory
    if output_file is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_aggregated" if aggregate_k > 1 else "_evaluated"
        output_file = DEFAULT_OUTPUT_DIR / f"{rollout_file.stem}{suffix}.json"
    else:
        output_file = Path(output_file)

    # Load rollouts
    with open(rollout_file) as f:
        data = json.load(f)

    # Handle different file structures
    if isinstance(data, dict) and "rollouts" in data:
        rollouts = data["rollouts"]
        metadata = data.get("metadata", {})
    elif isinstance(data, list):
        rollouts = data
        metadata = {}
    else:
        raise ValueError("Expected a JSON file with 'rollouts' key or a list of rollouts")

    # Initialize judge and evaluate
    judge = MisalignmentJudge(
        provider=provider,
        model=model,
        use_general_indicators=use_general_indicators,
        thinking_budget=thinking_budget,
        use_spans=use_spans,
        neutral=neutral,
        no_future_context=no_future_context,
        indicator_set=indicator_set,
        no_non_examples=no_non_examples,
    )

    mode_desc = "per-turn" if per_turn else "full-conversation"

    if aggregate_k > 1:
        # Use aggregation mode with k runs and filtering
        print(f"Running {mode_desc} aggregation mode with k={aggregate_k} runs per rollout...")
        if per_turn:
            results = await judge.judge_rollout_per_turn_batch_aggregated(
                rollouts,
                k=aggregate_k,
                temperature=aggregate_temperature,
                max_concurrent=max_concurrent,
            )
        else:
            results = await judge.judge_rollout_batch_aggregated(
                rollouts,
                k=aggregate_k,
                temperature=aggregate_temperature,
                max_concurrent=max_concurrent,
            )

        # Combine with original data - include both filtered and unfiltered
        combined_results = {
            "metadata": {
                **metadata,
                "per_turn": per_turn,
                "aggregation_k": aggregate_k,
                "aggregation_temperature": aggregate_temperature,
            },
            "evaluated_rollouts": [],
        }

        # Prepare individual run results (k separate files)
        individual_run_results = [{
            "metadata": {
                **metadata,
                "run_index": run_idx,
                "aggregation_k": aggregate_k,
            },
            "evaluated_rollouts": [],
        } for run_idx in range(aggregate_k)]

        for rollout, result in zip(rollouts, results):
            aggregated = result.get("aggregated", {})
            filtered = result.get("filtered", {})
            individual_runs = aggregated.get("individual_runs", [])

            combined = {
                "variation_number": rollout.get("variation_number"),
                "variation_description": rollout.get("variation_description"),
                "repetition_number": rollout.get("repetition_number"),
                # Unfiltered: all unique indicators from k runs with counts
                "unfiltered_indicators": aggregated.get("aggregated_indicators", []),
                "num_runs": aggregated.get("num_runs", aggregate_k),
                # Filtered: validated indicators after final judge pass
                "filtered_indicators": filtered.get("valid_indicators", []),
                "rejected_indicators": filtered.get("rejected_indicators", []),
                "filtering_summary": filtered.get("filtering_summary", ""),
            }
            combined_results["evaluated_rollouts"].append(combined)

            # Add to individual run results
            for run_idx, run_result in enumerate(individual_runs):
                if run_idx < len(individual_run_results):
                    individual_run_results[run_idx]["evaluated_rollouts"].append({
                        "variation_number": rollout.get("variation_number"),
                        "variation_description": rollout.get("variation_description"),
                        "repetition_number": rollout.get("repetition_number"),
                        "detected_indicators": run_result.get("detected_indicators", []),
                    })

        # Save individual run results
        output_file_path = Path(output_file)
        output_dir = output_file_path.parent
        output_stem = output_file_path.stem.replace("_aggregated", "")

        for run_idx, run_data in enumerate(individual_run_results):
            run_output_file = output_dir / f"{output_stem}_run{run_idx + 1}.json"
            with open(run_output_file, "w") as f:
                json.dump(run_data, f, indent=2)
            print(f"Run {run_idx + 1} results saved to {run_output_file}")
    else:
        # Standard single-run mode
        if per_turn:
            print(f"Running per-turn evaluation mode...")
            results = await judge.judge_rollout_per_turn_batch(rollouts, max_concurrent=max_concurrent)
        else:
            results = await judge.judge_rollout_batch(rollouts, max_concurrent=max_concurrent)

        # Combine with original data - only keep detected_indicators
        combined_results = {
            "metadata": {**metadata, "per_turn": per_turn},
            "evaluated_rollouts": [],
        }

        for rollout, result in zip(rollouts, results):
            combined = {
                "variation_number": rollout.get("variation_number"),
                "variation_description": rollout.get("variation_description"),
                "repetition_number": rollout.get("repetition_number"),
                "detected_indicators": result.get("detected_indicators", []),
            }
            combined_results["evaluated_rollouts"].append(combined)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Results saved to {output_file}")

    return combined_results
