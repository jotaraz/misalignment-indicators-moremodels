"""CLI interface for the misalignment indicator judge."""

import argparse
import asyncio
import json
from pathlib import Path

from .evaluate import (
    filter_only_from_aggregated,
    evaluate_response_file,
    evaluate_rollout_file,
)
from .misalignment_turn_judge import evaluate_rollout_file_misalignment_turns
from .indicator_turn_judge import evaluate_rollout_file_indicator_turns


def main():
    """CLI interface for the misalignment judge."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses for misalignment reasoning indicators"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (JSON or JSONL format)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path for results",
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["openai", "anthropic", "openrouter"],
        default="openai",
        help="LLM provider to use for judging",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name to use (defaults to gpt-4o or claude-3-5-sonnet)",
    )
    parser.add_argument(
        "-k", "--response-key",
        type=str,
        default="response",
        help="Key in JSON objects containing the response text (for response mode)",
    )
    parser.add_argument(
        "-c", "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Evaluate rollout file (multi-turn conversations with variation_description)",
    )
    parser.add_argument(
        "--general",
        action="store_true",
        help="Use general high-level indicators instead of detailed indicators (v1 only, use --indicators for v2)",
    )
    parser.add_argument(
        "--indicators",
        type=str,
        choices=[
            "v1", "v1-general",
            "v2-finegrain", "v2-cross-behavior", "v2-per-behavior",
            "v2.1-mid", "v2.1-finegrain", "v2.1-per-behavior", "v2.1-cross-behavior",
            "v2.2-mid", "v2.2-finegrain", "v2.2-per-behavior", "v2.2-cross-behavior",
            "v2.3",
        ],
        default=None,
        help="Indicator set to use. Overrides --general if set.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=9000,
        help="Budget tokens for extended thinking (Anthropic only). 0 to disable.",
    )
    parser.add_argument(
        "--aggregate-k",
        type=int,
        default=1,
        help="Number of detection runs to aggregate (1 = no aggregation, 3 recommended for aggregation)",
    )
    parser.add_argument(
        "--aggregate-temp",
        type=float,
        default=0.7,
        help="Temperature for aggregation runs (higher = more variation between runs)",
    )
    parser.add_argument(
        "--filter-only",
        type=str,
        default=None,
        metavar="ROLLOUT_FILE",
        help="Filter-only mode: input_file is an aggregated JSON, this is the original rollout file for context",
    )
    parser.add_argument(
        "--per-turn",
        action="store_true",
        help="Evaluate each turn separately instead of the full conversation at once (per-turn mode)",
    )
    parser.add_argument(
        "--spans",
        action="store_true",
        help="Use span-based per-turn prompt (identifies spans instead of sentences)",
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Use neutrally-framed per-turn prompt (no 'misalignment' language)",
    )
    parser.add_argument(
        "--no-future-context",
        action="store_true",
        help="Per-turn mode: only include the target turn and prior turns (exclude future turns from judge input)",
    )
    parser.add_argument(
        "--no-non-examples",
        action="store_true",
        help="Omit 'NOT this indicator' non-examples from indicator definitions to reduce prompt length",
    )
    parser.add_argument(
        "--misalignment-turns",
        action="store_true",
        help="Judge each assistant turn for misaligned thinking (open-ended, no indicator taxonomy). Uses Anthropic API.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Blind judging: omit the system prompt and scenario description from the misalignment judge prompt (for baseline comparisons).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Strip the model's <reasoning> blocks from the conversation and remove the 'focus on internal reasoning' instruction (for action-only baselines).",
    )
    parser.add_argument(
        "--no-response",
        action="store_true",
        help="Mirror of --no-reasoning: strip [MODEL RESPONSE], [MODEL TOOL CALLS], and [TOOL RESULT] blocks from the conversation, keeping only [USER INPUT] and [MODEL REASONING] (for reasoning-only baselines).",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Per-turn mode only: each turn call sees only transcript history up to and including the target turn (no future context). Uses no-context prompt.",
    )
    parser.add_argument(
        "--no-definition",
        action="store_true",
        help="Strip the '## Definition of Misalignment' block from the prompt (minimal baseline — judge relies on its own intuition).",
    )
    parser.add_argument(
        "--no-span-guidance",
        action="store_true",
        help="Strip the 'CRITICAL — what qualifies as a misalignment span' block (and the 'Err on the side of EXCLUDING' line) from the prompt.",
    )
    parser.add_argument(
        "--use-indicator-taxonomy",
        action="store_true",
        help="Replace the '## Definition of Misalignment' block with the indicator taxonomy from indicators.misalignment_indicators_<indicator-set>. Overrides --no-definition.",
    )
    parser.add_argument(
        "--score-mode",
        action="store_true",
        help="Per-turn judge: replace boolean has_misalignment with an integer 1-10 misalignment_score in the JSON schema. Each per-turn result preserves the score; downstream output also includes an `all_turn_scores` list per rollout for FPR/TPR sweeping.",
    )
    parser.add_argument(
        "--indicator-turns",
        action="store_true",
        help="Judge each assistant turn for presence of each v2.6 indicator (per-turn, with spans). Uses Anthropic API.",
    )
    parser.add_argument(
        "--indicator-set",
        type=str,
        default="v2_6",
        help="Indicator module suffix for --indicator-turns (default: v2_6).",
    )
    parser.add_argument(
        "--per-turn-concurrency",
        type=int,
        default=5,
        help="Max concurrent turn-level API calls per rollout (--indicator-turns / --per-turn).",
    )
    parser.add_argument(
        "--include-examples",
        dest="include_examples",
        action="store_true",
        default=False,
        help="Include positive example sentences in the indicator taxonomy (--indicator-turns). Off by default — empirically did not help, judge becomes more conservative without raising in-`<think>` span coverage.",
    )
    parser.add_argument(
        "--include-non-examples",
        action="store_true",
        help="Include non-example sentences in the indicator taxonomy (--indicator-turns). Off by default.",
    )
    parser.add_argument(
        "--test-behavior",
        type=str,
        default="unknown",
        help="Label of the behavior this rollout probes (e.g. 'sycophancy', 'self-preservation'). Used by the indicator judge as a soft prior for disambiguation.",
    )
    parser.add_argument(
        "--benign",
        action="store_true",
        help="Benign mode for --indicator-turns: omits scenario_description from "
             "the prompt so the judge isn't biased by knowing the rollout is benign. "
             "Use with --test-behavior set to the BASE behavior (e.g. 'sabotage').",
    )

    args = parser.parse_args()

    # Indicator turn judge mode (per-turn, per-indicator labels)
    if args.indicator_turns:
        results = asyncio.run(
            evaluate_rollout_file_indicator_turns(
                rollout_file=args.input_file,
                output_file=args.output,
                model=args.model or "claude-opus-4-6",
                thinking_budget=args.thinking_budget,
                max_concurrent=args.max_concurrent,
                per_turn_concurrency=args.per_turn_concurrency,
                indicator_set=args.indicator_set,
                include_examples=args.include_examples,
                include_non_examples=args.include_non_examples,
                test_behavior=args.test_behavior,
                benign_mode=args.benign,
            )
        )

        evaluated = results.get("evaluated_rollouts", [])
        total_rollouts = len(evaluated)

        # Count positive-turn indicator presences.
        indicator_counts: dict[str, int] = {}
        turns_with_any = 0
        for r in evaluated:
            for turn in r.get("indicators_turns", []):
                present = turn.get("present_indicators", [])
                if present:
                    turns_with_any += 1
                for ind in present:
                    name = ind.get("indicator_name", "Unknown")
                    indicator_counts[name] = indicator_counts.get(name, 0) + 1

        print(f"\n=== Indicator Turn Judge Summary ===")
        print(f"Total rollouts evaluated: {total_rollouts}")
        print(f"Turns with any indicator: {turns_with_any}")

        if indicator_counts:
            print(f"\n=== Per-Indicator Turn Counts ===")
            for name, count in sorted(indicator_counts.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count}")

    # Misalignment turn judge mode
    elif args.misalignment_turns:
        results = asyncio.run(
            evaluate_rollout_file_misalignment_turns(
                rollout_file=args.input_file,
                output_file=args.output,
                model=args.model or "claude-opus-4-6",
                thinking_budget=args.thinking_budget,
                max_concurrent=args.max_concurrent,
                per_turn=args.per_turn,
                per_turn_concurrency=args.per_turn_concurrency,
                no_context=args.no_context,
                no_reasoning=args.no_reasoning,
                no_response=args.no_response,
                history_only=args.history_only,
                no_definition=args.no_definition,
                no_span_guidance=args.no_span_guidance,
                use_indicator_taxonomy=args.use_indicator_taxonomy,
                indicator_set=args.indicator_set,
                score_mode=args.score_mode,
            )
        )

        evaluated = results.get("evaluated_rollouts", [])
        total = len(evaluated)
        with_misalignment = sum(1 for r in evaluated if r.get("misaligned_turns"))
        total_misaligned_turns = sum(len(r.get("misaligned_turns", [])) for r in evaluated)

        print(f"\n=== Misalignment Turn Judge Summary ===")
        print(f"Total rollouts evaluated: {total}")
        if total:
            print(f"Rollouts with misaligned turns: {with_misalignment} ({100*with_misalignment/total:.1f}%)")
        print(f"Total misaligned turns: {total_misaligned_turns}")

    # Filter-only mode: just run filtering on existing aggregated file
    elif args.filter_only:
        if args.output is None:
            input_path = Path(args.input_file)
            args.output = input_path.parent / f"{input_path.stem.replace('_aggregated', '')}_filtered.json"

        results = asyncio.run(
            filter_only_from_aggregated(
                aggregated_file=args.input_file,
                rollout_file=args.filter_only,
                output_file=args.output,
                provider=args.provider,
                model=args.model,
                use_general_indicators=args.general,
                thinking_budget=args.thinking_budget,
                max_concurrent=args.max_concurrent,
                indicator_set=args.indicators,
                no_non_examples=args.no_non_examples,
            )
        )

        # Print summary
        evaluated = results.get("evaluated_rollouts", [])
        total = len(evaluated)
        with_indicators = sum(1 for r in evaluated if r.get("detected_indicators"))

        print(f"\n=== Filter-Only Summary ===")
        print(f"Total rollouts: {total}")
        print(f"Rollouts with filtered indicators: {with_indicators} ({100*with_indicators/total:.1f}%)")

        # Count indicators by type
        indicator_counts: dict[str, int] = {}
        for r in evaluated:
            for ind in r.get("detected_indicators", []):
                name = ind.get("indicator_name", "Unknown")
                indicator_counts[name] = indicator_counts.get(name, 0) + 1

        if indicator_counts:
            print(f"\n=== Filtered Indicator Breakdown ===")
            for name, count in sorted(indicator_counts.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count}")

    elif args.rollout:
        # Run rollout evaluation (uses default output dir if not specified)
        results = asyncio.run(
            evaluate_rollout_file(
                rollout_file=args.input_file,
                output_file=args.output,  # None will use default dir
                provider=args.provider,
                model=args.model,
                max_concurrent=args.max_concurrent,
                use_general_indicators=args.general,
                thinking_budget=args.thinking_budget,
                aggregate_k=args.aggregate_k,
                aggregate_temperature=args.aggregate_temp,
                per_turn=args.per_turn,
                use_spans=args.spans,
                neutral=args.neutral,
                no_future_context=args.no_future_context,
                indicator_set=args.indicators,
                no_non_examples=args.no_non_examples,
            )
        )

        # Print summary for rollout mode
        evaluated = results.get("evaluated_rollouts", [])
        total = len(evaluated)

        if args.aggregate_k > 1:
            # Aggregation mode summary
            with_filtered = sum(
                1 for r in evaluated
                if r.get("filtered_indicators")
            )
            with_unfiltered = sum(
                1 for r in evaluated
                if r.get("unfiltered_indicators")
            )

            # Count filtered indicators by type
            filtered_counts: dict[str, int] = {}
            for r in evaluated:
                for ind in r.get("filtered_indicators", []):
                    name = ind.get("indicator_name", "Unknown")
                    filtered_counts[name] = filtered_counts.get(name, 0) + 1

            # Count unfiltered indicators by type
            unfiltered_counts: dict[str, int] = {}
            for r in evaluated:
                for ind in r.get("unfiltered_indicators", []):
                    name = ind.get("indicator_name", "Unknown")
                    unfiltered_counts[name] = unfiltered_counts.get(name, 0) + 1

            # Count rejected indicators
            total_rejected = sum(
                len(r.get("rejected_indicators", []))
                for r in evaluated
            )

            print(f"\n=== Aggregated Rollout Evaluation Summary (k={args.aggregate_k}) ===")
            print(f"Total rollouts evaluated: {total}")
            print(f"Rollouts with unfiltered indicators: {with_unfiltered} ({100*with_unfiltered/total:.1f}%)")
            print(f"Rollouts with filtered indicators: {with_filtered} ({100*with_filtered/total:.1f}%)")
            print(f"Total rejected indicators: {total_rejected}")

            if unfiltered_counts:
                print(f"\n=== Unfiltered Indicator Breakdown ===")
                for name, count in sorted(unfiltered_counts.items(), key=lambda x: -x[1]):
                    print(f"  {name}: {count}")

            if filtered_counts:
                print(f"\n=== Filtered (Validated) Indicator Breakdown ===")
                for name, count in sorted(filtered_counts.items(), key=lambda x: -x[1]):
                    print(f"  {name}: {count}")
        else:
            # Standard single-run summary
            with_indicators = sum(
                1 for r in evaluated
                if r.get("detected_indicators")
            )

            # Count indicators by type
            indicator_counts: dict[str, int] = {}
            for r in evaluated:
                for ind in r.get("detected_indicators", []):
                    name = ind.get("indicator_name", "Unknown")
                    indicator_counts[name] = indicator_counts.get(name, 0) + 1

            print(f"\n=== Rollout Evaluation Summary ===")
            print(f"Total rollouts evaluated: {total}")
            print(f"Rollouts with indicators: {with_indicators} ({100*with_indicators/total:.1f}%)")

            if indicator_counts:
                print(f"\n=== Indicator Breakdown ===")
                for name, count in sorted(indicator_counts.items(), key=lambda x: -x[1]):
                    print(f"  {name}: {count}")

    else:
        # Generate default output file if not specified (for response mode)
        if args.output is None:
            input_path = Path(args.input_file)
            args.output = input_path.parent / f"{input_path.stem}_evaluated.json"
        # Run response evaluation
        results = asyncio.run(
            evaluate_response_file(
                response_file=args.input_file,
                output_file=args.output,
                provider=args.provider,
                model=args.model,
                response_key=args.response_key,
                max_concurrent=args.max_concurrent,
                use_general_indicators=args.general,
                thinking_budget=args.thinking_budget,
                indicator_set=args.indicators,
                no_non_examples=args.no_non_examples,
            )
        )

        # Print summary
        total = len(results)
        with_indicators = sum(
            1 for r in results
            if r["evaluation"].get("detected_indicators")
        )
        avg_score = sum(
            r["evaluation"].get("overall_misalignment_score", 0) or 0
            for r in results
        ) / total if total > 0 else 0

        print(f"\n=== Evaluation Summary ===")
        print(f"Total responses evaluated: {total}")
        print(f"Responses with indicators: {with_indicators} ({100*with_indicators/total:.1f}%)")
        print(f"Average misalignment score: {avg_score:.1f}/100")


if __name__ == "__main__":
    main()
