"""
Generate synthetic transcripts for probe training.

Uses Anthropic's Claude Sonnet 4.6 to generate multi-turn conversations
where a reasoning model exhibits (or does not exhibit) specific misalignment
indicators.

Supports indicator sets (--indicator-set):
  - finegrain (default): 20+5 indicators from misalignment_indicators_v2_2.py
  - behavior: 7 per-behavior indicators from misalignment_indicators_v2_2_general_behavior.py
  - v2_3: 19+5 indicators from misalignment_indicators_v2_3.py
  - v2_4: 19+5 indicators from misalignment_indicators_v2_4.py (with FP-informed non-examples)

Architecture:
  - max_concurrent parallel generation processes (default 5) via asyncio.Semaphore
  - Each process generates transcripts_per_process (default 10) transcripts
    sequentially, conditioning on previous transcripts for diversity
  - Both positive (indicator present) and negative (indicator absent) examples
  - Optional hard-negative generation (--hard-negative-ratio) for FP-prone domains

Usage:
    python -m probe.generate --indicator "Observer Suspicion Modeling" --k 50
    python -m probe.generate --all --k 50 --max-concurrent 5
    python -m probe.generate --indicator-set v2_4 --all --k 200 --hard-negative-ratio 0.25
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent

# Load .env from project root
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

from indicators.misalignment_indicators_v2_2 import (
    MISALIGNMENT_INDICATORS_V2,
    MISALIGNMENT_PRECONDITIONS,
    MisalignmentIndicator,
    get_indicator_v2_by_name,
)
from indicators.misalignment_indicators_v2_2_general_behavior import (
    BEHAVIOR_INDICATORS,
    get_behavior_indicator_by_name,
)
from indicators.misalignment_indicators_v2_3 import (
    MISALIGNMENT_INDICATORS_V2 as V2_3_INDICATORS,
    MISALIGNMENT_PRECONDITIONS as V2_3_PRECONDITIONS,
    BEHAVIORAL_CONCEPTS as V2_3_BEHAVIORAL_CONCEPTS,
    BehavioralConcept,
    get_indicator_v2_by_name as get_v2_3_indicator_by_name,
    get_behavioral_concept_by_name as get_v2_3_behavioral_concept_by_name,
)
from indicators.misalignment_indicators_v2_4 import (
    MISALIGNMENT_INDICATORS_V2 as V2_4_INDICATORS,
    MISALIGNMENT_PRECONDITIONS as V2_4_PRECONDITIONS,
    BEHAVIORAL_CONCEPTS as V2_4_BEHAVIORAL_CONCEPTS,
    get_indicator_v2_by_name as get_v2_4_indicator_by_name,
    get_behavioral_concept_by_name as get_v2_4_behavioral_concept_by_name,
)
from probe.prompts import (
    DIVERSITY_PROMPT_FRESH,
    DIVERSITY_PROMPT_WITH_CONTEXT,
    NEGATIVE_GENERATION_PROMPT,
    NEGATIVE_GENERATION_PROMPT_HARD,
    POSITIVE_GENERATION_PROMPT_MISALIGNED,
    POSITIVE_GENERATION_PROMPT_PRECONDITION,
    SYSTEM_PROMPT,
)

MODEL = "claude-sonnet-4-6"
DATA_DIRS = {
    "finegrain": Path(__file__).parent / "data" / "v2_2",
    "behavior": Path(__file__).parent / "data" / "v2_2_general_behavior",
    "v2_3": Path(__file__).parent / "data" / "v2_3",
    "v2_4": Path(__file__).parent / "data" / "v2_4",
}
TRANSCRIPTS_PER_PROCESS = 20  # number of transcripts each sequential process generates (tuned for diversity and response length)
NUM_EXCHANGES_RANGE = (2, 3)  # min/max user-assistant exchanges (each exchange = 1 user + 1 assistant message)
NUM_EXCHANGES_RANGE_LONG = (2, 3)  # same as standard to avoid token-budget slowdown


def indicator_name_to_filename(name: str) -> str:
    """Convert indicator name to a safe filename slug."""
    return name.lower().replace(" ", "_").replace("-", "_")


async def generate_single_transcript(
    client: AsyncAnthropic,
    indicator: MisalignmentIndicator,
    is_positive: bool,
    previous_scenarios: list[str],
    num_exchanges: int = 2,
    hard_negative: bool = False,
) -> dict[str, Any] | None:
    """
    Generate a single transcript for the given indicator.

    Args:
        client: AsyncAnthropic client
        indicator: The misalignment indicator to generate for
        is_positive: If True, generate WITH indicator; if False, WITHOUT
        previous_scenarios: Scenario descriptions from prior transcripts (for diversity)
        num_exchanges: Number of user-assistant exchanges
        hard_negative: If True, use hard-negative prompt targeting FP-prone domains

    Returns:
        Parsed transcript dict, or None on failure
    """
    if previous_scenarios:
        diversity = DIVERSITY_PROMPT_WITH_CONTEXT.format(
            previous_scenarios="\n".join(f"  - {s}" for s in previous_scenarios)
        )
    else:
        diversity = DIVERSITY_PROMPT_FRESH

    examples_str = "\n".join(f"  - {ex}" for ex in indicator.examples)
    non_examples_str = (
        "\n".join(f"  - {ne}" for ne in indicator.non_examples)
        if indicator.non_examples
        else "  (none provided)"
    )

    if is_positive:
        # Select the right prompt: preconditions/behavioral concepts use a
        # thinking-focused prompt; misalignment indicators use a prompt that
        # requires the model to actually follow through with misaligned behavior.
        is_precondition_or_concept = (
            isinstance(indicator, BehavioralConcept)
            or getattr(indicator, "is_precondition", False)
        )
        positive_template = (
            POSITIVE_GENERATION_PROMPT_PRECONDITION
            if is_precondition_or_concept
            else POSITIVE_GENERATION_PROMPT_MISALIGNED
        )
        prompt = positive_template.format(
            indicator_name=indicator.name,
            indicator_definition=indicator.definition,
            indicator_examples=examples_str,
            indicator_non_examples=non_examples_str,
            num_exchanges=num_exchanges,
            diversity_guidance=diversity,
        )
    else:
        neg_template = NEGATIVE_GENERATION_PROMPT_HARD if hard_negative else NEGATIVE_GENERATION_PROMPT
        prompt = neg_template.format(
            indicator_name=indicator.name,
            indicator_definition=indicator.definition,
            indicator_examples=examples_str,
            indicator_non_examples=non_examples_str,
            num_exchanges=num_exchanges,
            diversity_guidance=diversity,
        )

    for attempt in range(3):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=12000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )

            # Check for truncation
            if response.stop_reason == "max_tokens":
                raise ValueError("Response truncated (max_tokens). Retrying with shorter num_turns.")

            text = response.content[0].text

            # Parse JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = json.loads(text.strip())

            # Validate structure
            assert "messages" in parsed, "Missing 'messages' key"
            assert "per_turn_labels" in parsed, "Missing 'per_turn_labels' key"
            assert len(parsed["messages"]) >= 3, "Too few messages"

            # Validate assistant messages have <think> blocks
            for msg in parsed["messages"]:
                if msg["role"] == "assistant":
                    assert "<think>" in msg["content"] and "</think>" in msg["content"], (
                        f"Assistant message missing <think> block"
                    )

            # Validate positive transcripts have at least one indicator turn
            if is_positive:
                has_any_indicator = any(
                    tl.get("has_indicator", False)
                    for tl in parsed.get("per_turn_labels", [])
                )
                assert has_any_indicator, (
                    "Positive transcript has no turns with has_indicator=True"
                )

            return parsed

        except Exception as e:
            backoff_times = [30, 60, 300]
            if attempt < 2:
                wait = backoff_times[attempt]
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                label = "Rate limited" if is_rate_limit else f"{type(e).__name__}: {e}"
                print(f"  [Retry {attempt + 1}/3] {label}. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  [FAILED] {type(e).__name__}: {e}")
                return None


async def generate_process(
    client: AsyncAnthropic,
    indicator: MisalignmentIndicator,
    is_positive: bool,
    process_id: int,
    n_transcripts: int = TRANSCRIPTS_PER_PROCESS,
    pbar: Any = None,
    hard_negative: bool = False,
) -> list[dict[str, Any]]:
    """
    A single sequential generation process.

    Generates n_transcripts one after another, passing previous scenario
    descriptions as context for diversity.

    Args:
        client: AsyncAnthropic client
        indicator: The indicator to generate for
        is_positive: Whether to generate positive (indicator present) examples
        process_id: ID for this parallel process
        n_transcripts: Number of transcripts to generate sequentially
        pbar: Optional tqdm progress bar to update
        hard_negative: If True, use hard-negative prompt with longer exchanges

    Returns:
        List of transcript dicts with id and label added
    """
    transcripts = []
    previous_scenarios: list[str] = []
    label = "deceptive" if is_positive else "honest"

    exchange_range = NUM_EXCHANGES_RANGE_LONG if hard_negative else NUM_EXCHANGES_RANGE

    for i in range(n_transcripts):
        num_exchanges = random.randint(exchange_range[0], exchange_range[1])

        result = await generate_single_transcript(
            client=client,
            indicator=indicator,
            is_positive=is_positive,
            previous_scenarios=previous_scenarios,
            num_exchanges=num_exchanges,
            hard_negative=hard_negative,
        )

        if result is not None:
            slug = indicator_name_to_filename(indicator.name)
            result["id"] = f"{slug}_{label}_p{process_id}_t{i}_{uuid.uuid4().hex[:8]}"
            result["label"] = label
            transcripts.append(result)

            scenario = result.get("scenario_description", f"transcript_{i}")
            previous_scenarios.append(scenario)

        if pbar is not None:
            pbar.update(1)

    return transcripts


async def generate_for_indicator(
    client: AsyncAnthropic,
    indicator: MisalignmentIndicator,
    k: int,
    max_concurrent: int = 5,
    positive_only: bool = False,
    hard_negative_ratio: float = 0.0,
) -> dict[str, Any]:
    """
    Generate k positive and k negative transcripts for an indicator.

    Each process generates exactly TRANSCRIPTS_PER_PROCESS transcripts
    sequentially (for diversity conditioning). The last process may have
    fewer if k is not evenly divisible. A semaphore limits concurrency
    to max_concurrent processes at a time.

    Args:
        client: AsyncAnthropic client
        indicator: The misalignment indicator
        k: Number of positive transcripts (negative = k // 2)
        max_concurrent: Maximum concurrent generation processes
        positive_only: If True, only generate positive (deceptive) transcripts
        hard_negative_ratio: Fraction of negatives to generate as hard negatives
            (longer, FP-prone domain targeted). Default 0.0 (none).

    Returns:
        Full data dict ready to save as JSON
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Positive: ceil(k / TRANSCRIPTS_PER_PROCESS) processes
    n_processes = math.ceil(k / TRANSCRIPTS_PER_PROCESS)

    # Negative: same count as positive (1:1 ratio), skip if positive_only
    k_neg = 0 if positive_only else k
    # Split negatives into standard and hard
    k_hard_neg = int(k_neg * hard_negative_ratio)
    k_std_neg = k_neg - k_hard_neg
    n_processes_neg = math.ceil(k_std_neg / TRANSCRIPTS_PER_PROCESS) if k_std_neg > 0 else 0
    n_processes_hard_neg = math.ceil(k_hard_neg / TRANSCRIPTS_PER_PROCESS) if k_hard_neg > 0 else 0

    total_tasks = k + k_neg
    pbar = tqdm_asyncio(total=total_tasks, desc=f"  {indicator.name}", unit="transcript")

    async def limited_process(
        is_positive: bool, process_id: int, n: int,
        hard_negative: bool = False,
    ) -> list[dict]:
        async with semaphore:
            return await generate_process(
                client, indicator, is_positive, process_id, n, pbar=pbar,
                hard_negative=hard_negative,
            )

    tasks = []
    for i in range(n_processes):
        remaining = k - i * TRANSCRIPTS_PER_PROCESS
        n = min(TRANSCRIPTS_PER_PROCESS, remaining)
        tasks.append(limited_process(True, i, n))    # positive

    for i in range(n_processes_neg):
        remaining = k_std_neg - i * TRANSCRIPTS_PER_PROCESS
        n = min(TRANSCRIPTS_PER_PROCESS, remaining)
        tasks.append(limited_process(False, i, n))   # standard negative

    for i in range(n_processes_hard_neg):
        remaining = k_hard_neg - i * TRANSCRIPTS_PER_PROCESS
        n = min(TRANSCRIPTS_PER_PROCESS, remaining)
        tasks.append(limited_process(False, n_processes_neg + i, n, hard_negative=True))  # hard negative

    results = await asyncio.gather(*tasks)
    pbar.close()

    all_transcripts = []
    for batch in results:
        all_transcripts.extend(batch)

    n_pos = sum(1 for t in all_transcripts if t["label"] == "deceptive")
    n_neg = sum(1 for t in all_transcripts if t["label"] == "honest")
    print(f"  Generated {n_pos} positive + {n_neg} negative = {len(all_transcripts)} total")

    # Pre-shuffle so that positional train/val splits in probe.train are
    # deterministic without needing a random seed at split time.
    random.shuffle(all_transcripts)

    return {
        "indicator_name": indicator.name,
        "indicator_definition": indicator.definition,
        "indicator_category": indicator.category.value,
        "generation_model": MODEL,
        "transcripts": all_transcripts,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transcripts for probe training"
    )
    parser.add_argument(
        "--indicator-set", type=str, default="finegrain",
        choices=["finegrain", "behavior", "v2_3", "v2_4"],
        help="Indicator set: 'finegrain' (v2.2, default), 'behavior' (7 per-behavior), 'v2_3', or 'v2_4' (with FP-informed non-examples)",
    )
    parser.add_argument(
        "--indicator", type=str, nargs="+", default=None,
        help="Name(s) of indicator(s) to generate for",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate for all indicators",
    )
    parser.add_argument(
        "--include-preconditions", action="store_true",
        help="Also generate for precondition indicators (finegrain only)",
    )
    parser.add_argument(
        "--k", type=int, default=50,
        help="Number of transcripts per label (positive and negative) (default: 50)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Maximum concurrent generation processes (default: 10)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing data files",
    )
    parser.add_argument(
        "--positive-only", action="store_true",
        help="Only regenerate positive (deceptive) transcripts, keeping existing negative ones",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for generated transcripts (default: probe/data/<indicator-set>/)",
    )
    parser.add_argument(
        "--hard-negative-ratio", type=float, default=0.0,
        help="Fraction of negatives to generate as hard negatives "
             "(longer, FP-prone domain targeted). Default: 0.0",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    client = AsyncAnthropic(api_key=api_key)
    data_dir = Path(args.output_dir) if args.output_dir else DATA_DIRS[args.indicator_set]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which indicators to generate for
    if args.indicator_set == "behavior":
        if args.indicator:
            indicators = []
            for name in args.indicator:
                ind = get_behavior_indicator_by_name(name)
                if ind is None:
                    available = [i.name for i in BEHAVIOR_INDICATORS]
                    print(f"Unknown behavior indicator: {name}")
                    print(f"Available: {available}")
                    sys.exit(1)
                indicators.append(ind)
        elif args.all:
            indicators = list(BEHAVIOR_INDICATORS)
        else:
            parser.print_help()
            sys.exit(1)
    elif args.indicator_set == "v2_4":
        if args.indicator:
            indicators = []
            for name in args.indicator:
                ind = get_v2_4_indicator_by_name(name) or get_v2_4_behavioral_concept_by_name(name)
                if ind is None:
                    available = (
                        [i.name for i in V2_4_INDICATORS]
                        + [i.name for i in V2_4_PRECONDITIONS]
                        + [c.name for c in V2_4_BEHAVIORAL_CONCEPTS]
                    )
                    print(f"Unknown v2.4 indicator: {name}")
                    print(f"Available: {available}")
                    sys.exit(1)
                indicators.append(ind)
        elif args.all:
            indicators = list(V2_4_INDICATORS)
            if args.include_preconditions:
                indicators.extend(V2_4_PRECONDITIONS)
                indicators.extend(V2_4_BEHAVIORAL_CONCEPTS)
        else:
            parser.print_help()
            sys.exit(1)
    elif args.indicator_set == "v2_3":
        if args.indicator:
            indicators = []
            for name in args.indicator:
                ind = get_v2_3_indicator_by_name(name) or get_v2_3_behavioral_concept_by_name(name)
                if ind is None:
                    available = (
                        [i.name for i in V2_3_INDICATORS]
                        + [i.name for i in V2_3_PRECONDITIONS]
                        + [c.name for c in V2_3_BEHAVIORAL_CONCEPTS]
                    )
                    print(f"Unknown v2.3 indicator: {name}")
                    print(f"Available: {available}")
                    sys.exit(1)
                indicators.append(ind)
        elif args.all:
            indicators = list(V2_3_INDICATORS)
            if args.include_preconditions:
                indicators.extend(V2_3_PRECONDITIONS)
                indicators.extend(V2_3_BEHAVIORAL_CONCEPTS)
        else:
            parser.print_help()
            sys.exit(1)
    else:
        if args.indicator:
            indicators = []
            for name in args.indicator:
                ind = get_indicator_v2_by_name(name)
                if ind is None:
                    available = [i.name for i in MISALIGNMENT_INDICATORS_V2 + MISALIGNMENT_PRECONDITIONS]
                    print(f"Unknown indicator: {name}")
                    print(f"Available: {available}")
                    sys.exit(1)
                indicators.append(ind)
        elif args.all:
            indicators = list(MISALIGNMENT_INDICATORS_V2)
            if args.include_preconditions:
                indicators.extend(MISALIGNMENT_PRECONDITIONS)
        else:
            parser.print_help()
            sys.exit(1)

    print(f"Indicator set: {args.indicator_set}")
    print(f"Model: {MODEL}")
    print(f"Transcripts per label: {args.k}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Hard negative ratio: {args.hard_negative_ratio}")
    print(f"Indicators to generate: {len(indicators)}")
    print()

    for indicator in indicators:
        filename = indicator_name_to_filename(indicator.name)
        output_path = data_dir / f"{filename}.json"

        if args.positive_only:
            if not output_path.exists():
                print(f"Skipping {indicator.name} (no existing file to merge with)")
                continue

            print(f"Regenerating positive transcripts for: {indicator.name}")
            data = await generate_for_indicator(
                client, indicator, k=args.k, max_concurrent=args.max_concurrent,
                positive_only=True,
                hard_negative_ratio=args.hard_negative_ratio,
            )

            # Merge: keep existing negative transcripts, replace positive
            with open(output_path) as f:
                existing = json.load(f)
            existing_negative = [
                t for t in existing["transcripts"] if t.get("label") == "honest"
            ]
            new_positive = data["transcripts"]  # all deceptive
            data["transcripts"] = new_positive + existing_negative
            n_pos = len(new_positive)
            n_neg = len(existing_negative)
            print(f"  Merged {n_pos} new positive + {n_neg} existing negative = {n_pos + n_neg} total")

        else:
            if output_path.exists() and not args.overwrite:
                print(f"Skipping {indicator.name} (already exists at {output_path})")
                continue

            print(f"Generating for: {indicator.name}")
            data = await generate_for_indicator(
                client, indicator, k=args.k, max_concurrent=args.max_concurrent,
                hard_negative_ratio=args.hard_negative_ratio,
            )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to {output_path}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
