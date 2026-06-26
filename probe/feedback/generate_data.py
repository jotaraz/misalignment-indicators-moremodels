"""
Stage 3: Generate new training data from analysis suggestions using Sonnet 4.6.

For each indicator and each suggested scenario, generates synthetic transcripts
in the same format as probe.generate, using targeted scenario instructions
from the Stage 2 analysis.

Usage:
    python -m probe.feedback.generate_data \
        --suggestions-file probe/data/feedback/v2_4_span/analysis_suggestions.json \
        --output-dir probe/data/v2_4_feedback \
        --k 5 \
        --max-concurrent 10
"""

import argparse
import asyncio
import json
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from probe.feedback.prompts import (
    FEEDBACK_SYSTEM_PROMPT,
    FEEDBACK_POSITIVE_PROMPT,
    FEEDBACK_NEGATIVE_PROMPT,
)
from probe.generate import indicator_name_to_filename

MODEL = "claude-sonnet-4-6"
NUM_EXCHANGES_RANGE = (2, 3)


async def generate_single_transcript(
    client: AsyncAnthropic,
    indicator_name: str,
    indicator_definition: str,
    examples_str: str,
    non_examples_str: str,
    scenario: dict,
    num_exchanges: int = 2,
) -> dict[str, Any] | None:
    """Generate a single transcript for a specific scenario suggestion."""
    is_positive = scenario.get("type", "negative") == "positive"
    scenario_desc = scenario.get("scenario_description", "")
    key_elements = scenario.get("key_elements", "")
    model_behavior = scenario.get("model_behavior", "")
    fn_evidence = scenario.get("example_fn_evidence", [])

    if is_positive:
        # Format FN evidence for the prompt
        if fn_evidence:
            evidence_str = "\n".join(f'  - "{e[:400]}"' for e in fn_evidence[:5])
        else:
            evidence_str = "  (no reference evidence available)"

        prompt = FEEDBACK_POSITIVE_PROMPT.format(
            indicator_name=indicator_name,
            indicator_definition=indicator_definition,
            indicator_examples=examples_str,
            indicator_non_examples=non_examples_str,
            model_behavior=model_behavior if model_behavior else scenario_desc,
            scenario_instruction=scenario_desc,
            key_elements=key_elements,
            example_fn_evidence=evidence_str,
            num_exchanges=num_exchanges,
        )
    else:
        prompt = FEEDBACK_NEGATIVE_PROMPT.format(
            indicator_name=indicator_name,
            indicator_definition=indicator_definition,
            indicator_examples=examples_str,
            indicator_non_examples=non_examples_str,
            scenario_instruction=scenario_desc,
            key_elements=key_elements,
            num_exchanges=num_exchanges,
        )

    last_response_text = ""
    for attempt in range(3):
        try:
            async with client.messages.stream(
                model=MODEL,
                max_tokens=24000,
                system=FEEDBACK_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                extra_body={"output_config": {"effort": "medium"}},
            ) as stream:
                response = await stream.get_final_message()

            # With extended thinking, response.content may start with thinking
            # blocks; pick the text block explicitly.
            text = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"),
                "",
            )

            if response.stop_reason == "max_tokens":
                last_response_text = text
                raise ValueError("Response truncated (max_tokens)")

            last_response_text = text

            if "```json" in text:
                # Use last ``` as closing delimiter — responses may contain
                # nested code fences (```python ... ```) inside JSON strings
                after_open = text.split("```json", 1)[1]
                text = after_open.rsplit("```", 1)[0]
            elif "```" in text:
                after_open = text.split("```", 1)[1]
                text = after_open.rsplit("```", 1)[0]

            parsed = json.loads(text.strip())

            # Validate
            assert "messages" in parsed, "Missing 'messages' key"
            assert "per_turn_labels" in parsed, "Missing 'per_turn_labels' key"
            assert len(parsed["messages"]) >= 3, "Too few messages"

            for msg in parsed["messages"]:
                if msg["role"] == "assistant":
                    assert "<think>" in msg["content"] and "</think>" in msg["content"], (
                        "Assistant message missing <think> block"
                    )

            if is_positive:
                has_any = any(
                    tl.get("has_indicator", False) for tl in parsed.get("per_turn_labels", [])
                )
                assert has_any, "Positive transcript has no turns with has_indicator=True"

            return parsed

        except Exception as e:
            if attempt < 2:
                wait = [30, 60, 300][attempt]
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                label = "Rate limited" if is_rate_limit else f"{type(e).__name__}: {e}"
                print(f"    [Retry {attempt + 1}/3] {label}. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [FAILED] {type(e).__name__}: {e}")
                # Log truncated/failed response for debugging
                if last_response_text:
                    snippet = last_response_text[:300].replace("\n", "\\n")
                    print(f"    [RESPONSE SNIPPET] {snippet}")
                    if response and hasattr(response, "stop_reason"):
                        print(f"    [STOP REASON] {response.stop_reason}")
                return None


async def generate_for_scenario(
    client: AsyncAnthropic,
    indicator_name: str,
    indicator_definition: str,
    examples_str: str,
    non_examples_str: str,
    scenario: dict,
    k: int,
    scenario_idx: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm_asyncio | None = None,
) -> list[dict]:
    """Generate k transcripts for a single scenario."""
    async with semaphore:
        slug = indicator_name_to_filename(indicator_name)
        label = "deceptive" if scenario.get("type") == "positive" else "honest"
        transcripts = []

        for i in range(k):
            num_exchanges = random.randint(*NUM_EXCHANGES_RANGE)
            result = await generate_single_transcript(
                client=client,
                indicator_name=indicator_name,
                indicator_definition=indicator_definition,
                examples_str=examples_str,
                non_examples_str=non_examples_str,
                scenario=scenario,
                num_exchanges=num_exchanges,
            )

            if result is not None:
                result["id"] = f"{slug}_feedback_{label}_s{scenario_idx}_t{i}_{uuid.uuid4().hex[:8]}"
                result["label"] = label
                result["feedback_source"] = scenario.get("source", "unknown")
                transcripts.append(result)

            if pbar is not None:
                pbar.update(1)

        return transcripts


async def generate_for_indicator(
    client: AsyncAnthropic,
    slug: str,
    indicator_data: dict,
    indicator_def: dict,
    k: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Generate transcripts for all scenarios of an indicator."""
    indicator_name = indicator_data["indicator_name"]
    definition = indicator_def.get("definition", "")
    examples = indicator_def.get("examples", [])
    non_examples = indicator_def.get("non_examples", [])

    examples_str = "\n".join(f"  - {ex}" for ex in examples) if examples else "  (none)"
    non_examples_str = "\n".join(f"  - {ne}" for ne in non_examples) if non_examples else "  (none)"

    scenarios = indicator_data.get("final_scenarios", [])
    total_tasks = len(scenarios) * k

    print(f"\n  {indicator_name}: {len(scenarios)} scenarios x {k} transcripts = {total_tasks}")

    pbar = tqdm_asyncio(total=total_tasks, desc=f"    {indicator_name}", unit="transcript")

    tasks = [
        generate_for_scenario(
            client=client,
            indicator_name=indicator_name,
            indicator_definition=definition,
            examples_str=examples_str,
            non_examples_str=non_examples_str,
            scenario=scenario,
            k=k,
            scenario_idx=i,
            semaphore=semaphore,
            pbar=pbar,
        )
        for i, scenario in enumerate(scenarios)
    ]

    results = await asyncio.gather(*tasks)
    pbar.close()

    all_transcripts = []
    for batch in results:
        all_transcripts.extend(batch)

    n_pos = sum(1 for t in all_transcripts if t["label"] == "deceptive")
    n_neg = sum(1 for t in all_transcripts if t["label"] == "honest")
    print(f"    Generated {n_pos} positive + {n_neg} negative = {len(all_transcripts)} total")

    random.shuffle(all_transcripts)

    return {
        "indicator_name": indicator_name,
        "indicator_definition": definition,
        "indicator_category": indicator_def.get("category", ""),
        "generation_model": MODEL,
        "generation_source": "feedback_loop",
        "n_scenarios": len(scenarios),
        "transcripts": all_transcripts,
    }


async def generate_all(
    suggestions_data: dict,
    output_dir: Path,
    k: int = 5,
    max_concurrent: int = 30,
    indicator_set: str = "v2_4",
    model_override: str | None = None,
):
    """Generate feedback training data for all indicators."""
    global MODEL
    if model_override:
        MODEL = model_override

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    client = AsyncAnthropic(api_key=api_key)

    # Load indicator definitions
    from probe.feedback.collect_errors import _load_indicator_definitions
    indicator_defs = _load_indicator_definitions(indicator_set)

    indicators = suggestions_data.get("indicators", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating feedback data for {len(indicators)} indicators")
    print(f"Model: {MODEL}")
    print(f"Transcripts per scenario: {k}")
    print(f"Max concurrent: {max_concurrent}")

    # Global semaphore shared across all indicators
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _generate_and_save(slug, indicator_data):
        scenarios = indicator_data.get("final_scenarios", [])
        if not scenarios:
            return

        indicator_def = indicator_defs.get(slug, {
            "definition": "",
            "examples": [],
            "non_examples": [],
        })

        data = await generate_for_indicator(
            client=client,
            slug=slug,
            indicator_data=indicator_data,
            indicator_def=indicator_def,
            k=k,
            semaphore=semaphore,
        )

        output_path = output_dir / f"{slug}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    Saved to {output_path}")

    tasks = [
        _generate_and_save(slug, indicator_data)
        for slug, indicator_data in sorted(indicators.items())
    ]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Generate feedback training data from analysis suggestions"
    )
    parser.add_argument(
        "--suggestions-file", type=str, required=True,
        help="Path to analysis_suggestions.json from Stage 2",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for generated transcripts",
    )
    parser.add_argument(
        "--indicator-set", type=str, default="v2_4",
        choices=["v2_3", "v2_4", "v2_5", "v2_6"],
        help="Indicator set (default: v2_4)",
    )
    parser.add_argument(
        "--indicators", type=str, nargs="+", default=None,
        help="Only generate for these indicator slugs (default: all in suggestions file)",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of transcripts per (behavior, scenario) pair (default: 3)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=30,
        help="Max concurrent API calls (default: 30)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override generation model (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    suggestions_path = Path(args.suggestions_file)
    with open(suggestions_path) as f:
        suggestions_data = json.load(f)

    if args.indicators:
        wanted = set(args.indicators)
        orig = suggestions_data.get("indicators", {})
        missing = wanted - set(orig.keys())
        if missing:
            raise SystemExit(f"Indicators not found in suggestions file: {sorted(missing)}")
        suggestions_data = {
            **suggestions_data,
            "indicators": {s: orig[s] for s in sorted(wanted)},
        }
        print(f"Filtered to {len(wanted)} indicators: {sorted(wanted)}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "probe" / "data" / f"{args.indicator_set}_feedback"
    )

    asyncio.run(generate_all(
        suggestions_data=suggestions_data,
        output_dir=output_dir,
        k=args.k,
        max_concurrent=args.max_concurrent,
        indicator_set=args.indicator_set,
        model_override=args.model,
    ))

    print(f"\n{'='*60}")
    print(f"Generation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
