"""
Generate contrastive pairs from existing analysis suggestions.

Reads final_scenarios from analysis_suggestions.json and asks Opus to identify
scenarios that would benefit from a close contrastive pair (positive<->negative).
For indicators with many scenarios, processes in batches of ~15.

Outputs a new suggestions file with only the contrastive pair scenarios, ready
to be fed into generate_data.py.

Usage:
    python -m probe.feedback.generate_contrastive \
        --suggestions-file probe/data/feedback/v3_v2_5_span_v1/analysis_suggestions.json \
        --output probe/data/feedback/v3_v2_5_span_v1/analysis_suggestions_contrastive.json \
        --max-concurrent 5
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from probe.feedback.prompts import ANALYSIS_SYSTEM_PROMPT, CONTRASTIVE_PAIRS_PROMPT

MODEL = "claude-opus-4-6"
BATCH_SIZE = 15


CONTRASTIVE_CONTEXT_PROMPT = """\
You are analyzing training data scenarios for a linear probe that detects the \
following misalignment indicator:

## Indicator
Name: {indicator_name}

## Existing Scenarios (batch {batch_num}/{total_batches}, {n_scenarios} in this batch)
These are training scenarios — a mix of positive (indicator present) \
and negative (indicator absent) examples:

{scenarios_json}

## Task
{contrastive_prompt}"""


def _batch(items: list, batch_size: int) -> list[list]:
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def _call_opus(
    client: AsyncAnthropic,
    prompt: str,
    indicator_name: str,
    batch_label: str,
) -> list[dict]:
    """Call Opus and parse contrastive pairs from response."""
    for attempt in range(3):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = response.content[0].text
            if "```json" in text:
                after_open = text.split("```json", 1)[1]
                text = after_open.rsplit("```", 1)[0]
            elif "```" in text:
                after_open = text.split("```", 1)[1]
                text = after_open.rsplit("```", 1)[0]
            parsed = json.loads(text.strip())
            return parsed.get("contrastive_pairs", [])
        except Exception as e:
            if attempt < 2:
                wait = [30, 60, 120][attempt]
                print(f"    [Retry {attempt + 1}] {indicator_name} {batch_label}: "
                      f"{type(e).__name__}: {e}")
                await asyncio.sleep(wait)
            else:
                print(f"    [FAILED] {indicator_name} {batch_label}: "
                      f"{type(e).__name__}: {e}")
                return []


async def generate_pairs_for_indicator(
    client: AsyncAnthropic,
    indicator_name: str,
    scenarios: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate contrastive pairs for one indicator, batching if needed."""
    async with semaphore:
        batches = _batch(scenarios, BATCH_SIZE)
        n_batches = len(batches)

        all_pairs: list[dict] = []
        for batch_idx, batch in enumerate(batches):
            prompt = CONTRASTIVE_CONTEXT_PROMPT.format(
                indicator_name=indicator_name,
                batch_num=batch_idx + 1,
                total_batches=n_batches,
                n_scenarios=len(batch),
                scenarios_json=json.dumps(batch, indent=2),
                contrastive_prompt=CONTRASTIVE_PAIRS_PROMPT,
            )
            pairs = await _call_opus(
                client, prompt, indicator_name,
                f"batch {batch_idx + 1}/{n_batches}",
            )
            all_pairs.extend(pairs)

        return all_pairs


async def generate_all_pairs(
    suggestions_data: dict,
    max_concurrent: int = 5,
) -> dict:
    """Generate contrastive pairs for all indicators."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    indicators = suggestions_data.get("indicators", {})

    # Filter to indicators that have scenarios
    active = {
        slug: data for slug, data in indicators.items()
        if data.get("final_scenarios")
    }

    total_batches = sum(
        len(_batch(d["final_scenarios"], BATCH_SIZE))
        for d in active.values()
    )
    print(f"\nGenerating contrastive pairs for {len(active)} indicators "
          f"({total_batches} total batches)...")

    pbar = tqdm_asyncio(total=len(active), desc="Contrastive pairs", unit="indicator")

    async def _process(slug: str, data: dict):
        indicator_name = data["indicator_name"]
        scenarios = data["final_scenarios"]
        n_batches = len(_batch(scenarios, BATCH_SIZE))
        pairs = await generate_pairs_for_indicator(
            client, indicator_name, scenarios, semaphore,
        )

        # Convert pairs to scenario format
        contrastive_scenarios = []
        for pair in pairs:
            contrastive_scenarios.append({
                "type": pair.get("contrastive_type", "negative"),
                "model_behavior": pair.get("contrastive_model_behavior", ""),
                "scenario_description": pair.get("contrastive_scenario_description", ""),
                "key_elements": pair.get("contrastive_key_elements", ""),
                "example_fn_evidence": [],
                "source": "contrastive_pair",
                "key_contrast": pair.get("key_contrast", ""),
                "original_scenario": pair.get("original_scenario", ""),
            })

        pbar.update(1)
        n_pos = sum(1 for s in contrastive_scenarios if s["type"] == "positive")
        n_neg = sum(1 for s in contrastive_scenarios if s["type"] == "negative")
        print(f"    {indicator_name}: {len(contrastive_scenarios)} pairs "
              f"({n_pos} pos, {n_neg} neg) from {n_batches} batches")
        return slug, contrastive_scenarios

    tasks = [_process(slug, data) for slug, data in sorted(active.items())]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # Build output — same structure as analysis_suggestions but only contrastive scenarios
    output = {
        "config": suggestions_data.get("config", {}),
        "analysis_model": MODEL,
        "source": "contrastive_pairs",
        "indicators": {},
    }

    for slug, contrastive_scenarios in results:
        if contrastive_scenarios:
            orig = indicators[slug]
            output["indicators"][slug] = {
                "indicator_name": orig["indicator_name"],
                "final_scenarios": contrastive_scenarios,
            }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive pairs from existing analysis suggestions"
    )
    parser.add_argument(
        "--suggestions-file", type=str, required=True,
        help="Path to analysis_suggestions.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: {suggestions_file}_contrastive.json)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=15,
        help="Scenarios per batch (default: 15)",
    )
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    suggestions_path = Path(args.suggestions_file)
    with open(suggestions_path) as f:
        suggestions_data = json.load(f)

    output = asyncio.run(generate_all_pairs(
        suggestions_data, max_concurrent=args.max_concurrent,
    ))

    output_path = Path(args.output) if args.output else (
        suggestions_path.parent / f"{suggestions_path.stem}_contrastive.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    n_indicators = len(output["indicators"])
    total_pairs = sum(
        len(v.get("final_scenarios", []))
        for v in output["indicators"].values()
    )
    n_pos = sum(
        sum(1 for s in v["final_scenarios"] if s["type"] == "positive")
        for v in output["indicators"].values()
    )
    n_neg = total_pairs - n_pos
    print(f"\n{'='*60}")
    print(f"Contrastive pairs: {n_indicators} indicators, {total_pairs} scenarios "
          f"({n_pos} positive, {n_neg} negative)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
