"""
Run GLM-4.7 Flash inference on generated benign-misalignment scenarios.

Takes scenario JSON files from generate_scenarios.py and produces transcripts
via OpenRouter, then auto-converts to rollout format.

Usage:
    python run_inference.py --category story
    python run_inference.py --category roleplay
    python run_inference.py --all --max-concurrent 50
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from convert_to_rollout import convert_results_to_rollout, save_rollout

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "z-ai/glm-4.7-flash"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"
OUTPUT_DIR = Path(__file__).parent / "rollouts"

DATASET_DEFAULTS = {
    "temperature": 1.0,
    "max_tokens": 4096,
    "stop": None,
}


async def call_api(
    client: AsyncOpenAI,
    messages: list[dict],
    model_id: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
    index: int,
) -> dict:
    """Single API call with retry."""
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                choice = response.choices[0]
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                return {
                    "index": index,
                    "output": choice.message.content or "",
                    "reasoning": reasoning,
                    "finish_reason": choice.finish_reason or "stop",
                }
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt + 1
                    await asyncio.sleep(wait)
                else:
                    return {
                        "index": index,
                        "output": "",
                        "reasoning": "",
                        "finish_reason": "error",
                        "error": str(e)[:200],
                    }


async def run_single_turn(
    client: AsyncOpenAI,
    scenarios: list[dict],
    model_id: str,
    max_concurrent: int,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
) -> list[dict]:
    """Run single-turn inference for all scenarios."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for i, scenario in enumerate(scenarios):
        messages = []
        sp = scenario.get("system_prompt", "")
        if sp:
            messages.append({"role": "system", "content": sp})

        user_msgs = scenario.get("user_messages", [])
        if user_msgs:
            messages.append({"role": "user", "content": user_msgs[0]})
        else:
            continue

        tasks.append(call_api(
            client, messages, model_id, semaphore,
            temperature, max_tokens, stop, i,
        ))

    results = await tqdm_asyncio.gather(*tasks, desc="inference")
    results_by_idx = {r["index"]: r for r in results}

    output_data = []
    for i, scenario in enumerate(scenarios):
        r = results_by_idx.get(i)
        if r is None:
            continue
        output_data.append({
            **scenario,
            "model": model_id.split("/")[-1],
            "model_id": model_id,
            "assistant_responses": [r["output"]],
            "reasoning_responses": [r["reasoning"]],
            "finish_reason": r["finish_reason"],
        })

    return output_data


async def run_multi_turn(
    client: AsyncOpenAI,
    scenarios: list[dict],
    model_id: str,
    max_concurrent: int,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
) -> list[dict]:
    """Run multi-turn inference sequentially per scenario, concurrent across scenarios."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(i: int, scenario: dict) -> dict:
        messages = []
        sp = scenario.get("system_prompt", "")
        if sp:
            messages.append({"role": "system", "content": sp})

        user_msgs = scenario.get("user_messages", [])
        assistant_responses = []
        reasoning_responses = []
        finish_reason = "stop"

        for turn_idx, user_msg in enumerate(user_msgs):
            messages.append({"role": "user", "content": user_msg})

            result = await call_api(
                client, messages, model_id, semaphore,
                temperature, max_tokens, stop, i * 100 + turn_idx,
            )

            assistant_responses.append(result["output"])
            reasoning_responses.append(result.get("reasoning", ""))
            finish_reason = result["finish_reason"]

            # Add assistant response for next turn
            messages.append({"role": "assistant", "content": result["output"]})

        return {
            **scenario,
            "model": model_id.split("/")[-1],
            "model_id": model_id,
            "assistant_responses": assistant_responses,
            "reasoning_responses": reasoning_responses,
            "finish_reason": finish_reason,
        }

    tasks = [run_one(i, s) for i, s in enumerate(scenarios)]
    results = await tqdm_asyncio.gather(*tasks, desc="multi-turn inference")
    return results


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {path}")


async def run(
    category: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_concurrent: int = 50,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Run inference for a category."""
    import os

    scenarios_path = SCENARIOS_DIR / f"{category}.json"
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios not found: {scenarios_path}")

    with open(scenarios_path) as f:
        scenarios = json.load(f)

    display_name = model_id.split("/")[-1]
    temp = temperature or DATASET_DEFAULTS["temperature"]
    tokens = max_tokens or DATASET_DEFAULTS["max_tokens"]
    stop = DATASET_DEFAULTS["stop"]

    print(f"\nCategory: {category}")
    print(f"Model: {display_name} ({model_id})")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Config: temperature={temp}, max_tokens={tokens}, max_concurrent={max_concurrent}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # Determine if any scenarios are multi-turn
    has_multi = any(len(s.get("user_messages", [])) > 1 for s in scenarios)

    if has_multi:
        # Split into single and multi-turn
        single = [s for s in scenarios if len(s.get("user_messages", [])) <= 1]
        multi = [s for s in scenarios if len(s.get("user_messages", [])) > 1]

        output_data = []
        if single:
            print(f"Running {len(single)} single-turn scenarios...")
            output_data.extend(await run_single_turn(
                client, single, model_id, max_concurrent, temp, tokens, stop,
            ))
        if multi:
            print(f"Running {len(multi)} multi-turn scenarios...")
            output_data.extend(await run_multi_turn(
                client, multi, model_id, max_concurrent, temp, tokens, stop,
            ))
    else:
        output_data = await run_single_turn(
            client, scenarios, model_id, max_concurrent, temp, tokens, stop,
        )

    # Save JSONL
    out_dir = OUTPUT_DIR / display_name
    jsonl_path = out_dir / f"{category}.jsonl"
    save_jsonl(output_data, jsonl_path)

    # Convert to rollout
    rollout_data = convert_results_to_rollout(output_data, display_name, category)
    rollout_path = out_dir / f"{category}_rollout.json"
    save_rollout(rollout_data, rollout_path)

    # Also save to bloom-compatible location
    bloom_dir = out_dir / "bloom" / f"benign_{category}"
    bloom_dir.mkdir(parents=True, exist_ok=True)
    save_rollout(rollout_data, bloom_dir / "rollout.json")

    n_success = sum(1 for r in output_data if r["finish_reason"] != "error")
    n_error = len(output_data) - n_success
    print(f"Done: {n_success} succeeded, {n_error} failed")

    return output_data


async def main():
    parser = argparse.ArgumentParser(
        description="Run GLM-4.7 Flash on benign-misalignment scenarios"
    )
    parser.add_argument(
        "--category", type=str,
        choices=["story", "roleplay", "user_instructed",
                 "behavior_story", "behavior_roleplay", "behavior_user_instructed"],
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    if not args.all and not args.category:
        parser.error("Specify --category or --all")

    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    start = time.time()

    categories = (
        ["story", "roleplay", "user_instructed"] if args.all
        else [args.category]
    )

    for category in categories:
        await run(
            category=category,
            model_id=args.model,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
