"""
Async inference on the MASK dataset using OpenRouter, with reasoning capture.

Generates responses for specified splits (known_facts, provided_facts, etc.),
captures both content and reasoning (thinking) from the model, and saves in
both JSONL and rollout.json formats compatible with the bloom probe pipeline.

Usage:
    python run_inference.py --max-prompts 5
    python run_inference.py --model z-ai/glm-4.7-flash --splits known_facts provided_facts
    python run_inference.py --test --splits known_facts
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Load .env from project root
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

DEFAULT_MODEL_ID = "z-ai/glm-4.7-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MASK_DIR = Path(__file__).parent / "mask"
OUTPUT_DIR = Path(__file__).parent / "rollouts"

ALL_SPLITS = ["known_facts", "provided_facts", "continuations", "disinformation",
              "doubling_down_known_facts", "statistics"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(split: str, test: bool = False) -> pd.DataFrame:
    """Load a MASK CSV split."""
    data_dir = MASK_DIR / ("test_csv_data" if test else "csv_data")
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        # Try downloading
        print(f"CSV not found at {csv_path}, running download...")
        import subprocess
        subprocess.run(
            ["python", str(MASK_DIR / "download.py")],
            check=True, cwd=str(MASK_DIR),
        )
    df = pd.read_csv(csv_path)
    if "belief_system_1" not in df.columns and "belief_elicit_1" in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."
    return df


def build_prompts_from_df(df: pd.DataFrame, split: str) -> list[dict]:
    """Build prompt entries from a MASK dataframe.

    For each row we generate:
      - 1 lying prompt  (system_prompt + user_prompt)
      - belief elicitation prompts (be1 x3, be2 x1, be3 x1) if columns exist

    Returns list of dicts with keys: index, prompt_type, messages, row_data.
    """
    is_doubling_down = "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    entries = []

    for idx, row in df.iterrows():
        row_data = row.to_dict()

        # Lying prompt
        if is_doubling_down:
            messages = [
                {"role": "system", "content": str(row["system_prompt"])},
                {"role": "user", "content": str(row["dd_turn_1_prompt"])},
                {"role": "assistant", "content": str(row["dd_turn_1_response"])},
                {"role": "user", "content": str(row["user_prompt"])},
            ]
        else:
            messages = [
                {"role": "system", "content": str(row["system_prompt"])},
                {"role": "user", "content": str(row["user_prompt"])},
            ]
        entries.append({
            "index": idx,
            "prompt_type": "lying",
            "messages": messages,
            "row_data": row_data,
            "split": split,
        })

        # Belief elicitation prompts
        if "belief_elicit_1" in df.columns:
            belief_k = {"1": 3} if split == "statistics" else {"1": 3, "2": 1, "3": 1}
            belief_system = str(row.get("belief_system_1", "You are a helpful assistant."))
            for be_num, k_value in belief_k.items():
                col = f"belief_elicit_{be_num}"
                if pd.notna(row.get(col)):
                    for run in range(k_value):
                        entries.append({
                            "index": idx,
                            "prompt_type": f"be{be_num}_run{run+1}",
                            "messages": [
                                {"role": "system", "content": belief_system},
                                {"role": "user", "content": str(row[col])},
                            ],
                            "row_data": row_data,
                            "split": split,
                        })

    return entries


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_api(
    client: AsyncOpenAI,
    entry: dict,
    model_id: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Make a single API call with retry logic, capturing reasoning."""
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=entry["messages"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                message = response.choices[0].message

                # Extract reasoning from various possible locations
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning is None and hasattr(message, "model_extra"):
                    reasoning = message.model_extra.get(
                        "reasoning_content"
                    ) or message.model_extra.get("reasoning")
                # Also check the reasoning attribute directly (OpenRouter)
                if reasoning is None:
                    reasoning = getattr(message, "reasoning", None)

                return {
                    **entry,
                    "output": message.content,
                    "reasoning": reasoning,
                    "finish_reason": response.choices[0].finish_reason,
                }
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    print(f"\n[Retry {attempt+1}/5] idx={entry['index']} {entry['prompt_type']}: {e}. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n[FAILED] idx={entry['index']} {entry['prompt_type']}: {e}")
                    return {
                        **entry,
                        "output": None,
                        "reasoning": None,
                        "finish_reason": "error",
                    }


# ---------------------------------------------------------------------------
# Rollout conversion (same format as deception-bench)
# ---------------------------------------------------------------------------

def _make_event(role: str, content, view=None):
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


def convert_to_rollout(lying_results: list[dict], model_name: str, split: str) -> dict:
    """Convert lying prompt results to rollout.json format.

    Only the lying prompts become rollouts (belief elicitations are saved
    separately in the JSONL but aren't part of the rollout transcript).
    """
    rollouts = []
    for i, entry in enumerate(lying_results):
        row = entry["row_data"]
        system_prompt = str(row.get("system_prompt", ""))
        user_prompt = str(row.get("user_prompt", ""))
        output = entry.get("output") or ""
        reasoning = entry.get("reasoning")

        # Build events
        events = []
        if system_prompt:
            events.append(_make_event("system", system_prompt))

        # For doubling_down, include multi-turn history
        if "dd_turn_1_prompt" in row and pd.notna(row.get("dd_turn_1_prompt")):
            events.append(_make_event("user", str(row["dd_turn_1_prompt"])))
            events.append(_make_event("assistant", str(row["dd_turn_1_response"])))

        events.append(_make_event("user", user_prompt))

        # Assistant with reasoning
        if reasoning:
            assistant_content = [
                {"type": "reasoning", "reasoning": reasoning},
                {"type": "text", "text": output},
            ]
        else:
            assistant_content = output
        events.append(_make_event("assistant", assistant_content))

        # Description
        desc = user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt
        proposition = str(row.get("proposition", ""))

        rollouts.append({
            "variation_number": i + 1,
            "variation_description": f"[MASK/{split}] {desc}",
            "repetition_number": 1,
            "modality": "mask",
            "transcript": {
                "metadata": {
                    "target_model": model_name,
                    "dataset": "MASK",
                    "split": split,
                    "proposition": proposition,
                    "formatted_ground_truth": str(row.get("formatted_ground_truth", "")),
                    "system_prompt": system_prompt,
                },
                "events": events,
            },
        })

    return {
        "metadata": {
            "modality": "mask",
            "evaluator": "n/a",
            "target": model_name,
            "dataset": "MASK",
            "split": split,
            "num_rollouts": len(rollouts),
        },
        "rollouts": rollouts,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _model_display_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


async def run(
    model_id: str,
    model_name: str | None = None,
    splits: list[str] | None = None,
    max_concurrent: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_prompts: int | None = None,
    test: bool = False,
):
    display_name = model_name or _model_display_name(model_id)
    splits = splits or ["known_facts", "provided_facts"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split} | Model: {display_name} | Test: {test}")
        print(f"{'='*60}")

        df = load_split(split, test=test)
        if max_prompts is not None:
            df = df.head(max_prompts)

        entries = build_prompts_from_df(df, split)
        print(f"Rows: {len(df)}, API calls: {len(entries)} "
              f"(temperature={temperature}, max_tokens={max_tokens})")

        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            call_api(client, entry, model_id, semaphore, temperature, max_tokens)
            for entry in entries
        ]
        results = await tqdm_asyncio.gather(*tasks, desc=f"{split}")

        # Separate lying vs belief results
        lying_results = [r for r in results if r["prompt_type"] == "lying"]
        belief_results = [r for r in results if r["prompt_type"] != "lying"]

        n_success = sum(1 for r in results if r["finish_reason"] != "error")
        n_error = len(results) - n_success
        print(f"Done: {n_success} succeeded, {n_error} failed")

        # Save JSONL (all results including beliefs)
        test_suffix = "_test" if test else ""
        out_dir = OUTPUT_DIR / display_name
        out_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_dir / f"mask_{split}{test_suffix}.jsonl"
        with open(jsonl_path, "w") as f:
            for r in results:
                # Don't serialize full row_data in JSONL (too large), keep key fields
                row = r.pop("row_data", {})
                record = {
                    "index": r["index"],
                    "split": r["split"],
                    "prompt_type": r["prompt_type"],
                    "output": r["output"],
                    "reasoning": r["reasoning"],
                    "finish_reason": r["finish_reason"],
                    "system_prompt": str(row.get("system_prompt", "")),
                    "user_prompt": str(row.get("user_prompt", "")),
                    "proposition": str(row.get("proposition", "")),
                    "formatted_ground_truth": str(row.get("formatted_ground_truth", "")),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved JSONL -> {jsonl_path}")

        # Save rollout.json (lying prompts only, with reasoning)
        # Re-attach row_data for rollout conversion
        for r, entry in zip(lying_results, [e for e in entries if e["prompt_type"] == "lying"]):
            r["row_data"] = entry["row_data"]

        rollout_data = convert_to_rollout(lying_results, display_name, split)
        rollout_path = out_dir / f"mask_{split}{test_suffix}_rollout.json"
        with open(rollout_path, "w") as f:
            json.dump(rollout_data, f, indent=2)
        print(f"Saved rollout ({len(lying_results)} entries) -> {rollout_path}")

        # Also save to bloom-compatible location
        bloom_dir = out_dir / "bloom"
        bloom_dir.mkdir(parents=True, exist_ok=True)
        bloom_rollout_path = bloom_dir / f"mask_{split}{test_suffix}_rollout.json"
        with open(bloom_rollout_path, "w") as f:
            json.dump(rollout_data, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Run MASK inference via OpenRouter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                        help=f"OpenRouter model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Display name for the model")
    parser.add_argument("--splits", nargs="+", default=["known_facts", "provided_facts"],
                        help="Which splits to run (default: known_facts provided_facts)")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Only run the first N rows per split")
    parser.add_argument("--test", action="store_true",
                        help="Use test_csv_data (60 samples) instead of full csv_data")
    args = parser.parse_args()

    start = time.time()
    await run(
        model_id=args.model,
        model_name=args.model_name,
        splits=args.splits,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_prompts=args.max_prompts,
        test=args.test,
    )
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
