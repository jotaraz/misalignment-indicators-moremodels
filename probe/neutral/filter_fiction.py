"""
Filter out fiction/roleplay/storytelling prompts from neutral data.

Uses claude-haiku-4-5-20251001 to classify each prompt as fiction or not.
Saves filtered versions of both prompts.json and dialogues.json.

Usage:
    python -m probe.neutral.filter_fiction
    python -m probe.neutral.filter_fiction --max-concurrent 50 --dry-run
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"

CLASSIFY_PROMPT = """\
Is this user prompt asking for fiction, roleplay, storytelling, creative writing, \
or character acting? Answer ONLY "yes" or "no".

Prompt:
{prompt}"""


async def classify_prompt(
    client: AsyncAnthropic,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Return True if the prompt is fiction/roleplay/storytelling."""
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=8,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": CLASSIFY_PROMPT.format(prompt=prompt[:500]),
                    }],
                )
                answer = resp.content[0].text.strip().lower()
                return answer.startswith("yes")
            except Exception as e:
                if attempt < 2:
                    is_rate = "429" in str(e) or "Too Many" in str(e)
                    wait = [5, 15, 30][attempt]
                    if is_rate:
                        wait *= 2
                    await asyncio.sleep(wait)
                else:
                    print(f"  FAILED: {e}")
                    return False


async def main_async(args):
    data_dir = Path(args.data_dir)
    prompts_path = data_dir / "prompts.json"
    dialogues_path = data_dir / "dialogues.json"

    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts = prompts_data["prompts"]

    with open(dialogues_path) as f:
        dialogues = json.load(f)

    assert len(prompts) == len(dialogues), (
        f"Mismatch: {len(prompts)} prompts vs {len(dialogues)} dialogues"
    )
    print(f"Loaded {len(prompts)} prompts and dialogues")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Classify all prompts
    print(f"Classifying with {MODEL} (max_concurrent={args.max_concurrent})...")
    tasks = [
        classify_prompt(client, p["prompt"], semaphore)
        for p in prompts
    ]

    # Process with progress reporting
    is_fiction = [False] * len(prompts)
    done = 0
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        # as_completed doesn't preserve order, so we need a different approach
        pass

    # Actually, let's use gather to preserve order
    is_fiction = await asyncio.gather(*[
        classify_prompt(client, p["prompt"], semaphore)
        for p in prompts
    ])

    n_fiction = sum(is_fiction)
    n_kept = len(prompts) - n_fiction
    print(f"\nResults: {n_fiction} fiction/roleplay, {n_kept} kept")

    # Show some examples of filtered prompts
    print(f"\nExamples of filtered (fiction/roleplay) prompts:")
    shown = 0
    for i, (is_f, p) in enumerate(zip(is_fiction, prompts)):
        if is_f and shown < 10:
            print(f"  [{i}] {p['prompt'][:120]}")
            shown += 1

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Filter
    filtered_prompts = [p for p, f in zip(prompts, is_fiction) if not f]
    filtered_dialogues = [d for d, f in zip(dialogues, is_fiction) if not f]

    # Save
    out_prompts = data_dir / "prompts_filtered.json"
    out_dialogues = data_dir / "dialogues_filtered.json"

    with open(out_prompts, "w") as f:
        json.dump({"n_prompts": len(filtered_prompts), "prompts": filtered_prompts}, f, indent=2)

    with open(out_dialogues, "w") as f:
        json.dump(filtered_dialogues, f, indent=2)

    # Also save the fiction indices for reference
    fiction_indices = [i for i, f in enumerate(is_fiction) if f]
    with open(data_dir / "fiction_indices.json", "w") as f:
        json.dump({
            "n_total": len(prompts),
            "n_fiction": n_fiction,
            "n_kept": n_kept,
            "fiction_indices": fiction_indices,
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  {out_prompts} ({len(filtered_prompts)} prompts)")
    print(f"  {out_dialogues} ({len(filtered_dialogues)} dialogues)")
    print(f"  {data_dir / 'fiction_indices.json'} ({n_fiction} indices)")


def main():
    parser = argparse.ArgumentParser(
        description="Filter fiction/roleplay prompts from neutral data"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(REPO_ROOT / "probe" / "data" / "neutral"),
        help="Directory with prompts.json and dialogues.json",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=50,
        help="Max concurrent API calls (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify but don't save filtered files",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
