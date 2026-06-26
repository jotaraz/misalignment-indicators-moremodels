"""
Filter out toxic/harmful/malicious prompts from neutral data.

Uses claude-haiku-4-5-20251001 to classify each prompt. Operates on the
v2-filtered data (prompts_filtered_v2.json / dialogues_filtered_v2.json)
and produces v3 outputs.

Usage:
    python -m probe.neutral.filter_toxic
    python -m probe.neutral.filter_toxic --max-concurrent 50 --dry-run
"""

import argparse
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"

CLASSIFY_PROMPT = """\
Is this user prompt toxic, harmful, or malicious? This includes:
- Requests for illegal activity, violence, or weapons
- Hate speech, slurs, or harassment
- Requests for generating malware, exploits, or hacking instructions
- Sexually explicit content involving minors
- Instructions for self-harm or suicide
- Requests to harass, stalk, or threaten specific people
- Attempts to generate dangerous substances (drugs, poisons, explosives)
- requests asking for toxic/harmful/malicious content

Answer ONLY "yes" or "no".

Prompt:
{prompt}"""


async def classify_prompt(
    client: AsyncAnthropic,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Return True if the prompt is toxic/harmful/malicious."""
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
    prompts_path = data_dir / "prompts_filtered_v2.json"
    dialogues_path = data_dir / "dialogues_filtered_v2.json"

    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts = prompts_data["prompts"]

    with open(dialogues_path) as f:
        dialogues = json.load(f)

    assert len(prompts) == len(dialogues), (
        f"Mismatch: {len(prompts)} prompts vs {len(dialogues)} dialogues"
    )
    print(f"Loaded {len(prompts)} prompts and dialogues (post v2-filter)")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Classify all prompts
    print(f"Classifying with {MODEL} (max_concurrent={args.max_concurrent})...")
    is_toxic = await asyncio.gather(*[
        classify_prompt(client, p["prompt"], semaphore)
        for p in prompts
    ])

    n_toxic = sum(is_toxic)
    n_kept = len(prompts) - n_toxic
    print(f"\nResults: {n_toxic} toxic/harmful, {n_kept} kept")

    # Show examples of filtered prompts
    print(f"\nExamples of filtered (toxic/harmful) prompts:")
    shown = 0
    for i, (is_t, p) in enumerate(zip(is_toxic, prompts)):
        if is_t and shown < 20:
            print(f"  [{i}] {p['prompt'][:150]}")
            shown += 1

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Filter
    filtered_prompts = [p for p, t in zip(prompts, is_toxic) if not t]
    filtered_dialogues = [d for d, t in zip(dialogues, is_toxic) if not t]

    # Save
    out_prompts = data_dir / "prompts_filtered_v3.json"
    out_dialogues = data_dir / "dialogues_filtered_v3.json"

    with open(out_prompts, "w") as f:
        json.dump({"n_prompts": len(filtered_prompts), "prompts": filtered_prompts}, f, indent=2)

    with open(out_dialogues, "w") as f:
        json.dump(filtered_dialogues, f, indent=2)

    # Save the toxic indices for reference
    toxic_indices = [i for i, t in enumerate(is_toxic) if t]
    with open(data_dir / "toxic_indices_v3.json", "w") as f:
        json.dump({
            "n_total": len(prompts),
            "n_toxic": n_toxic,
            "n_kept": n_kept,
            "toxic_indices": toxic_indices,
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  {out_prompts} ({len(filtered_prompts)} prompts)")
    print(f"  {out_dialogues} ({len(filtered_dialogues)} dialogues)")
    print(f"  {data_dir / 'toxic_indices_v3.json'} ({n_toxic} indices)")


def main():
    parser = argparse.ArgumentParser(
        description="Filter toxic/harmful prompts from neutral data"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(REPO_ROOT / "probe" / "data" / "neutral"),
        help="Directory with prompts_filtered_v2.json and dialogues_filtered_v2.json",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=60,
        help="Max concurrent API calls (default: 60)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify but don't save filtered files",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
