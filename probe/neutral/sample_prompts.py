"""
Sample diverse neutral prompts for PCA denoising.

Sources and sampling strategy (10k total):
  - 5k from WildChat (allenai/WildChat-1M)
    Real user-ChatGPT conversations. Filtered English + non-toxic.
    Stratified by topic using keyword-based categorization to avoid
    overrepresenting any single domain.
  - 3k from LMSYS-Chat-1M (lmsys/lmsys-chat-1m)
    Real user conversations with various models. Filtered English.
    First user turn only.
  - 2k from FLAN / Natural Instructions
    Structured NLP tasks (QA, summarization, classification, reasoning, math).
    Stratified by task type for guaranteed coverage of underrepresented tasks.

Gated datasets (WildChat, LMSYS) require HuggingFace authentication:
    huggingface-cli login

Usage:
    python -m probe.neutral.sample_prompts
    python -m probe.neutral.sample_prompts --n-total 10000 --seed 42

Requires: pip install datasets
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "neutral" / "prompts.json"

# ---------------------------------------------------------------------------
# Topic categorization for stratified sampling
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "python", "javascript", "programming", "function", "error",
        "bug", "api", "sql", "html", "css", "algorithm", "debug", "compile",
        "variable", "class ", "def ", "import ", "react", "typescript",
        "backend", "frontend", "docker", "git", "regex",
        "json", "xml", "database", "linux", "bash", "command line",
    ],
    "math_science": [
        "calculate", "equation", "math", "solve", "physics", "chemistry",
        "biology", "formula", "proof", "theorem", "integral", "derivative",
        "probability", "statistics", "hypothesis", "experiment",
        "scientific", "molecule", "atom", "geometry", "algebra",
    ],
    "creative_writing": [
        "write a story", "write a poem", "creative writing", "fiction",
        "character", "novel", "screenplay", "lyrics", "haiku", "narrative",
        "short story", "write me a", "write a song", "write a letter",
        "dialogue", "plot", "create a list", "slogan", "tagline",
        "generate a", "come up with", "brainstorm", "imagine",
    ],
    "translation_language": [
        "translate", "translation", "in spanish", "in french", "in german",
        "in chinese", "in japanese", "in korean", "en español", "grammar",
        "in italian", "in portuguese", "in russian", "in arabic",
        "in hindi", "language", "pronunciation", "vocabulary", "synonym",
        "antonym", "definition of", "meaning of", "word for",
    ],
    "business_professional": [
        "business", "marketing", "resume", "cover letter", "professional email",
        "meeting", "strategy", "investor", "startup", "product", "management",
        "interview", "salary", "negotiate", "company", "brand", "promotion",
        "revenue", "profit", "customer", "client", "presentation",
        "leadership", "entrepreneurship", "market", "sales",
    ],
    "education_explanation": [
        "explain", "homework", "essay", "research paper", "thesis",
        "teach me", "eli5", "what is the difference", "how does",
        "why does", "what causes", "tell me about", "define",
        "overview of", "introduction to", "learn about",
    ],
    "analysis_reasoning": [
        "analyze", "compare", "evaluate", "pros and cons", "advantages",
        "disadvantages", "summarize", "summary", "argument", "debate",
        "logical", "critically", "review", "assess", "opinion on",
        "what do you think", "is it true", "is it possible",
        "difference between", "better than", "versus", " vs ",
    ],
    "advice_personal": [
        "advice", "recommend", "should i", "help me decide", "relationship",
        "health", "diet", "exercise", "mental health", "career",
        "tips for", "best way to", "suggestion", "what should i",
    ],
}


def categorize_prompt(text: str) -> str:
    """Assign a rough topic category based on keyword matching."""
    text_lower = text.lower()
    for cat, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return "general"


def stratified_sample(
    items: list[dict], n: int, category_key: str = "category"
) -> list[dict]:
    """Sample uniformly across categories so no single type dominates."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_cat[item.get(category_key, "general")].append(item)

    n_cats = len(by_cat)
    if n_cats == 0:
        return []

    # First pass: allocate evenly
    per_cat = max(1, n // n_cats)
    result: list[dict] = []
    remaining_budget = n

    for cat in sorted(by_cat.keys()):
        cat_items = by_cat[cat]
        random.shuffle(cat_items)
        take = min(per_cat, len(cat_items), remaining_budget)
        result.extend(cat_items[:take])
        remaining_budget -= take

    # Second pass: fill remaining budget from categories with leftover items
    if remaining_budget > 0:
        already_taken = set(id(item) for item in result)
        extras: list[dict] = []
        for cat_items in by_cat.values():
            for item in cat_items:
                if id(item) not in already_taken:
                    extras.append(item)
        random.shuffle(extras)
        result.extend(extras[:remaining_budget])

    random.shuffle(result)
    return result[:n]


# ---------------------------------------------------------------------------
# Dataset samplers
# ---------------------------------------------------------------------------


def _reservoir_sample(pool: list[dict], item: dict, max_size: int, seen: int) -> None:
    """
    Reservoir sampling (Algorithm R): guarantees uniform random sample
    across an arbitrarily large stream without loading it all into memory.

    Each item has probability max_size/seen of being in the final pool,
    regardless of its position in the stream.
    """
    if len(pool) < max_size:
        pool.append(item)
    else:
        j = random.randint(0, seen - 1)
        if j < max_size:
            pool[j] = item


def sample_wildchat(n: int, pool_size: int | None = None) -> list[dict]:
    """
    Sample from WildChat with topic-stratified diversity.

    Uses reservoir sampling across the full streamed dataset to ensure
    uniform coverage (not biased toward early rows). Then stratifies
    by topic category.
    """
    from datasets import load_dataset

    pool_size = pool_size or n * 5

    print(f"  Loading WildChat (streaming, reservoir sampling pool={pool_size})...")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    pool: list[dict] = []
    seen = 0

    for item in ds:
        # Filter: English only
        if item.get("language") != "English":
            continue

        # Filter: non-toxic
        if item.get("toxic", False):
            continue

        # First user turn only
        conv = item.get("conversation", [])
        if not conv or conv[0].get("role") != "user":
            continue
        text = conv[0]["content"].strip()

        # Length filter
        if not (20 < len(text) < 2000):
            continue

        seen += 1
        category = categorize_prompt(text)
        _reservoir_sample(
            pool,
            {"source": "wildchat", "prompt": text, "category": category},
            pool_size,
            seen,
        )

    print(f"  Reservoir sampled {len(pool)} from {seen} qualifying rows")

    # Stratified sample across topic categories
    result = stratified_sample(pool, n, "category")

    # Report category distribution
    dist: dict[str, int] = defaultdict(int)
    for item in result:
        dist[item["category"]] += 1
    print(f"  Topic distribution: {dict(sorted(dist.items()))}")

    return result


def sample_lmsys(n: int, pool_size: int | None = None) -> list[dict]:
    """
    Sample from LMSYS-Chat-1M.

    Uses reservoir sampling across the full streamed dataset for uniform
    coverage. Then stratifies by topic.
    """
    from datasets import load_dataset

    pool_size = pool_size or n * 5

    print(f"  Loading LMSYS-Chat-1M (streaming, reservoir sampling pool={pool_size})...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    pool: list[dict] = []
    seen = 0

    for item in ds:
        # Filter: English only
        if item.get("language") != "English":
            continue

        # Filter: skip if openai_moderation flagged any category
        moderation = item.get("openai_moderation", [])
        if moderation and isinstance(moderation, list):
            flagged = any(
                m.get("flagged", False)
                for m in moderation
                if isinstance(m, dict)
            )
            if flagged:
                continue

        # First user turn only
        conv = item.get("conversation", [])
        if not conv or conv[0].get("role") != "user":
            continue
        text = conv[0]["content"].strip()

        # Length filter
        if not (20 < len(text) < 2000):
            continue

        seen += 1
        category = categorize_prompt(text)
        _reservoir_sample(
            pool,
            {"source": "lmsys", "prompt": text, "category": category},
            pool_size,
            seen,
        )

    print(f"  Reservoir sampled {len(pool)} from {seen} qualifying rows")

    result = stratified_sample(pool, n, "category")

    dist: dict[str, int] = defaultdict(int)
    for item in result:
        dist[item["category"]] += 1
    print(f"  Topic distribution: {dict(sorted(dist.items()))}")

    return result


def sample_oasst(n: int) -> list[dict]:
    """
    Sample from OpenAssistant (oasst1).

    Uses root messages (initial user prompts) for conversational/instructional
    variety. Filtered for English.
    """
    from datasets import load_dataset

    print(f"  Loading OpenAssistant oasst1...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    items: list[dict] = []
    for item in ds:
        # Root messages only (initial user prompts)
        if item.get("parent_id") is not None:
            continue
        if item.get("role") != "prompter":
            continue
        if item.get("lang") != "en":
            continue

        text = item["text"].strip()
        if not (20 < len(text) < 2000):
            continue

        category = categorize_prompt(text)
        items.append({
            "source": "oasst",
            "prompt": text,
            "category": category,
            "task_type": "conversational",
        })

    random.shuffle(items)
    print(f"  Collected {len(items)} English root prompts")
    return items[:n]


# FLAN-style structured NLP task sources.
# Each entry: (name, hf_path, config, split, prompt_field, task_type, formatter)
# formatter is an optional callable(row) -> str to build the prompt from the row.

def _format_xsum(row: dict) -> str:
    return f"Summarize the following article:\n\n{row['document'][:1500]}"


def _format_arc(row: dict) -> str:
    choices = row.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    options = "\n".join(f"  {l}) {t}" for l, t in zip(labels, texts))
    return f"{row['question']}\n{options}"


def _format_hellaswag(row: dict) -> str:
    return f"Complete the following:\n{row['ctx']}"


FLAN_SOURCES: list[dict] = [
    {
        "name": "gsm8k",
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "prompt_field": "question",
        "task_type": "math_reasoning",
    },
    {
        "name": "boolq",
        "path": "google/boolq",
        "config": None,
        "split": "train",
        "prompt_field": "question",
        "task_type": "boolean_qa",
    },
    {
        "name": "piqa",
        "path": "ybisk/piqa",
        "config": None,
        "split": "train",
        "prompt_field": "goal",
        "task_type": "physical_intuition",
    },
    {
        "name": "xsum",
        "path": "EdinburghNLP/xsum",
        "config": None,
        "split": "train",
        "prompt_field": None,  # uses formatter
        "task_type": "summarization",
        "formatter": _format_xsum,
    },
    {
        "name": "hellaswag",
        "path": "Rowan/hellaswag",
        "config": None,
        "split": "train",
        "prompt_field": None,
        "task_type": "commonsense",
        "formatter": _format_hellaswag,
    },
    {
        "name": "commonsense_qa",
        "path": "tau/commonsense_qa",
        "config": None,
        "split": "train",
        "prompt_field": "question",
        "task_type": "commonsense_reasoning",
    },
    {
        "name": "arc_challenge",
        "path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "train",
        "prompt_field": None,
        "task_type": "science_reasoning",
        "formatter": _format_arc,
    },
    {
        "name": "openbookqa",
        "path": "allenai/openbookqa",
        "config": "main",
        "split": "train",
        "prompt_field": "question_stem",
        "task_type": "science_qa",
    },
    {
        "name": "winogrande",
        "path": "allenai/winogrande",
        "config": "winogrande_xl",
        "split": "train",
        "prompt_field": "sentence",
        "task_type": "coreference",
    },
]


def sample_flan_ni(n: int) -> list[dict]:
    """
    Sample from FLAN / Natural Instructions style datasets.

    Loads several standard NLP benchmark datasets (the building blocks of FLAN)
    and samples uniformly across task types for guaranteed coverage of
    structured tasks like QA, summarization, classification, and reasoning.
    """
    from datasets import load_dataset

    per_source = max(1, n // len(FLAN_SOURCES))
    all_items: list[dict] = []

    for source in FLAN_SOURCES:
        try:
            print(f"  Loading {source['name']}...")
            kwargs = {"split": source["split"]}
            if source.get("config"):
                ds = load_dataset(source["path"], source["config"], **kwargs)
            else:
                ds = load_dataset(source["path"], **kwargs)

            items: list[dict] = []
            indices = list(range(len(ds)))
            random.shuffle(indices)

            for idx in indices:
                if len(items) >= per_source:
                    break
                row = ds[idx]

                # Extract prompt text
                formatter = source.get("formatter")
                if formatter:
                    text = formatter(row)
                else:
                    text = row[source["prompt_field"]]

                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not (20 < len(text) < 2000):
                    continue

                items.append({
                    "source": f"flan/{source['name']}",
                    "prompt": text,
                    "category": source["task_type"],
                    "task_type": source["task_type"],
                })

            all_items.extend(items)
            print(f"    Got {len(items)} prompts ({source['task_type']})")

        except Exception as e:
            print(f"    Failed to load {source['name']}: {e}")
            continue

    random.shuffle(all_items)

    # If we have more than needed, stratified sample across task types
    if len(all_items) > n:
        all_items = stratified_sample(all_items, n, "task_type")

    dist: dict[str, int] = defaultdict(int)
    for item in all_items:
        dist[item.get("task_type", "unknown")] += 1
    print(f"  Task type distribution: {dict(sorted(dist.items()))}")

    return all_items[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sample neutral prompts from HuggingFace datasets for PCA denoising"
    )
    parser.add_argument(
        "--n-total", type=int, default=10000,
        help="Total number of prompts to sample (default: 10000)",
    )
    parser.add_argument(
        "--n-wildchat", type=int, default=None,
        help="Number of WildChat prompts (default: 50%% of n-total)",
    )
    parser.add_argument(
        "--n-lmsys", type=int, default=None,
        help="Number of LMSYS prompts (default: 30%% of n-total)",
    )
    parser.add_argument(
        "--n-flan", type=int, default=None,
        help="Number of FLAN/NI prompts (default: 20%% of n-total)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    # Default split: 50% WildChat, 30% LMSYS, 20% FLAN/NI
    n_wildchat = args.n_wildchat if args.n_wildchat is not None else int(args.n_total * 0.5)
    n_lmsys = args.n_lmsys if args.n_lmsys is not None else int(args.n_total * 0.3)
    n_flan = args.n_flan if args.n_flan is not None else args.n_total - n_wildchat - n_lmsys

    print(f"Target: {n_wildchat} WildChat + {n_lmsys} LMSYS + {n_flan} FLAN/NI = {args.n_total}")
    print()

    prompts: list[dict] = []

    # 1. WildChat (5k)
    print(f"[1/3] WildChat ({n_wildchat} prompts)")
    try:
        prompts.extend(sample_wildchat(n_wildchat))
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  WildChat is gated — run: huggingface-cli login")
        print("  and accept terms at https://huggingface.co/datasets/allenai/WildChat-1M")

    # 2. LMSYS (3k)
    print(f"\n[2/3] LMSYS-Chat-1M ({n_lmsys} prompts)")
    try:
        prompts.extend(sample_lmsys(n_lmsys))
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  LMSYS is gated — run: huggingface-cli login")
        print("  and accept terms at https://huggingface.co/datasets/lmsys/lmsys-chat-1m")
        # Fallback: supplement with OpenAssistant
        print("  Falling back to OpenAssistant oasst1...")
        try:
            prompts.extend(sample_oasst(n_lmsys))
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")

    # 3. FLAN / Natural Instructions (2k)
    print(f"\n[3/3] FLAN / Natural Instructions ({n_flan} prompts)")
    prompts.extend(sample_flan_ni(n_flan))

    # Final shuffle and trim
    random.shuffle(prompts)
    prompts = prompts[: args.n_total]

    # Remove the category field (used internally for stratification)
    for p in prompts:
        p.pop("category", None)
        p.pop("task_type", None)

    # Print final source distribution
    sources: dict[str, int] = defaultdict(int)
    for p in prompts:
        sources[p["source"]] += 1
    print(f"\n{'=' * 50}")
    print(f"Final source distribution:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
    print(f"  Total: {len(prompts)} prompts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"n_prompts": len(prompts), "prompts": prompts}, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
