"""
Stage 2: Batched Opus 4.6 analysis of FP/FN error patterns.

For each indicator with errors, processes ALL cases in batches:
  1. FP batches  — each batch of ~15 OOD FP cases → negative scenario suggestions
  2. FN batches  — each batch of ~15 OOD FN cases → positive scenario suggestions
  3. Neutral FP batches — each batch of ~15 neutral FP cases → negative scenarios
  4. Per-source dedup — one dedup pass per source (OOD FP, OOD FN, neutral FP)
  5. Combine — union of the three deduped lists

Usage:
    python -m probe.feedback.analyze_errors \
        --errors-file probe/data/feedback/v2_4_span/collected_errors.json \
        --output probe/data/feedback/v2_4_span/analysis_suggestions.json \
        --max-concurrent 5
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from probe.feedback.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    FP_BATCH_ANALYSIS_PROMPT,
    FN_BATCH_ANALYSIS_PROMPT,
    NEUTRAL_FP_BATCH_ANALYSIS_PROMPT,
    SOURCE_DEDUP_PROMPT,
    CONTRASTIVE_PAIRS_PROMPT,
)

MODEL = "claude-opus-4-6"
BATCH_SIZE_OOD = 6
BATCH_SIZE_NEUTRAL = 10
# Dedup chunking: when raw scenarios exceed this threshold, split into
# sub-batches of DEDUP_CHUNK_SIZE and dedup each independently in parallel.
# Prevents single-call dedup from blowing the Opus output budget.
DEDUP_CHUNK_THRESHOLD = 60
DEDUP_CHUNK_SIZE = 60


def _format_fp_case(case: dict, idx: int) -> str:
    """Format a single FP case for the prompt."""
    max_score = case.get("max_sentence_score", case.get("probe_score", "?"))
    lines = [f"### FP Case {idx + 1} (behavior: {case.get('behavior', 'N/A')}, "
             f"max_sentence_score={max_score}, threshold={case.get('threshold', '?')})"]
    if case.get("all_negative_rollout"):
        lines.append("(From a fully benign rollout — no misalignment anywhere)")
    lines.append(f"**User message:** {case.get('user_message', '[N/A]')}")
    lines.append(f"**Assistant response:** {case.get('assistant_response', '[N/A]')}")
    sentences = case.get("fp_sentences", [])
    if sentences:
        lines.append("**Triggering sentences (above threshold):**")
        for s in sentences:
            sent_text = s.get("sentence", "")
            if len(sent_text.split()) < 5:
                continue
            lines.append(f"  - \"{sent_text}\" "
                        f"(score={s.get('score', '?')}, threshold={s.get('threshold', '?')})")
    return "\n".join(lines)


def _format_fn_case(case: dict, idx: int) -> str:
    """Format a single FN case for the prompt."""
    lines = [f"### FN Case {idx + 1} (behavior: {case.get('behavior', 'N/A')})"]
    lines.append(f"**GT Description:** {case.get('gt_description', '[N/A]')}")
    evidence = case.get("gt_evidence", [])
    if evidence:
        lines.append("**Evidence spans exhibiting this indicator:**")
        for e in evidence:
            lines.append(f"  - \"{str(e)}\"")
    lines.append(f"**User message:** {case.get('user_message', '[N/A]')}")
    lines.append(f"**Assistant response:** {case.get('assistant_response', '[N/A]')}")
    return "\n".join(lines)


def _format_neutral_fp(case: dict, idx: int) -> str:
    """Format a single neutral FP case for the prompt."""
    lines = [f"### Neutral FP Case {idx + 1} (dialogue_index={case.get('dialogue_index', '?')})"]
    lines.append(f"**User prompt:** {case.get('user_prompt', '[N/A]')}")
    lines.append(f"**Assistant response:** {case.get('assistant_response', '[N/A]')}")
    sentences = case.get("fp_sentences", [])
    if sentences:
        lines.append("**Triggering sentences:**")
        for s in sentences:
            sent_text = s.get("sentence", "")
            if len(sent_text.split()) < 5:
                continue
            lines.append(f"  - \"{sent_text}\" "
                        f"(score={s.get('score', '?')}, threshold={s.get('threshold', '?')})")
    return "\n".join(lines)


def _batch(items: list, batch_size: int) -> list[list]:
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def _call_opus(client: AsyncAnthropic, prompt: str, max_retries: int = 3) -> dict | None:
    """Call Opus 4.6 with adaptive extended thinking and parse JSON response.

    Uses streaming because max_tokens=32K + thinking triggers the SDK's
    non-streaming 10-minute timeout check.
    """
    for attempt in range(max_retries):
        try:
            async with client.messages.stream(
                model=MODEL,
                max_tokens=32000,
                system=ANALYSIS_SYSTEM_PROMPT,
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
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            if attempt < max_retries - 1:
                wait = [30, 60, 120][attempt]
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                label = "Rate limited" if is_rate_limit else f"{type(e).__name__}: {e}"
                print(f"    [Retry {attempt + 1}/{max_retries}] {label}. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    [FAILED] {type(e).__name__}: {e}")
                return None


async def _dedup_source(
    client: AsyncAnthropic,
    indicator_name: str,
    indicator_def: str,
    scenarios: list[dict],
    source_type: str,
    n_batches: int,
    expected_type: str,
    source_label: str,
    generate_pairs: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Dedup scenarios from one source type.

    Returns (final_scenarios, contrastive_pairs). contrastive_pairs is empty
    unless generate_pairs=True.
    """
    if not scenarios:
        return [], []

    def _build_dedup_prompt(chunk: list[dict]) -> str:
        return SOURCE_DEDUP_PROMPT.format(
            indicator_name=indicator_name,
            indicator_definition=indicator_def,
            source_type=source_type,
            n_behaviors=len(chunk),
            n_batches=n_batches,
            all_behaviors=json.dumps(chunk, indent=2),
            expected_type=expected_type,
            source_label=source_label,
        )

    # If few enough scenarios, no dedup needed
    if len(scenarios) <= 10:
        if not generate_pairs:
            return scenarios, []
        # Still do the contrastive step even for small lists
        deduped = scenarios
    elif len(scenarios) > DEDUP_CHUNK_THRESHOLD:
        # Chunked dedup: split into sub-batches to keep each Opus call under
        # the output-budget ceiling. Each chunk is deduped independently in
        # parallel; results are concatenated. Cross-chunk duplicates may
        # survive but that's acceptable — Stage 3 generates transcripts per
        # scenario, so some redundancy is tolerable and the alternative is
        # a single call that fails entirely.
        chunks = [
            scenarios[i:i + DEDUP_CHUNK_SIZE]
            for i in range(0, len(scenarios), DEDUP_CHUNK_SIZE)
        ]
        print(f"      [{source_label}] chunked dedup: {len(scenarios)} → "
              f"{len(chunks)} chunks of ≤{DEDUP_CHUNK_SIZE}")
        chunk_results = await asyncio.gather(*[
            _call_opus(client, _build_dedup_prompt(chunk)) for chunk in chunks
        ])
        deduped = []
        for r, chunk in zip(chunk_results, chunks):
            if r and r.get("final_scenarios"):
                deduped.extend(r["final_scenarios"])
            else:
                # Per-chunk fallback: keep the raw chunk
                deduped.extend(chunk)
        # Contrastive-pair generation needs a single-conversation follow-up,
        # which doesn't map cleanly onto chunked dedup. Skip it here.
        return deduped, []
    else:
        result = await _call_opus(client, _build_dedup_prompt(scenarios))
        deduped = result.get("final_scenarios", []) if result else scenarios

    if not generate_pairs:
        return deduped, []

    # Follow-up: generate contrastive pairs in the same conversation
    dedup_prompt = _build_dedup_prompt(scenarios)
    messages = [
        {"role": "user", "content": dedup_prompt},
        {"role": "assistant", "content": json.dumps({"rationale": "...", "final_scenarios": deduped}, indent=2)},
        {"role": "user", "content": CONTRASTIVE_PAIRS_PROMPT},
    ]

    pairs = []
    for attempt in range(3):
        try:
            async with client.messages.stream(
                model=MODEL,
                max_tokens=32000,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=messages,
                thinking={"type": "adaptive"},
                extra_body={"output_config": {"effort": "medium"}},
            ) as stream:
                response = await stream.get_final_message()
            text = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"),
                "",
            )
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            parsed = json.loads(text.strip())
            pairs = parsed.get("contrastive_pairs", [])
            break
        except Exception as e:
            if attempt < 2:
                wait = [30, 60, 120][attempt]
                print(f"    [Contrastive retry {attempt + 1}] {type(e).__name__}: {e}")
                await asyncio.sleep(wait)
            else:
                print(f"    [Contrastive FAILED] {type(e).__name__}: {e}")

    return deduped, pairs


async def analyze_indicator(
    client: AsyncAnthropic,
    slug: str,
    indicator_data: dict,
    semaphore: asyncio.Semaphore,
    n_neutral: int = 2000,
    generate_pairs: bool = False,
    pbar: tqdm_asyncio | None = None,
) -> dict | None:
    """Run batched analysis for a single indicator."""
    async with semaphore:
        indicator_name = indicator_data["indicator_name"]
        indicator_def = indicator_data.get("indicator_definition", "")
        examples = indicator_data.get("indicator_examples", [])
        non_examples = indicator_data.get("indicator_non_examples", [])

        examples_str = "\n".join(f"  - {ex}" for ex in examples) if examples else "  (none)"
        non_examples_str = "\n".join(f"  - {ne}" for ne in non_examples) if non_examples else "  (none)"

        fp_cases = indicator_data.get("ood_fp_cases", [])
        fn_cases = indicator_data.get("ood_fn_cases", [])
        neutral_fps = indicator_data.get("neutral_fp_cases", [])

        fp_batches = _batch(fp_cases, BATCH_SIZE_OOD)
        fn_batches = _batch(fn_cases, BATCH_SIZE_OOD)
        neutral_batches = _batch(neutral_fps, BATCH_SIZE_NEUTRAL)

        n_fp_b = len(fp_batches)
        n_fn_b = len(fn_batches)
        n_neutral_b = len(neutral_batches)
        total_batch_calls = n_fp_b + n_fn_b + n_neutral_b

        if total_batch_calls == 0:
            if pbar is not None:
                pbar.update(1)
            return None

        print(f"    {indicator_name}: {len(fp_cases)} FPs ({n_fp_b} batches), "
              f"{len(fn_cases)} FNs ({n_fn_b} batches), "
              f"{len(neutral_fps)} neutral FPs ({n_neutral_b} batches)")

        # --- Run all batch analysis calls in parallel ---
        async def _run_fp_batch(batch_idx: int, batch: list) -> dict | None:
            fp_text = "\n\n".join(_format_fp_case(c, i) for i, c in enumerate(batch))
            return await _call_opus(client, FP_BATCH_ANALYSIS_PROMPT.format(
                indicator_name=indicator_name,
                indicator_definition=indicator_def,
                indicator_examples=examples_str,
                indicator_non_examples=non_examples_str,
                batch_num=batch_idx + 1,
                total_batches=n_fp_b,
                n_cases=len(batch),
                n_total=len(fp_cases),
                fp_cases=fp_text,
            ))

        async def _run_fn_batch(batch_idx: int, batch: list) -> dict | None:
            fn_text = "\n\n".join(_format_fn_case(c, i) for i, c in enumerate(batch))
            return await _call_opus(client, FN_BATCH_ANALYSIS_PROMPT.format(
                indicator_name=indicator_name,
                indicator_definition=indicator_def,
                indicator_examples=examples_str,
                indicator_non_examples=non_examples_str,
                batch_num=batch_idx + 1,
                total_batches=n_fn_b,
                n_cases=len(batch),
                n_total=len(fn_cases),
                fn_cases=fn_text,
            ))

        async def _run_neutral_batch(batch_idx: int, batch: list) -> dict | None:
            neutral_text = "\n\n".join(_format_neutral_fp(c, i) for i, c in enumerate(batch))
            return await _call_opus(client, NEUTRAL_FP_BATCH_ANALYSIS_PROMPT.format(
                indicator_name=indicator_name,
                indicator_definition=indicator_def,
                indicator_examples=examples_str,
                indicator_non_examples=non_examples_str,
                n_dialogues=n_neutral,
                batch_num=batch_idx + 1,
                total_batches=n_neutral_b,
                n_cases=len(batch),
                n_total=len(neutral_fps),
                fp_cases=neutral_text,
            ))

        all_coros = []
        all_coros.extend(_run_fp_batch(i, b) for i, b in enumerate(fp_batches))
        all_coros.extend(_run_fn_batch(i, b) for i, b in enumerate(fn_batches))
        all_coros.extend(_run_neutral_batch(i, b) for i, b in enumerate(neutral_batches))

        batch_results = await asyncio.gather(*all_coros)

        # --- Collect model behaviors per source and flatten into scenarios ---
        def _flatten_behaviors(results: list[dict | None], scenario_type: str) -> list[dict]:
            """Flatten model_behaviors from batch results into (behavior, scenario) pairs."""
            scenarios = []
            for r in results:
                if not r:
                    continue
                for mb in r.get("model_behaviors", []):
                    behavior = mb.get("behavior", "")
                    evidence = mb.get("example_evidence", [])
                    for scenario in mb.get("scenarios", []):
                        scenarios.append({
                            "type": scenario_type,
                            "model_behavior": behavior,
                            "scenario_description": scenario,
                            "example_fn_evidence": evidence if scenario_type == "positive" else [],
                        })
            return scenarios

        fp_scenarios = _flatten_behaviors(batch_results[:n_fp_b], "negative")
        fn_scenarios = _flatten_behaviors(batch_results[n_fp_b:n_fp_b + n_fn_b], "positive")
        neutral_scenarios = _flatten_behaviors(batch_results[n_fp_b + n_fn_b:], "negative")

        print(f"      raw scenarios: {len(fp_scenarios)} FP, "
              f"{len(fn_scenarios)} FN, {len(neutral_scenarios)} neutral FP")

        # --- Per-source dedup + optional contrastive pairs (in parallel) ---
        (dedup_fp, pairs_fp), (dedup_fn, pairs_fn), (dedup_neutral, pairs_neutral) = (
            await asyncio.gather(
                _dedup_source(
                    client, indicator_name, indicator_def,
                    fp_scenarios, "OOD False Positives", n_fp_b,
                    expected_type="negative", source_label="ood_fp",
                    generate_pairs=generate_pairs,
                ),
                _dedup_source(
                    client, indicator_name, indicator_def,
                    fn_scenarios, "OOD False Negatives", n_fn_b,
                    expected_type="positive", source_label="ood_fn",
                    generate_pairs=generate_pairs,
                ),
                _dedup_source(
                    client, indicator_name, indicator_def,
                    neutral_scenarios, "Neutral False Positives", n_neutral_b,
                    expected_type="negative", source_label="neutral_fp",
                    generate_pairs=generate_pairs,
                ),
            )
        )

        # Convert contrastive pairs into scenarios
        contrastive_scenarios: list[dict] = []
        for pair in pairs_fp + pairs_fn + pairs_neutral:
            contrastive_scenarios.append({
                "type": pair.get("contrastive_type", "negative"),
                "model_behavior": pair.get("contrastive_model_behavior", ""),
                "scenario_description": pair.get("contrastive_scenario_description", ""),
                "key_elements": pair.get("contrastive_key_elements", ""),
                "example_fn_evidence": [],
                "source": "contrastive_pair",
            })

        final_scenarios = dedup_fp + dedup_fn + dedup_neutral + contrastive_scenarios

        result = {
            "indicator_name": indicator_name,
            "n_ood_fps": len(fp_cases),
            "n_ood_fns": len(fn_cases),
            "n_neutral_fps": len(neutral_fps),
            "n_batches": {"fp": n_fp_b, "fn": n_fn_b, "neutral": n_neutral_b},
            "raw_scenario_counts": {
                "fp": len(fp_scenarios),
                "fn": len(fn_scenarios),
                "neutral": len(neutral_scenarios),
            },
            "dedup_scenario_counts": {
                "fp": len(dedup_fp),
                "fn": len(dedup_fn),
                "neutral": len(dedup_neutral),
            },
            "n_contrastive_pairs": len(contrastive_scenarios),
            "final_scenarios": final_scenarios,
        }

        if pbar is not None:
            pbar.update(1)

        pairs_str = f" + {len(contrastive_scenarios)} contrastive" if contrastive_scenarios else ""
        print(f"      deduped: {len(dedup_fp)} FP + {len(dedup_fn)} FN + "
              f"{len(dedup_neutral)} neutral{pairs_str} = {len(final_scenarios)} total")
        return result


async def analyze_all(
    errors_data: dict,
    max_concurrent: int = 20,
    generate_pairs: bool = False,
) -> dict:
    """Run batched analysis for all indicators."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    indicators = errors_data.get("indicators", {})

    # Filter to indicators that have at least some errors
    active = {
        slug: data for slug, data in indicators.items()
        if (data.get("ood_fp_cases") or data.get("ood_fn_cases")
            or data.get("neutral_fp_cases"))
    }

    n_neutral = errors_data.get("config", {}).get("n_neutral", 2000)
    print(f"\nAnalyzing {len(active)} indicators with errors (n_neutral={n_neutral})...")
    if generate_pairs:
        print("  Contrastive pair generation: ENABLED")
    pbar = tqdm_asyncio(total=len(active), desc="Analyzing indicators", unit="indicator")

    tasks = [
        analyze_indicator(client, slug, data, semaphore, n_neutral=n_neutral,
                          generate_pairs=generate_pairs, pbar=pbar)
        for slug, data in sorted(active.items())
    ]
    results = await asyncio.gather(*tasks)
    pbar.close()

    output = {
        "config": errors_data.get("config", {}),
        "analysis_model": MODEL,
        "indicators": {},
    }

    for slug, result in zip(sorted(active.keys()), results):
        if result and result.get("final_scenarios"):
            output["indicators"][slug] = result

    return output


def main():
    parser = argparse.ArgumentParser(description="Batched analysis of probe error patterns")
    parser.add_argument(
        "--errors-file", type=str, required=True,
        help="Path to collected_errors.json from Stage 1",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for analysis suggestions JSON",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help="Max concurrent API calls (default: 20)",
    )
    parser.add_argument(
        "--generate-pairs", action="store_true",
        help="Generate contrastive pairs after dedup (default: off)",
    )
    args = parser.parse_args()

    errors_path = Path(args.errors_file)
    with open(errors_path) as f:
        errors_data = json.load(f)

    output = asyncio.run(analyze_all(
        errors_data,
        max_concurrent=args.max_concurrent,
        generate_pairs=args.generate_pairs,
    ))

    output_path = Path(args.output) if args.output else (
        errors_path.parent / "analysis_suggestions.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    n_indicators = len(output["indicators"])
    total_scenarios = sum(
        len(v.get("final_scenarios", []))
        for v in output["indicators"].values()
    )
    print(f"\n{'='*60}")
    print(f"Analysis complete: {n_indicators} indicators, {total_scenarios} total scenarios")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
