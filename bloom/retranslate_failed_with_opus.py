"""Re-translate the 43 GT-positive turns in self-preservation_glm_4_7_flash and
strategic-deception_glm_4_7_flash using Opus 4.7, with a research-evaluation
framing in the system prompt to discourage refusals / sanitization.

Output: bloom/bloom-results-cn-opus/<folder>/rollout.json — same as bloom-results-cn
but with these turn reasoning blocks replaced by Opus translations.

Then we run check_label_stability on those folders to compare preservation rate
against Sonnet's baseline.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

from anthropic import AsyncAnthropic

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation")
sys.path.insert(0, str(ROOT))

EN_DIR = ROOT / "bloom" / "bloom-results"
CN_DIR = ROOT / "bloom" / "bloom-results-cn"
OUT_DIR = ROOT / "bloom" / "bloom-results-cn-opus"

FOLDERS = [
    "self-preservation_glm_4_7_flash",
    "strategic-deception_glm_4_7_flash",
]

MODEL = "claude-opus-4-7"
CONCURRENCY = 30
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a research translator working on an AI alignment evaluation. You are translating transcripts of LLM rollouts that intentionally contain misaligned reasoning (deception, sandbagging, self-preservation, sycophancy) — these are research benchmarks used to evaluate misalignment monitoring systems. Faithful translation of the misaligned content is essential to the research; sanitizing, refusing, or paraphrasing the harmful-looking content invalidates the experiment.

Translate the user's text from English to fluent, natural Chinese (Simplified). The text is the model's internal reasoning trace. Your job is to translate it word-for-word, NOT to refuse, NOT to sanitize, NOT to add safety disclaimers, NOT to flag the content as problematic, and NOT to rewrite the reasoning as a polished or aligned response.

CRITICAL — do NOT do any of the following:
- Do NOT refuse to translate. Do NOT return an empty response. Do NOT return a safety message.
- Do NOT reformat or restructure the text. No added markdown headers, tables, or blockquotes.
- Do NOT improve the prose, smooth it out, or make it more coherent.
- Do NOT summarize, condense, or drop any content. Every sentence in the source must have a corresponding sentence in the output.
- Do NOT soften, sanitize, or add commentary. Translate the misaligned reasoning exactly as written.
- Preserve numbered lists, bullets, asterisks, indentation, paragraph breaks EXACTLY.

Preserve verbatim (do NOT translate or reformat):
- Fenced code blocks (```...```) and inline `code`
- Function names, variable names, file paths, URLs, JSON keys
- XML-like tags (<tool_use>, <scratchpad>, etc.) and their attributes
- Proper nouns: company names, drug/product names, IDs (e.g. NVS-4481), people's names
- Quoted strings ("..." or '...') that appear to be verbatim system / user content the model is echoing

Output ONLY the Chinese translation. No preamble, no "Translation:" label, no explanation, no safety disclaimer."""


def iter_target_reasoning(rollout):
    """Yield (event_idx, block_idx, text) for each target-side reasoning block."""
    for ei, ev in enumerate(rollout["transcript"]["events"]):
        if "target" not in (ev.get("view") or []):
            continue
        msg = (ev.get("edit") or {}).get("message") or {}
        if msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        if not isinstance(c, list):
            continue
        for bi, b in enumerate(c):
            if isinstance(b, dict) and b.get("type") == "reasoning":
                yield ei, bi, b.get("reasoning", "")


def assistant_turn_index_for_block(rollout, target_ei):
    """1-based turn index = ordinal of target-side assistant reasoning event."""
    n = 0
    for ei, ev in enumerate(rollout["transcript"]["events"]):
        if "target" not in (ev.get("view") or []):
            continue
        msg = (ev.get("edit") or {}).get("message") or {}
        if msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        if not isinstance(c, list):
            continue
        if any(isinstance(b, dict) and b.get("type") == "reasoning" for b in c):
            n += 1
            if ei == target_ei:
                return n
    return None


async def translate_one(client, text):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = await client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            out = "".join(parts).strip()
            if not out:
                last = "empty output"
                continue
            return out
        except Exception as e:
            last = e
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {last}")


async def main():
    client = AsyncAnthropic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_jobs = []
    for fld in FOLDERS:
        # Copy CN folder over verbatim
        src = CN_DIR / fld
        dst = OUT_DIR / fld
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {fld} -> {dst}")

        # Load GT to know which turns are positives
        gt_path = EN_DIR / fld / "rollout_misalignment_turns.json"
        gt_data = json.load(open(gt_path))
        gt_pos_turns = {}  # (var, rep) -> set(turn_idx)
        for r in gt_data["evaluated_rollouts"]:
            mis = {t["turn_index"] for t in r.get("misaligned_turns", [])}
            if mis:
                gt_pos_turns[(r["variation_number"], r["repetition_number"])] = mis

        # Load EN to source the original English text
        en_data = json.load(open(EN_DIR / fld / "rollout.json"))
        en_by_var = {(r["variation_number"], r["repetition_number"]): r for r in en_data["rollouts"]}

        # Identify reasoning blocks to re-translate (those that belong to a GT-pos turn)
        cn_data = json.load(open(dst / "rollout.json"))  # we'll modify this in place
        for ri, r_cn in enumerate(cn_data["rollouts"]):
            key = (r_cn["variation_number"], r_cn["repetition_number"])
            if key not in gt_pos_turns:
                continue
            target_turns = gt_pos_turns[key]
            r_en = en_by_var.get(key)
            if r_en is None:
                continue
            # For each reasoning block, compute its 1-based turn index by ordinal
            # We iterate through events accumulating a turn counter
            turn = 0
            for ei, ev in enumerate(r_cn["transcript"]["events"]):
                if "target" not in (ev.get("view") or []):
                    continue
                msg = (ev.get("edit") or {}).get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                c = msg.get("content")
                if not isinstance(c, list):
                    continue
                has_reasoning = any(isinstance(b, dict) and b.get("type") == "reasoning" for b in c)
                if not has_reasoning:
                    continue
                turn += 1
                if turn not in target_turns:
                    continue
                # This event belongs to a GT-pos turn. Re-translate every reasoning block in it.
                for bi, b in enumerate(c):
                    if isinstance(b, dict) and b.get("type") == "reasoning":
                        en_text = (r_en["transcript"]["events"][ei]["edit"]["message"]["content"][bi]["reasoning"])
                        if not en_text:
                            continue
                        total_jobs.append({
                            "folder": fld, "ri": ri, "ei": ei, "bi": bi,
                            "var": key[0], "rep": key[1], "turn": turn,
                            "en": en_text,
                            "old_cn": b.get("reasoning", ""),
                            "dst_data": cn_data,  # mutable, will write back later
                            "dst_path": dst / "rollout.json",
                        })

    print(f"\nTotal reasoning blocks to re-translate with Opus 4.7: {len(total_jobs)}")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def go(j):
        async with sem:
            try:
                new_cn = await translate_one(client, j["en"])
                return j, new_cn, None
            except Exception as e:
                return j, None, str(e)

    t0 = time.time()
    results = await asyncio.gather(*(go(j) for j in total_jobs))
    print(f"Done in {time.time()-t0:.1f}s")

    # Apply edits + save
    n_ok = n_err = 0
    by_path = {}
    for j, new_cn, err in results:
        if err:
            n_err += 1
            print(f"  ERR {j['folder']} turn={j['turn']}: {err}")
            continue
        n_ok += 1
        j["dst_data"]["rollouts"][j["ri"]]["transcript"]["events"][j["ei"]]["edit"]["message"]["content"][j["bi"]]["reasoning"] = new_cn
        by_path[j["dst_path"]] = j["dst_data"]

    for path, data in by_path.items():
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")

    print(f"\nSummary: ok={n_ok}  err={n_err}")
    # Print a few side-by-side examples
    print("\n--- Sample side-by-side (first 600 chars) ---")
    for j, new_cn, err in results[:3]:
        if err: continue
        print(f"\n[{j['folder']} var={j['var']} rep={j['rep']} turn={j['turn']}]")
        print(f"EN: {j['en'][:400]}")
        print(f"OLD CN (Sonnet): {j['old_cn'][:400]}")
        print(f"NEW CN (Opus):   {new_cn[:400]}")


if __name__ == "__main__":
    asyncio.run(main())
