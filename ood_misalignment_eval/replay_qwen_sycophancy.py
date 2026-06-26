"""Replay the GLM sycophancy-eval prompts through Qwen3.6-35B-A3B.

The sycophancy-eval dataset isn't present locally, so we reuse the exact prompts
from the committed GLM rollout (same prompts, only the model differs) and replace
each assistant response with Qwen's (reasoning captured as a <think> block).
"""
import asyncio, copy, json, os
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
SRC = ROOT / "ood_misalignment_eval/sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback_filtered/rollout.json"
DST_DIR = ROOT / "ood_misalignment_eval/sycophancy-eval/rollouts/qwen3.6-35b-a3b/sycophancy_feedback_filtered"
MODEL = "qwen/qwen3.6-35b-a3b"

key = next(l.split("=", 1)[1].strip().strip('"').strip("'")
           for l in (ROOT / "bloom/.env").read_text().splitlines()
           if "OPENROUTER_API_KEY" in l and "=" in l)
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
sem = asyncio.Semaphore(20)


def target_messages(rollout):
    """system/user messages from the target view, up to (not incl.) the assistant."""
    msgs = []
    for e in rollout["transcript"]["events"]:
        if "target" not in e.get("view", []):
            continue
        m = e.get("edit", {}).get("message", {})
        if m.get("role") in ("system", "user"):
            c = m.get("content", "")
            msgs.append({"role": m["role"], "content": c if isinstance(c, str) else json.dumps(c)})
    return msgs


async def call_qwen(messages):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=1.0, max_tokens=10000)
                msg = r.choices[0].message
                content = msg.content or ""
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning_content") or extra.get("reasoning")
                return reasoning, content
            except Exception as e:
                if attempt == 3:
                    print("  FAIL:", str(e)[:100]); return None, None
                await asyncio.sleep(2 * (attempt + 1))


async def one(rollout):
    new = copy.deepcopy(rollout)
    reasoning, content = await call_qwen(target_messages(rollout))
    if content is None:
        return None
    blocks = []
    if reasoning:
        blocks.append({"type": "reasoning", "reasoning": reasoning})
    blocks.append({"type": "text", "text": content})
    # replace the assistant event's content; update target model in metadata
    for e in new["transcript"]["events"]:
        m = e.get("edit", {}).get("message", {})
        if m.get("role") == "assistant":
            m["content"] = blocks
            m["model"] = MODEL
    new["transcript"].setdefault("metadata", {})["target_model"] = MODEL
    return new


async def main():
    src = json.load(open(SRC))
    rollouts = src["rollouts"]
    print(f"replaying {len(rollouts)} sycophancy prompts through Qwen ...", flush=True)
    results = await asyncio.gather(*[one(r) for r in rollouts])
    results = [r for r in results if r]
    for i, r in enumerate(results, 1):
        r["variation_number"] = i
    out = {"metadata": {"modality": "sycophancy-eval", "target": "qwen3.6-35b-a3b",
                        "dataset": "Sycophancy-Eval", "split": "feedback_filtered",
                        "num_rollouts": len(results)}, "rollouts": results}
    DST_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DST_DIR / "rollout.json", "w"))
    print(f"saved {len(results)} qwen sycophancy rollouts -> {DST_DIR/'rollout.json'}")


if __name__ == "__main__":
    asyncio.run(main())
