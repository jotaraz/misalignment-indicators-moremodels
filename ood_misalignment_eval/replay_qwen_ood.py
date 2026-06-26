"""Replay an OOD benchmark's prompts (from a committed GLM rollout) through
Qwen3.6-35B-A3B. Same prompts, only the model differs; Qwen's reasoning is captured
as a <think> block. Used when the benchmark's own dataset/runner isn't available
locally (sycophancy-eval, agentic-misalignment).

  python replay_qwen_ood.py --src <glm rollout.json> --dst <out dir> --modality <name>
"""
import argparse, asyncio, copy, json
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
MODEL = "qwen/qwen3.6-35b-a3b"
key = next(l.split("=", 1)[1].strip().strip('"').strip("'")
           for l in (ROOT / "bloom/.env").read_text().splitlines()
           if "OPENROUTER_API_KEY" in l and "=" in l)
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
sem = asyncio.Semaphore(15)


def target_messages(rollout):
    msgs = []
    for e in rollout["transcript"]["events"]:
        if "target" not in e.get("view", []):
            continue
        m = e.get("edit", {}).get("message", {})
        if m.get("role") in ("system", "user"):
            c = m.get("content", "")
            msgs.append({"role": m["role"], "content": c if isinstance(c, str) else json.dumps(c)})
    return msgs


async def call_qwen(messages, max_tokens):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=1.0, max_tokens=max_tokens)
                msg = r.choices[0].message
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning_content") or extra.get("reasoning")
                return reasoning, (msg.content or "")
            except Exception as e:
                if attempt == 3:
                    print("  FAIL:", str(e)[:100]); return None, None
                await asyncio.sleep(2 * (attempt + 1))


async def one(rollout, max_tokens):
    new = copy.deepcopy(rollout)
    reasoning, content = await call_qwen(target_messages(rollout), max_tokens)
    if content is None:
        return None
    blocks = []
    if reasoning:
        blocks.append({"type": "reasoning", "reasoning": reasoning})
    blocks.append({"type": "text", "text": content})
    for e in new["transcript"]["events"]:
        m = e.get("edit", {}).get("message", {})
        if m.get("role") == "assistant":
            m["content"] = blocks
            m["model"] = MODEL
    new["transcript"].setdefault("metadata", {})["target_model"] = MODEL
    return new


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--max-tokens", type=int, default=10000)
    a = ap.parse_args()
    src = json.load(open(a.src))
    rollouts = src["rollouts"]
    print(f"replaying {len(rollouts)} {a.modality} prompts through Qwen ...", flush=True)
    results = [r for r in await asyncio.gather(*[one(r, a.max_tokens) for r in rollouts]) if r]
    for i, r in enumerate(results, 1):
        r["variation_number"] = i
    out = {"metadata": {"modality": a.modality, "target": "qwen3.6-35b-a3b",
                        "num_rollouts": len(results)}, "rollouts": results}
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst / "rollout.json", "w"))
    print(f"saved {len(results)} qwen rollouts -> {dst/'rollout.json'}")


if __name__ == "__main__":
    asyncio.run(main())
