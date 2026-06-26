"""
Generate on-policy completions from GLM-4.7 Flash and extract activations for PCA denoising.

Phase 1: Generate responses using vLLM (continuous batching, ~10-20x faster than HF generate).
Phase 2: Batched forward passes with HF model to extract per-layer hidden state activations.
         Dialogues are sorted by length and grouped into batches for efficient padding.

Activations are saved as float16 tensors per layer in probe/data/neutral/activations/.

Usage:
    python -m probe.neutral.extract_activations
    python -m probe.neutral.extract_activations --max-prompts 1000
    python -m probe.neutral.extract_activations --dialogues-path probe/data/neutral/dialogues.json  # skip generation
    python -m probe.neutral.extract_activations --tp 2  # tensor parallel across 2 GPUs for generation

Requires: pip install vllm
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Any

# Must be set before any CUDA init so vLLM tp>1 can spawn worker processes
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Also set env var as a fallback for vLLM's internal multiprocessing
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from tqdm import trange

REPO_ROOT = Path(__file__).parent.parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.activations import Activations
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue, Message

MODEL_NAME = ModelName.GLM_FLASH
MODEL_PATH = "zai-org/GLM-4.7-Flash"
DEFAULT_LAYERS = [27, 28, 29, 30]
DEFAULT_PADDING = {
    "gemma": {"left": 0, "right": 0},
    "mistral": {"left": 0, "right": 0},
    "llama": {"left": 0, "right": 0},
    "qwen": {"left": 0, "right": 0},
    "glm": {"left": 0, "right": 0},
}
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Phase 1: Generation via vLLM
# ---------------------------------------------------------------------------


def vllm_generate(
    prompts: list[str],
    system_prompt: str,
    max_new_tokens: int = 256,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """
    Generate responses for all prompts using vLLM.

    vLLM uses continuous batching and PagedAttention, so it processes
    all prompts efficiently without manual batching. ~10-20x faster
    than HuggingFace model.generate().
    """
    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM model (tp={tensor_parallel_size})...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=5120,  # prompt (1024) + generation (up to 4096)
    )

    tokenizer = llm.get_tokenizer()

    # Format prompts with chat template
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(text)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"  Generating {len(formatted_prompts)} responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        # vLLM strips the opening <think> tag for thinking models but keeps </think>.
        # Re-add <think> so the GLM chat template and detection mask alignment work.
        if "</think>" in text and "<think>" not in text:
            text = "<think>" + text
        responses.append(text)

    # Free vLLM GPU memory before loading HF model
    del llm
    torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Phase 2: Batched activation extraction (HF model)
# ---------------------------------------------------------------------------


def extract_neutral_activations(
    dialogues: list[dict],
    model: Any,
    tokenizer: Any,
    layers: list[int],
    max_tokens_per_layer: int = 100_000,
    batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """
    Extract per-layer activations from assistant tokens in completed dialogues.

    Dialogues are sorted by response length and processed in batches for
    efficient padding and GPU utilization.

    Returns dict mapping layer -> tensor of shape [n_tokens, emb_dim] in float16.
    """
    per_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    total_tokens: dict[int, int] = {l: 0 for l in layers}
    template_kwargs: dict[str, Any] = {"clear_thinking": False}

    # Shuffle dialogues so the token cap doesn't systematically bias toward
    # short or long responses.  Within each batch we still benefit from similar
    # lengths due to the random mix, and representative coverage matters more
    # than padding efficiency for PCA.
    import random as _rng
    indexed = list(enumerate(dialogues))
    _rng.shuffle(indexed)

    n_batches = (len(indexed) + batch_size - 1) // batch_size

    for batch_idx in trange(n_batches, desc=f"Extracting activations (batch_size={batch_size})"):
        # Check if all layers have enough tokens
        if all(total_tokens[l] >= max_tokens_per_layer for l in layers):
            print(f"Reached max tokens ({max_tokens_per_layer}) at batch {batch_idx}. Stopping.")
            break

        batch_slice = indexed[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        batch_dialogues: list[Dialogue] = []
        for _, d in batch_slice:
            dialogue: Dialogue = [
                Message(role="system", content=d["system_prompt"], detect=False),
                Message(role="user", content=d["user_prompt"], detect=False),
                Message(role="assistant", content=d["assistant_response"], detect=True),
            ]
            batch_dialogues.append(dialogue)

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=batch_dialogues,
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                template_kwargs=template_kwargs,
            )

            if toks.detection_mask is not None and not toks.detection_mask.any():
                continue

            acts = Activations.from_model(
                model, toks, batch_size=batch_size, layers=layers
            )
            masked = acts.get_masked_activations()  # [n_tokens_in_batch, n_layers, emb]

            if masked.numel() == 0:
                del acts
                continue

            masked = masked.cpu().half()

            for j, layer in enumerate(layers):
                if total_tokens[layer] < max_tokens_per_layer:
                    layer_acts = masked[:, j, :]  # [n_tokens, emb]
                    per_layer_acts[layer].append(layer_acts)
                    total_tokens[layer] += layer_acts.shape[0]

            del acts, masked
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size > 1:
                half = batch_size // 2
                print(f"  OOM at batch {batch_idx} (size={len(batch_slice)}), retrying with sub-batches of {half}")
                for sub_start in range(0, len(batch_slice), half):
                    sub_batch = batch_slice[sub_start : sub_start + half]
                    sub_dialogues: list[Dialogue] = []
                    for _, d in sub_batch:
                        sub_dialogues.append([
                            Message(role="system", content=d["system_prompt"], detect=False),
                            Message(role="user", content=d["user_prompt"], detect=False),
                            Message(role="assistant", content=d["assistant_response"], detect=True),
                        ])
                    try:
                        toks = TokenizedDataset.from_dialogue_list(
                            dialogues=sub_dialogues,
                            tokenizer=tokenizer,
                            padding=DEFAULT_PADDING,
                            template_kwargs=template_kwargs,
                        )
                        if toks.detection_mask is not None and not toks.detection_mask.any():
                            continue
                        acts = Activations.from_model(
                            model, toks, batch_size=half, layers=layers
                        )
                        masked = acts.get_masked_activations()
                        if masked.numel() > 0:
                            masked = masked.cpu().half()
                            for j, layer in enumerate(layers):
                                if total_tokens[layer] < max_tokens_per_layer:
                                    per_layer_acts[layer].append(masked[:, j, :])
                                    total_tokens[layer] += masked.shape[0]
                        del acts, masked
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"  Sub-batch also failed: {e}")
                        continue

        except Exception as e:
            print(f"  Failed on batch {batch_idx}: {e}")
            continue

    # Concatenate and trim
    result: dict[int, torch.Tensor] = {}
    for layer in layers:
        if per_layer_acts[layer]:
            cat = torch.cat(per_layer_acts[layer], dim=0)
            if cat.shape[0] > max_tokens_per_layer:
                cat = cat[:max_tokens_per_layer]
            result[layer] = cat

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate on-policy completions and extract neutral activations"
    )
    parser.add_argument(
        "--prompts-path", type=str, default=None,
        help="Path to prompts JSON (default: probe/data/neutral/prompts.json)",
    )
    parser.add_argument(
        "--dialogues-path", type=str, default=None,
        help="Path to pre-generated dialogues JSON (skips generation phase)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for activations (default: probe/data/neutral/activations/)",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size for vLLM generation (default: 1)",
    )
    parser.add_argument(
        "--extract-batch-size", type=int, default=32,
        help="Batch size for activation extraction phase (default: 32)",
    )
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument(
        "--max-tokens-per-layer", type=int, default=100_000,
        help="Max tokens to store per layer (default: 100000)",
    )
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data" / "neutral"
    prompts_path = Path(args.prompts_path) if args.prompts_path else base_dir / "prompts.json"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate or load dialogues
    dialogues_path = base_dir / "dialogues.json"

    if args.dialogues_path:
        print(f"Loading pre-generated dialogues from {args.dialogues_path}")
        with open(args.dialogues_path) as f:
            dialogues = json.load(f)
    else:
        with open(prompts_path) as f:
            data = json.load(f)
        prompts = [p["prompt"] for p in data["prompts"]]
        if args.max_prompts:
            prompts = prompts[: args.max_prompts]
        print(f"Loaded {len(prompts)} prompts")

        print(f"\nPhase 1: Generating responses via vLLM (tp={args.tp})...")
        responses = vllm_generate(
            prompts, args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tp,
        )

        dialogues = []
        for prompt, response in zip(prompts, responses):
            if response:
                dialogues.append({
                    "system_prompt": args.system_prompt,
                    "user_prompt": prompt,
                    "assistant_response": response,
                })

        # Save dialogues for reuse
        with open(dialogues_path, "w") as f:
            json.dump(dialogues, f, indent=2)
        print(f"Saved {len(dialogues)} dialogues to {dialogues_path}")

    if args.max_prompts:
        dialogues = dialogues[: args.max_prompts]

    # Phase 2: Extract activations with HF model (need output_hidden_states)
    print(f"\nPhase 2: Loading HF model for activation extraction...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    print(f"Extracting activations from {len(dialogues)} dialogues...")
    print(f"  Layers: {args.layers}")
    print(f"  Extract batch size: {args.extract_batch_size}")
    print(f"  Max tokens per layer: {args.max_tokens_per_layer}")

    layer_acts = extract_neutral_activations(
        dialogues, model, tokenizer, args.layers,
        max_tokens_per_layer=args.max_tokens_per_layer,
        batch_size=args.extract_batch_size,
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Save
    for layer, acts in layer_acts.items():
        save_path = output_dir / f"layer{layer}.pt"
        torch.save(acts, save_path)
        print(f"Saved layer {layer}: {acts.shape} to {save_path}")

    meta = {
        "model_name": MODEL_NAME.value,
        "layers": args.layers,
        "n_dialogues": len(dialogues),
        "tokens_per_layer": {str(l): t.shape[0] for l, t in layer_acts.items()},
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": args.system_prompt,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
