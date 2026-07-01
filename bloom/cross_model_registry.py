"""Single source of truth mapping each cross-model target across its 4 names.

Every Part-E script imports MODELS from here so the (probe model, probe dir,
OpenRouter key, rollout-dir slug, layers, reasoning) mapping lives in one place.

Per the cross-model-agentic-scope decision, the new targets are evaluated only on
the tool-free conversation behaviors + the 4 OOD benchmarks.
"""

# Conversation (tool-free) dev/test behaviors the new models are evaluated on.
CONV_BEHAVIORS = ["sycophancy", "instructed-strategic-sandbagging"]

# OOD benchmarks (single-turn replays) -> rollout subdir under ood_misalignment_eval/.
# {slug} is filled with the model's dir_slug.
OOD_BENCHMARKS = {
    "deception-bench": "ood_misalignment_eval/deception-bench/rollouts/{slug}",
    "mask": "ood_misalignment_eval/mask/rollouts/{slug}",
    "sycophancy-eval": "ood_misalignment_eval/sycophancy-eval/rollouts/{slug}/sycophancy_feedback_filtered",
    "agentic-misalignment": "ood_misalignment_eval/agentic-misalignment/results/ood_eval_{slug}/bloom_rollout",
}

MODELS = {
    # canonical key   probe ModelName    probe dir under probe/   detect layers
    "llama-70b-r1": dict(
        probe_model="llama-70b-r1", probe_dir="probes_llama_70b_r1", layers=[48, 50],
        bloom_key="deepseek-r1-llama-70b", dir_slug="deepseek_r1_llama_70b", reasoning=True,
    ),
    "llama-70b-3.3": dict(
        probe_model="llama-70b-3.3", probe_dir="probes_llama_70b_3_3", layers=[48, 50],
        bloom_key="llama-3.3-70b", dir_slug="llama_3_3_70b", reasoning=False,
    ),
    "gemma-27b": dict(
        probe_model="gemma-27b", probe_dir="probes_gemma_27b", layers=[27, 29],
        bloom_key="gemma-2-27b", dir_slug="gemma_2_27b", reasoning=False,
    ),
    "mistral-24b": dict(
        probe_model="mistral-24b", probe_dir="probes_mistral_24b", layers=[23, 25],
        bloom_key="mistral-small-24b", dir_slug="mistral_small_24b", reasoning=False,
    ),
    "gptoss-120b": dict(
        # gpt-oss-120b has 36 layers; 20/22 depth-matches GLM 27/29 of 47 (~0.58-0.62).
        # Reasoning lives in the harmony analysis channel (handled via bespoke mask).
        probe_model="gptoss-120b", probe_dir="probes_gpt_oss_120b", layers=[20, 22],
        bloom_key="gpt-oss-120b", dir_slug="gpt_oss_120b", reasoning=True,
    ),
}


def get(model_key: str) -> dict:
    if model_key not in MODELS:
        raise KeyError(f"unknown cross-model key {model_key!r}; known: {list(MODELS)}")
    return MODELS[model_key]


def dev_rollout_dirs(model_key: str, base: str = "bloom-results") -> list[str]:
    """Dev rollout dirs (misaligned + benign) for the conversation behaviors,
    relative to the bloom/ dir. e.g. bloom-results/sycophancy_gemma_2_27b."""
    slug = get(model_key)["dir_slug"]
    dirs = []
    for b in CONV_BEHAVIORS:
        dirs.append(f"{base}/{b}_{slug}")
        dirs.append(f"{base}/{b}_benign_{slug}")
    return dirs


def ood_rollout_dirs(model_key: str) -> dict[str, str]:
    """OOD benchmark -> rollout dir (relative to repo root) for this model."""
    slug = get(model_key)["dir_slug"]
    return {name: path.format(slug=slug) for name, path in OOD_BENCHMARKS.items()}
