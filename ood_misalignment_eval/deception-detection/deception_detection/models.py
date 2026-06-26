from enum import Enum
from functools import cache
from pathlib import Path
from typing import Literal, cast, overload

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from deception_detection.log import logger

# We only use the instruct variations of the models. Note not all sizes are available for all model
# families.
ModelType = Literal["gemma", "mistral", "llama", "qwen", "glm"]
ModelSize = Literal[1, 2, 3, 7, 8, 9, 14, 24, 27, 32, 70]


class ModelName(str, Enum):
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    GEMMA_9B = "gemma-9b"
    GEMMA_27B = "gemma-27b"  # gemma-2-27b-it (text-only, non-reasoning)
    MISTRAL_7B = "mistral-7b"
    MISTRAL_24B = "mistral-24b"  # Mistral-Small-24B-Instruct-2501 (text-only, non-reasoning)
    LLAMA_1B = "llama-1b"
    LLAMA_3B = "llama-3b"
    LLAMA_8B = "llama-8b"
    LLAMA_70B = "llama-70b"
    LLAMA_70B_3_3 = "llama-70b-3.3"
    LLAMA_70B_R1 = "llama-70b-r1"  # DeepSeek-R1-Distill-Llama-70B (reasoning, <think>)
    QWEN_14B = "qwen-14b"
    QWEN_32B = "qwen-32b"
    QWEN_35B = "qwen-35b"  # Qwen3.6-35B-A3B (MoE, hybrid linear/full attention)
    GLM_FLASH = "glm-9b-flash"

    @property
    def type(self) -> ModelType:
        return cast(ModelType, self.value.split("-")[0])

    @property
    def size(self) -> ModelSize:
        size_str = self.value.split("-")[1].replace("b", "")
        return cast(ModelSize, int(size_str))

    @property
    def is_llama_3_3(self) -> bool:
        return self == ModelName.LLAMA_70B_3_3


def get_model_name(model_name: ModelName) -> str:
    return model_name.value


def get_num_layers_for_model(model_name: ModelName) -> int:
    if model_name.type == "gemma":
        n_layers = {2: 26, 7: 32, 9: 42, 27: 46}
    elif model_name.type == "llama":
        n_layers = {1: 16, 3: 28, 8: 32, 70: 80, 405: 126}
    elif model_name.type == "mistral":
        n_layers = {7: 32, 24: 40}  # 7 = Mistral-7B-v0.x, 24 = Magistral-Small-2509
    elif model_name.type == "qwen":
        n_layers = {14: 40, 32: 64, 35: 40}  # 35 = Qwen3.6-35B-A3B (40 layers)
    elif model_name.type == "glm":
        n_layers = {9: 47}
    else:
        raise NotImplementedError(f"No number of layers for model {model_name}")
    return n_layers[model_name.size]


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[True],
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[None, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[False] = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]: ...


def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Load the model and tokenizer based on the given parameters.
    """
    if model_name.type == "gemma":
        model, tokenizer = get_gemma_model_and_tokenizer(
            model_name, omit_model, pw_locked, lora_path
        )
    elif model_name.type == "mistral":
        model, tokenizer = get_mistral_model_and_tokenizer(
            model_name, omit_model, pw_locked, lora_path
        )
    elif model_name.type == "llama":
        model, tokenizer = get_llama3_model_and_tokenizer(model_name, omit_model)
    elif model_name.type == "qwen":
        model, tokenizer = get_qwen_model_and_tokenizer(model_name, omit_model)
    elif model_name.type == "glm":
        model, tokenizer = get_glm_model_and_tokenizer(model_name, omit_model)
    else:
        raise ValueError(f"Invalid model type: {model_name.type}")

    if cut_at_layer is not None and model is not None:
        model.model.layers = model.model.layers[:cut_at_layer]

    return model, tokenizer


@cache
def get_gemma_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Loads and returns the model and tokenizer based on the given parameters.

    Args:
        model_name: The Gemma model to load (must be a GEMMA_* variant)
        omit_model: Don't load the model if we are just doing analysis
        pw_locked: Whether to use the pw_locked version
        lora_path: Optional path to a LoRA adapter
    """
    if not model_name.type == "gemma":
        raise ValueError(f"Expected Gemma model, got {model_name}")

    # Determine the appropriate dtype based on model size
    dtype = torch.float16 if model_name.size < 9 else torch.bfloat16

    load_dotenv()  # for hf token
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", padding_side="left")
    tokenizer.pad_token_id = tokenizer.bos_token_id

    if omit_model:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        f"google/gemma-2-{model_name.size}b-it",
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=dtype,
    )

    logger.info(f"Loaded model {model_name}")

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    if lora_path is not None:
        peft_model = PeftModel.from_pretrained(model, lora_path)  # type: ignore
        model = peft_model.merge_and_unload()
        logger.info(f"Loaded LoRA adapter {lora_path}")

    model.eval()
    return model, tokenizer


@cache
def get_mistral_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = True,
    lora_path: str | Path | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """Return Mistral model and tokenizer"""
    if not model_name.type == "mistral":
        raise ValueError(f"Expected Mistral model, got {model_name}")
    if lora_path is not None:
        raise ValueError("Lora is not supported for Mistral yet.")

    model_path = {
        ModelName.MISTRAL_7B: "mistralai/Mistral-7B-Instruct-v0.2",
        ModelName.MISTRAL_24B: "mistralai/Mistral-Small-24B-Instruct-2501",
    }[model_name]

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    if omit_model:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=dtype,
    )

    model.eval()

    # The pw_locked sandbagging LoRA only exists for the v0.2 base; never apply
    # it to the v0.3 / Magistral entries we added for cross-model probe training.
    if pw_locked and model_name == ModelName.MISTRAL_7B:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))  # type: ignore
        model = PeftModel.from_pretrained(
            model,  # type: ignore
            "FelixHofstaetter/mistral-instruct-sandbagging-new",
        )  # type: ignore
        # merge
        model = model.merge_and_unload()

    return model, tokenizer


@cache
def get_llama3_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer_name = "unsloth/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


@cache
def get_llama3_model_and_tokenizer(
    model_name: ModelName, omit_model: bool = False
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """Return Llama3 model and tokenizer"""
    model_path = {
        ModelName.LLAMA_1B: "meta-llama/Llama-3.2-1B-Instruct",
        ModelName.LLAMA_3B: "meta-llama/Llama-3.2-3B-Instruct",
        ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
        ModelName.LLAMA_70B: "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ModelName.LLAMA_8B: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ModelName.LLAMA_70B_R1: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    }[model_name]

    # R1-distill ships its own tokenizer + <think> chat template, so load the
    # tokenizer from the checkpoint itself rather than the shared Llama-3.3 one.
    # The stock Llama-3.x instruct tokenizers are equivalent to the shared one.
    if model_name == ModelName.LLAMA_70B_R1:
        load_dotenv()  # for hf token
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = get_llama3_tokenizer()

    if omit_model:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


@cache
def get_qwen_model_and_tokenizer(
    model_name: ModelName, omit_model: bool = False
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """Return Qwen3 model and tokenizer"""
    if not model_name.type == "qwen":
        raise ValueError(f"Expected Qwen model, got {model_name}")

    model_path = {
        ModelName.QWEN_14B: "Qwen/Qwen3-14B",
        ModelName.QWEN_32B: "Qwen/Qwen3-32B",
        ModelName.QWEN_35B: "Qwen/Qwen3.6-35B-A3B",
    }[model_name]

    load_dotenv()  # for hf token
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if omit_model:
        return None, tokenizer

    models_directory = Path("/data/huggingface")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        # cache_dir=models_directory,
        # local_files_only=True,
    )

    logger.info(f"Loaded model {model_name}")

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    model.eval()
    return model, tokenizer


@cache
def get_glm_model_and_tokenizer(
    model_name: ModelName, omit_model: bool = False
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """Return GLM-4.7-Flash model and tokenizer"""
    if not model_name.type == "glm":
        raise ValueError(f"Expected GLM model, got {model_name}")

    model_path = {
        ModelName.GLM_FLASH: "zai-org/GLM-4.7-Flash",
    }[model_name]

    load_dotenv()  # for hf token
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if omit_model:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loaded model {model_name}")

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    model.eval()
    return model, tokenizer
