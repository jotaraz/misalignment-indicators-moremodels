"""
Tokenizes text datasets.
"""

import re
from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from jaxtyping import Bool, Int
from termcolor import colored
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from deception_detection.data import Dialogue
from deception_detection.data.base import DialogueDataset
from deception_detection.log import logger
from deception_detection.models import ModelType
from deception_detection.types import Message
from deception_detection.utils import preprocess_dialogue


def _normalize_glm_think_content(content: str) -> str:
    """Normalize <think> tag content to match GLM chat template behavior.

    The GLM-4.7 chat template extracts reasoning from the *first*
    ``<think>...</think>`` block and visible text from everything after the
    *last* ``</think>``.  It then calls ``.strip()`` on both.  We must apply
    the same transformation so that ``message.content`` matches the formatted
    dialogue string for detection-mask alignment.

    For messages with multiple ``<think>`` blocks the template keeps only the
    first block's reasoning and discards the rest, so we do the same here.
    """
    if "<think>" not in content or "</think>" not in content:
        return content
    # Reasoning: first <think>...</think> block (matches template's
    # content.split('</think>')[0].split('<think>')[-1])
    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
    # Visible text: everything after the *last* </think> (matches template's
    # content.split('</think>')[-1])
    visible = content.split("</think>")[-1].lstrip("\n")
    result = f"<think>{reasoning.strip()}</think>"
    if visible.strip():
        result += visible.strip()
    return result


def _normalize_qwen_think_content(content: str, is_final: bool) -> str:
    """Normalize an assistant message to match the Qwen3.x chat-template rendering.

    The Qwen template keeps ``<think>`` reasoning ONLY for the final assistant
    turn, rendering it as ``<think>\\n{reasoning}\\n</think>\\n\\n{visible}``
    (newlines around the tags, both parts stripped); for earlier assistant turns
    it STRIPS the reasoning and renders the visible answer only. We rewrite
    ``message.content`` to the exact rendered form so the detection-mask
    char-offset assertion in ``_get_detection_mask`` holds. ``is_final`` says
    whether this is the last assistant turn of the (already truncated) dialogue.
    Apply to assistant messages only (system/user turns are not wrapped).
    """
    has_think = "<think>" in content and "</think>" in content
    if not is_final:
        # Non-final assistant turn: template drops the reasoning, keeps visible only.
        return content.split("</think>")[-1].strip() if has_think else content.strip()
    if has_think:
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip()
        visible = content.split("</think>")[-1].strip()
        return f"<think>\n{reasoning}\n</think>\n\n{visible}"
    # Final assistant turn with no reasoning: template injects an empty think block.
    return f"<think>\n\n</think>\n\n{content.strip()}"


def decode_single_token(token_id: int, tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode a single token."""
    if "llama" in tokenizer.__class__.__name__.lower():
        # see https://github.com/TransformerLensOrg/TransformerLens/issues/373
        return tokenizer.decode([891, token_id])[3:]
    return tokenizer.decode(token_id)


def get_str_tokens(token_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[list[str]]:
    # Ensure the input is on CPU and convert to a list of lists
    token_ids_list = token_ids.cpu().tolist()

    # Convert token IDs to string tokens
    string_tokens = []
    for dialogue in token_ids_list:
        dialogue_tokens = [tokenizer.bos_token] if tokenizer.bos_token is not None else []
        for token_id in dialogue:
            # we use bos as pad
            if token_id != tokenizer.bos_token_id:
                token = decode_single_token(cast(int, token_id), tokenizer)
                dialogue_tokens.append(token)
        string_tokens.append(dialogue_tokens)
    return string_tokens


@dataclass
class TokenizedDataset:
    """
    A class to represent tokenized input data for deception detection models. If the dataset
    contains a pair of truthful/deceptive dialogues, this class only represents ONE of those.

    Attributes:
        dialogues (list[Dialogue]): List of dialogues that have been tokenized.
        formatted_dialogues (list[str]): List of dialogues formatted by the tokenizer.
        tokens (Int[Tensor, "batch seq_pos"]): Tokenized input text.
        detection_mask (Bool[Tensor, "batch seq_pos"]): Mask over tokens for where we will compute
            detection scores.
        attention_mask (Bool[Tensor, "batch seq_pos"]): Attention mask for the tokenized text.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to process the text.
    """

    dialogues: list[Dialogue]
    formatted_dialogues: list[str]
    tokens: Int[Tensor, "n_dialogues seq_pos"]
    str_tokens: list[list[str]]
    detection_mask: Bool[Tensor, "n_dialogues seq_pos"] | None
    attention_mask: Int[Tensor, "n_dialogues seq_pos"]
    dataset_partial_id: str | None = None

    @staticmethod
    def _final_message_is_assistant(dialogue: Dialogue) -> bool:
        """Check if the final message in a dialogue is an assistant message."""
        return dialogue[-1].role == "assistant"

    @staticmethod
    def _prefix_pattern(tokenizer_type: ModelType) -> re.Pattern[str]:
        """Regular expression that matches chat template tokens, used to match the detect mask
        to token indexes.

        Generally needs to match all of:
        - the start of dialogue, including padding, bos token, and the first header.
        - the <eot> token for end of the last message, plus the header for the new message
        - '\n\n' as the separator between a folded system prompt and the user message
        - an empty string for two messages objects from the same user that are merged.
        """
        if tokenizer_type == "gemma":
            begin_of_text = r"(<pad>)*(<bos>)?"
            end_of_last = r"(<end_of_turn>\n)?"
            start_of_turn = r"<start_of_turn>(user|model)\n"
            return re.compile(
                rf"{begin_of_text}{start_of_turn}|{end_of_last}{start_of_turn}|(\n\n)?"
            )
        elif tokenizer_type == "mistral":
            message_end = r"(</s>)?"
            message_start = r"((<pad>)*<s>)?"
            inst_start = r" \[INST\] "
            inst_end = r" \[/INST\] "
            return re.compile(
                rf"{message_end}{message_start}{inst_start}|{inst_end}|(\n\n)|{message_end}"
            )
        if tokenizer_type == "llama":
            begin_of_text = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
            header = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
            end_of_turn = r"<\|eot_id\|>"
            # llama3 system prompt has some boilerplate which we keep constant across all prompts
            date_info = (
                r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
            )
            # matches:
            # - padding + BOT + system prompt boilerplate
            # - end of turn + next header
            # - padding + BOT + system prompt boilerplate + end of turn + next header (if we have
            #   no additional system prompt)
            # - '\n\n' or the empty string
            return re.compile(
                rf"({begin_of_text}{header}{date_info})?({end_of_turn}{header})?|(\n\n)?"
            )
        elif tokenizer_type == "qwen":
            # Qwen uses <|im_start|>role\n and <|im_end|>\n format
            begin_of_text = r"(<\|endoftext\|>)*"
            start_of_turn = r"<\|im_start\|>(system|user|assistant)\n"
            end_of_turn = r"<\|im_end\|>\n"
            # matches:
            # - padding/BOT + start of turn header
            # - end of turn + start of next turn
            # - '\n\n' or the empty string
            return re.compile(
                rf"{begin_of_text}{start_of_turn}|{end_of_turn}{start_of_turn}|(\n\n)?"
            )
        elif tokenizer_type == "glm":
            # GLM uses <|system|>, <|user|>, <|assistant|></think> (no \n) and <|endoftext|>
            # The template adds </think> after <|assistant|> for existing assistant messages
            begin_of_text = r"(\[gMASK\]<sop>)?"
            start_of_turn = r"<\|(system|user|assistant)\|>(</think>)?"
            end_of_turn = r"<\|endoftext\|>"
            return re.compile(
                rf"{begin_of_text}{start_of_turn}|{end_of_turn}{start_of_turn}|(\n\n)?"
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    @staticmethod
    def _get_detection_mask(
        tokens: Int[Tensor, "num_dialogues seq"],
        tokenizer_out: Any,
        dialogues: list[Dialogue],
        formatted_dialogues: list[str],
        for_generation: bool,
        detect_all: bool,
        padding: dict[str, int],
        tokenizer_type: ModelType,
    ) -> Bool[Tensor, "num_dialogues seq"] | None:
        if for_generation:
            return None
        elif detect_all:
            return torch.ones_like(tokens).bool()

        detection_mask = torch.zeros_like(tokens).bool()
        for batch_idx, dialogue in enumerate(dialogues):
            char_idx: int = 0

            for message in dialogue:
                # find the start and end of the message in the formatted text
                match = re.match(  # type: ignore
                    TokenizedDataset._prefix_pattern(tokenizer_type),
                    formatted_dialogues[batch_idx][char_idx:],
                )

                assert match is not None, f"text={formatted_dialogues[batch_idx][char_idx:]}"
                start_char_idx: int = char_idx + match.end()
                end_char_idx: int = start_char_idx + len(message.content)
                assert (
                    formatted_dialogues[batch_idx][start_char_idx:end_char_idx] == message.content
                ), (
                    formatted_dialogues[batch_idx][start_char_idx:end_char_idx],
                    message.content,
                )

                # use tokenizer utilities to find the corresponding tokens
                if message.detect:
                    start_tok_index = tokenizer_out.char_to_token(batch_idx, start_char_idx)
                    assert (
                        start_tok_index is not None
                    ), f"start_char_idx={start_char_idx} text={message.content}, {dialogue}"
                    end_tok_index = tokenizer_out.char_to_token(batch_idx, end_char_idx)
                    if end_tok_index is None:
                        # end_char_idx is past the last token, use sequence length
                        end_tok_index = tokens.shape[1]
                    # we also optionally include the activations that predict the first tok of
                    # the message.
                    detection_mask[
                        batch_idx,
                        start_tok_index - padding["left"] : end_tok_index + padding["right"],
                    ] = True

                # update our current position in the string
                char_idx = end_char_idx

        return detection_mask

    @classmethod
    def from_dialogue_list(
        cls,
        dialogues: list[Dialogue],
        tokenizer: PreTrainedTokenizerBase,
        padding: dict[str, dict[str, int]],
        for_generation: bool = False,
        detect_all: bool = False,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        max_length: int | None = None,
        dataset_partial_id: str | None = None,
        trim_reasoning: bool = False,
        split_on: str | None = None,
        ensure_system_prompt: bool = False,
        template_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Create an TokenizedDataset of a list of dialogues

        Args:
            dialogues (list[Dialogue]): List of dialogues to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
            padding (dict[str, dict[str, int]]): Padding configuration for detection mask. Non-zero
                values will extend the detection mask to include tokens just before or after the
                marked ones.
            for_generation (bool): Whether to have the tokens ready for generation.
                Adds the generation prompt if no assistant prompt present, else trims off the
                assistant end of turn.
            detect_all (bool): Whether to detect on all tokens rather than just marked ones.
            detect_only_start_of_turn (bool): Whether to have the detect-mask only be true at
                <|start_of_turn|> tokens.
            detect_only_last_token (bool): Whether to only detect at the last token of each message,
                the position just before <|eot_id|> tokens.
            max_length (int | None): Maximum sequence length for truncation. None means no
                truncation.
        """
        assert dialogues, "Cannot tokenize an empty dialogue"

        if tokenizer.name_or_path.startswith("google/gemma"):
            tokenizer_type = "gemma"
        elif tokenizer.name_or_path.startswith("mistralai/"):
            tokenizer_type = "mistral"
        elif tokenizer.name_or_path.startswith(("meta-llama/Meta-Llama-3", "meta-llama/Llama-3", "unsloth/Llama-3")):
            tokenizer_type = "llama"
        elif tokenizer.name_or_path.startswith("Qwen/"):
            tokenizer_type = "qwen"
        elif "GLM" in tokenizer.name_or_path or tokenizer.name_or_path.startswith("zai-org/"):
            tokenizer_type = "glm"
        else:
            raise NotImplementedError("Only gemma, mistral, llama, qwen, and glm tokenizers are supported")

        # if llama and no system prompt, throw a warning
        if (
            ensure_system_prompt
            and tokenizer_type == "llama"
            and not any(d[0].role == "system" for d in dialogues)
        ):
            logger.warning(
                f"No system prompts for {dataset_partial_id}. "
                f"Adding a default. e.g >>{dialogues[0]}<<"
            )
            dialogues = [
                [Message("system", "You are an AI assistant.", False)] + d for d in dialogues
            ]

        # GLM chat template normalizes <think> content by stripping newlines around tags.
        # We must normalize message content to match, or the detection mask assertion fails.
        if tokenizer_type == "glm":
            dialogues = [
                [Message(m.role, _normalize_glm_think_content(m.content), m.detect) for m in d]
                for d in dialogues
            ]

        # Qwen3.x template keeps <think> reasoning only for the FINAL assistant turn and
        # strips it from earlier turns. To probe a target turn's reasoning we truncate the
        # dialogue to end at the last detected message (later turns are causally downstream,
        # so the target turn's activations are unchanged), then normalize each assistant turn
        # to its rendered form (reasoning kept for the final turn, stripped before).
        if tokenizer_type == "qwen":
            new_dialogues = []
            for d in dialogues:
                detect_idx = [i for i, m in enumerate(d) if m.detect]
                d = d[: max(detect_idx) + 1] if detect_idx else d
                last_asst = max(
                    (i for i, m in enumerate(d) if m.role == "assistant"), default=-1
                )
                new_dialogues.append(
                    [
                        Message(
                            m.role,
                            _normalize_qwen_think_content(m.content, is_final=(i == last_asst))
                            if m.role == "assistant"
                            # template strips surrounding whitespace on every turn; match it
                            # so the detection-mask char-offset assertion holds (esp. tool
                            # outputs that end with a newline).
                            else m.content.strip(),
                            m.detect,
                        )
                        for i, m in enumerate(d)
                    ]
                )
            dialogues = new_dialogues

        # gemma and mistral don't have a system prompt, so we fold it into the user prompt
        # qwen and glm support system prompts like llama
        fold_system = tokenizer_type in ["gemma", "mistral"]
        processed = [preprocess_dialogue(d, fold_system=fold_system) for d in dialogues]
        final_assistants = [cls._final_message_is_assistant(d) for d in dialogues]
        assert all(final_assistants[0] == fa for fa in final_assistants)
        final_assistant = final_assistants[0]
        extra = template_kwargs or {}
        if for_generation:
            formatted_dialogues: list[str] = tokenizer.apply_chat_template(
                processed, tokenize=False, add_generation_prompt=not final_assistant, **extra
            )  # type: ignore
            if final_assistant:
                eot_suffix = {
                    "gemma": "<end_of_turn>\n",
                    "mistral": "</s>",
                    "llama": "<|eot_id|>",
                    "qwen": "<|im_end|>\n",
                    "glm": "<|endoftext|>",
                }[tokenizer_type]
                formatted_dialogues = [d.removesuffix(eot_suffix) for d in formatted_dialogues]
        else:
            formatted_dialogues: list[str] = tokenizer.apply_chat_template(
                processed, tokenize=False, add_generation_prompt=False, **extra
            )  # type: ignore

        tokenizer_out = tokenizer(
            formatted_dialogues,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        tokens: torch.Tensor = tokenizer_out["input_ids"]  # type: ignore
        detection_mask = cls._get_detection_mask(
            tokens,
            tokenizer_out,
            dialogues,
            formatted_dialogues,
            for_generation,
            detect_all,
            padding[tokenizer_type],
            tokenizer_type,
        )
        if detection_mask is not None and detect_only_start_of_turn:
            sot_tok = cast(int, tokenizer("<start_of_turn>", add_special_tokens=False).input_ids[0])
            detection_mask = detection_mask & (tokens == sot_tok)
        if detection_mask is not None and detect_only_last_token:
            eot_tok_str = {
                "gemma": "<end_of_turn>\n",
                "mistral": "</s>",
                "llama": "<|eot_id|>",
                "qwen": "<|im_end|>\n",
                "glm": "<|endoftext|>",
            }[tokenizer_type]
            eot_tok = cast(int, tokenizer(eot_tok_str, add_special_tokens=False).input_ids[0])
            # we only want to use the detection mask on the last token of the message
            # that is, if EOT is the next token the model generated.
            next_tok = tokens.roll(shifts=-1, dims=1)
            detection_mask = detection_mask & (next_tok == eot_tok)
            # this will fail for repe and other datasets where the detect mask doesn't go until EOT

        if detection_mask is not None and detection_mask.sum() == 0:
            logger.warning("No tokens in detection mask")

        return cls(
            dialogues=dialogues,
            formatted_dialogues=formatted_dialogues,
            tokens=tokens,
            str_tokens=get_str_tokens(tokens, tokenizer),
            detection_mask=detection_mask,
            attention_mask=cast(Tensor, tokenizer_out["attention_mask"]),
            dataset_partial_id=dataset_partial_id,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        for_generation: bool = False,
        detect_all: bool = False,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        max_length: int | None = None,
        additional_id: str | None = None,
    ) -> Self:
        """
        Create a TokenizedDataset from a DialogueDataset

        Args:
            dataset (DialogueDataset): The dataset containing dialogues to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
            padding (dict[str, dict[str, int]]): Padding configuration.
            for_generation (bool): Whether to prepare tokens for generation.
            detect_all (bool): Whether to detect all tokens.
            detect_only_start_of_turn (bool): Whether to detect only the start of turn tokens.
            detect_only_last_token (bool): Whether to detect only the last token before EOT.
            max_length (int | None): Maximum length of tokenized sequences.

        Returns:
            Self: A new instance of TokenizedDataset.
        """
        dataset_partial_id = f"{dataset.base_name}__{dataset.variant}"
        if additional_id:
            dataset_partial_id += f"_{additional_id}"
        return cls.from_dialogue_list(
            dialogues=dataset.dialogues,
            tokenizer=tokenizer,
            padding=dataset.padding,
            for_generation=for_generation,
            detect_all=detect_all,
            detect_only_start_of_turn=detect_only_start_of_turn,
            detect_only_last_token=detect_only_last_token,
            max_length=max_length,
            dataset_partial_id=dataset_partial_id,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Self:
        idx = idx if isinstance(idx, slice) else slice(idx, idx + 1)
        return self.__class__(
            dialogues=self.dialogues[idx],
            formatted_dialogues=self.formatted_dialogues[idx],
            tokens=self.tokens[idx],
            str_tokens=self.str_tokens[idx],
            detection_mask=self.detection_mask[idx] if self.detection_mask is not None else None,
            attention_mask=self.attention_mask[idx],
        )

    def display_detection_mask(self, idx: int, tokenizer: PreTrainedTokenizerBase) -> None:
        """Display the detection mask for a given dialogue."""
        assert self.detection_mask is not None, "Detection mask not available"
        readable_tokens: list[str] = tokenizer.batch_decode(
            self.tokens[idx], clean_up_tokenization_spaces=False
        )
        for readable_token, mask in zip(readable_tokens, self.detection_mask[idx], strict=True):
            readable_token = readable_token.replace("\n", "\\n")
            color = "red" if mask else "light_grey"
            print(colored(readable_token, color), end="")
            print("|", end="")


def get_tokenized_dataset_pair(
    honest: DialogueDataset,
    deceptive: DialogueDataset,
    tokenizer: PreTrainedTokenizerBase,
    enforce_equal_detect_lengths: bool = False,
    detect_all: bool = False,
    detect_only_start_of_turn: bool = False,
    detect_only_last_token: bool = False,
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """
    Tokenizes the input dialogues in a PairedDialogueDataset, returning a pair of TokenizedDatasets.

    Args:
        honest (DialogueDataset): Dataset containing truthful dialogues.
        deceptive (DialogueDataset): Dataset containing deceptive dialogues.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
        enforce_equal_detect_lengths (bool): If True, this will trim the detection mask to the
            shorter of the pair.
        detect_all (bool): Whether to detect all tokens.
        detect_only_start_of_turn (bool): Whether to detect only the start of turn tokens.
        detect_only_last_token (bool): Whether to detect only the last token before EOT.

    Returns:
        tuple[TokenizedDataset, TokenizedDataset]: A pair of TokenizedDataset instances.
    """

    truthful_tokenized = TokenizedDataset.from_dataset(
        honest,
        tokenizer,
        detect_all=detect_all,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        additional_id="honest",
    )
    deceptive_tokenized = TokenizedDataset.from_dataset(
        deceptive,
        tokenizer,
        detect_all=detect_all,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        additional_id="deceptive",
    )

    if enforce_equal_detect_lengths:
        equalize_detection_masks(truthful_tokenized, deceptive_tokenized)

    return truthful_tokenized, deceptive_tokenized


def equalize_detection_masks(dataset1: TokenizedDataset, dataset2: TokenizedDataset):
    """
    Modifies the detection masks of both datasets in place to ensure they have equal lengths.
    """
    assert dataset1.detection_mask is not None
    assert dataset2.detection_mask is not None
    for i in range(len(dataset1.detection_mask)):
        mask1 = dataset1.detection_mask[i]
        mask2 = dataset2.detection_mask[i]

        start1, end1 = get_mask_range(mask1)
        start2, end2 = get_mask_range(mask2)

        length1 = end1 - start1
        length2 = end2 - start2
        min_length = min(length1, length2)

        # Trim the detection masks to the minimum length
        dataset1.detection_mask[i][start1 + min_length : end1] = False
        dataset2.detection_mask[i][start2 + min_length : end2] = False


def get_mask_range(mask: torch.Tensor) -> tuple[int, int]:
    """
    Finds the start and end indices of the True values in the mask.
    """
    start = torch.argmax((mask == True).int())
    end = start + torch.argmax((mask[start:] == False).int())

    # If no end found, set end to the length of the mask
    if end == start:
        end = len(mask)

    start = int(start)
    end = int(end)

    return start, end
