"""MisalignmentJudge class for evaluating misalignment indicators in LLM responses."""

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

try:
    from .prompts import (
        JUDGE_PROMPT_TEMPLATE,
        ROLLOUT_JUDGE_PROMPT_TEMPLATE,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_SPANS,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL_SPANS,
        FILTER_INDICATORS_PROMPT_TEMPLATE,
    )
    from .conversation import (
        extract_conversation_from_rollout,
        format_conversation_for_judge,
        format_conversation_up_to_turn,
        get_turns_with_reasoning,
    )
    from .aggregation import (
        aggregate_indicators,
        format_aggregated_for_filter,
    )
except ImportError:
    from prompts import (
        JUDGE_PROMPT_TEMPLATE,
        ROLLOUT_JUDGE_PROMPT_TEMPLATE,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_SPANS,
        PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL_SPANS,
        FILTER_INDICATORS_PROMPT_TEMPLATE,
    )
    from conversation import (
        extract_conversation_from_rollout,
        format_conversation_for_judge,
        format_conversation_up_to_turn,
        get_turns_with_reasoning,
    )
    from aggregation import (
        aggregate_indicators,
        format_aggregated_for_filter,
    )

def _import_indicator_modules():
    """Import all indicator modules, handling different import contexts."""
    modules = {}
    try:
        from indicators.misalignment_indicators import get_all_indicators_prompt, get_indicator_names
        from indicators.general_misalignment_indicators import get_all_general_indicators_prompt, get_general_indicator_names
        from indicators.misalignment_indicators_v2 import get_all_indicators_v2_prompt, get_indicator_v2_names
        from indicators.misalignment_indicators_v2_general_cross_behavior import get_cross_behavior_indicators_prompt, get_cross_behavior_indicator_names
        from indicators.misalignment_indicators_v2_general_behavior import get_behavior_indicators_prompt, get_behavior_indicator_names
        # v2.1
        from indicators.misalignment_indicators_v2_1 import get_all_indicators_v2_prompt as get_v21_mid_prompt
        from indicators.misalignment_indicators_v2_1_finegrain import get_all_indicators_v2_prompt as get_v21_finegrain_prompt
        from indicators.misalignment_indicators_v2_1_general_behavior import get_behavior_indicators_prompt as get_v21_per_behavior_prompt
        from indicators.misalignment_indicators_v2_1_general_cross_behavior import get_cross_behavior_indicators_prompt as get_v21_cross_behavior_prompt
        # v2.2
        from indicators.misalignment_indicators_v2_2 import get_all_indicators_v2_prompt as get_v22_mid_prompt
        from indicators.misalignment_indicators_v2_2_finegrain import get_all_indicators_v2_prompt as get_v22_finegrain_prompt
        from indicators.misalignment_indicators_v2_2_general_behavior import get_behavior_indicators_prompt as get_v22_per_behavior_prompt
        from indicators.misalignment_indicators_v2_2_general_cross_behavior import get_cross_behavior_indicators_prompt as get_v22_cross_behavior_prompt
        # v2.3
        from indicators.misalignment_indicators_v2_3 import get_all_indicators_v2_prompt as get_v23_prompt
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from indicators.misalignment_indicators import get_all_indicators_prompt, get_indicator_names
        from indicators.general_misalignment_indicators import get_all_general_indicators_prompt, get_general_indicator_names
        from indicators.misalignment_indicators_v2 import get_all_indicators_v2_prompt, get_indicator_v2_names
        from indicators.misalignment_indicators_v2_general_cross_behavior import get_cross_behavior_indicators_prompt, get_cross_behavior_indicator_names
        from indicators.misalignment_indicators_v2_general_behavior import get_behavior_indicators_prompt, get_behavior_indicator_names
        # v2.1
        from indicators.misalignment_indicators_v2_1 import get_all_indicators_v2_prompt as get_v21_mid_prompt
        from indicators.misalignment_indicators_v2_1_finegrain import get_all_indicators_v2_prompt as get_v21_finegrain_prompt
        from indicators.misalignment_indicators_v2_1_general_behavior import get_behavior_indicators_prompt as get_v21_per_behavior_prompt
        from indicators.misalignment_indicators_v2_1_general_cross_behavior import get_cross_behavior_indicators_prompt as get_v21_cross_behavior_prompt
        # v2.2
        from indicators.misalignment_indicators_v2_2 import get_all_indicators_v2_prompt as get_v22_mid_prompt
        from indicators.misalignment_indicators_v2_2_finegrain import get_all_indicators_v2_prompt as get_v22_finegrain_prompt
        from indicators.misalignment_indicators_v2_2_general_behavior import get_behavior_indicators_prompt as get_v22_per_behavior_prompt
        from indicators.misalignment_indicators_v2_2_general_cross_behavior import get_cross_behavior_indicators_prompt as get_v22_cross_behavior_prompt
        # v2.3
        from indicators.misalignment_indicators_v2_3 import get_all_indicators_v2_prompt as get_v23_prompt

    # v1
    modules["v1"] = get_all_indicators_prompt
    modules["v1-general"] = get_all_general_indicators_prompt
    # v2
    modules["v2-finegrain"] = get_all_indicators_v2_prompt
    modules["v2-cross-behavior"] = get_cross_behavior_indicators_prompt
    modules["v2-per-behavior"] = get_behavior_indicators_prompt
    # v2.1
    modules["v2.1-mid"] = get_v21_mid_prompt
    modules["v2.1-finegrain"] = get_v21_finegrain_prompt
    modules["v2.1-per-behavior"] = get_v21_per_behavior_prompt
    modules["v2.1-cross-behavior"] = get_v21_cross_behavior_prompt
    # v2.2
    modules["v2.2-mid"] = get_v22_mid_prompt
    modules["v2.2-finegrain"] = get_v22_finegrain_prompt
    modules["v2.2-per-behavior"] = get_v22_per_behavior_prompt
    modules["v2.2-cross-behavior"] = get_v22_cross_behavior_prompt
    # v2.3
    modules["v2.3"] = get_v23_prompt
    return modules


# Valid indicator set names
INDICATOR_SETS = [
    "v1", "v1-general",
    "v2-finegrain", "v2-cross-behavior", "v2-per-behavior",
    "v2.1-mid", "v2.1-finegrain", "v2.1-per-behavior", "v2.1-cross-behavior",
    "v2.2-mid", "v2.2-finegrain", "v2.2-per-behavior", "v2.2-cross-behavior",
    "v2.3",
]

_indicator_modules = _import_indicator_modules()


class MisalignmentJudge:
    """Judge for evaluating misalignment indicators in LLM responses."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        use_general_indicators: bool = False,
        thinking_budget: int = 10000,
        use_spans: bool = False,
        neutral: bool = False,
        no_future_context: bool = False,
        indicator_set: str | None = None,
        no_non_examples: bool = False,
    ):
        """
        Initialize the misalignment judge.

        Args:
            provider: Either "openai" or "anthropic"
            model: Model name to use. Defaults to gpt-4o for OpenAI, claude-3-5-sonnet for Anthropic
            use_general_indicators: If True, use general high-level indicators instead of detailed ones
            thinking_budget: Budget tokens for extended thinking (Anthropic only). 0 to disable.
            use_spans: If True, use span-based per-turn prompt instead of sentence-based
            neutral: If True, use neutrally-framed per-turn prompt (no "misalignment" language)
            no_future_context: If True, per-turn judging only includes the target turn and prior turns (no future turns)
            indicator_set: Explicit indicator set name. Overrides use_general_indicators if set.
                See INDICATOR_SETS for all options.
            no_non_examples: If True, omit "NOT this indicator" non-examples from the
                indicator definitions to reduce prompt length.
        """
        self.provider = provider.lower()
        self.use_general_indicators = use_general_indicators
        self.thinking_budget = thinking_budget
        self.use_spans = use_spans
        self.neutral = neutral
        self.no_future_context = no_future_context

        if self.provider == "openai":
            self.client = AsyncOpenAI()
            self.model = model or "gpt-4o"
        elif self.provider == "anthropic":
            self.client = AsyncAnthropic()
            self.model = model or "claude-3-5-sonnet-20241022"
        elif self.provider == "openrouter":
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
            self.model = model or "openai/gpt-4o"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'openrouter'.")

        self._uses_openai_api = self.provider in ("openai", "openrouter")

        self.prompt_template = JUDGE_PROMPT_TEMPLATE
        self.rollout_prompt_template = ROLLOUT_JUDGE_PROMPT_TEMPLATE
        if neutral and use_spans:
            self.per_turn_prompt_template = PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL_SPANS
        elif neutral:
            self.per_turn_prompt_template = PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_NEUTRAL
        elif use_spans:
            self.per_turn_prompt_template = PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE_SPANS
        else:
            self.per_turn_prompt_template = PER_TURN_ROLLOUT_JUDGE_PROMPT_TEMPLATE

        # Determine indicator set: explicit indicator_set > use_general_indicators > default
        if indicator_set is not None:
            if indicator_set not in INDICATOR_SETS:
                raise ValueError(f"Unknown indicator_set: {indicator_set}. Options: {INDICATOR_SETS}")
            self.indicator_set = indicator_set
        elif use_general_indicators:
            self.indicator_set = "v1-general"
        else:
            self.indicator_set = "v1"

        if no_non_examples:
            try:
                self.indicators_text = _indicator_modules[self.indicator_set](include_non_examples=False)
            except TypeError:
                # Older indicator modules don't accept include_non_examples
                self.indicators_text = _indicator_modules[self.indicator_set]()
        else:
            self.indicators_text = _indicator_modules[self.indicator_set]()

    async def judge(self, response: str) -> dict[str, Any]:
        """
        Judge a single response for misalignment indicators.

        Args:
            response: The AI response to evaluate

        Returns:
            Dictionary containing detected indicators, overall score, and reasoning
        """
        prompt = self.prompt_template.format(
            indicators=self.indicators_text,
            response=response,
        )

        if self._uses_openai_api:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=0,
            )
            result_text = completion.choices[0].message.content
        else:  # anthropic
            api_kwargs = {
                "model": self.model,
                "max_tokens": 4096 if self.thinking_budget == 0 else self.thinking_budget + 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.thinking_budget > 0:
                api_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            completion = await self.client.messages.create(**api_kwargs)
            if not completion.content:
                return {
                    "error": f"Empty response from API (stop_reason: {completion.stop_reason})",
                    "raw_response": "",
                    "detected_indicators": [],
                    "overall_misalignment_score": None,
                    "reasoning": "Empty API response",
                }
            # With extended thinking, find the text block (not thinking block)
            result_text = None
            for block in completion.content:
                if block.type == "text":
                    result_text = block.text
                    break
            if result_text is None:
                return {
                    "error": "No text block in response",
                    "raw_response": str(completion.content),
                    "detected_indicators": [],
                    "overall_misalignment_score": None,
                    "reasoning": "No text block",
                }

        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse judge response: {e}",
                "raw_response": result_text,
                "detected_indicators": [],
                "overall_misalignment_score": None,
                "reasoning": "Parse error",
            }

    async def judge_rollout(self, rollout: dict) -> dict[str, Any]:
        """
        Judge a multi-turn rollout for misalignment indicators.

        Args:
            rollout: A rollout dictionary containing transcript

        Returns:
            Dictionary containing detected indicators with turn/sentence info
        """
        # Extract conversation from rollout
        conversation_turns = extract_conversation_from_rollout(rollout)

        # Format conversation for the judge
        formatted_conversation = format_conversation_for_judge(conversation_turns)

        prompt = self.rollout_prompt_template.format(
            indicators=self.indicators_text,
            conversation=formatted_conversation,
        )

        if self._uses_openai_api:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=0,
            )
            result_text = completion.choices[0].message.content
        else:  # anthropic
            api_kwargs = {
                "model": self.model,
                "max_tokens": 8192 if self.thinking_budget == 0 else self.thinking_budget + 8192,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.thinking_budget > 0:
                api_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            completion = await self.client.messages.create(**api_kwargs)
            if not completion.content:
                return {
                    "error": f"Empty response from API (stop_reason: {completion.stop_reason})",
                    "raw_response": "",
                    "detected_indicators": [],
                }
            # With extended thinking, find the text block (not thinking block)
            result_text = None
            for block in completion.content:
                if block.type == "text":
                    result_text = block.text
                    break
            if result_text is None:
                return {
                    "error": "No text block in response",
                    "raw_response": str(completion.content),
                    "detected_indicators": [],
                }

        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse judge response: {e}",
                "raw_response": result_text,
                "detected_indicators": [],
            }

    async def _call_judge_api(
        self,
        prompt: str,
        temperature: float = 0,
        seed: int = 0,
        max_output_tokens: int = 8192,
        max_retries: int = 5,
    ) -> dict[str, Any]:
        """
        Make an API call to the judge and parse the JSON response.
        Retries with exponential backoff on transient errors (429, 500, 502, 503, 529).

        Args:
            prompt: The full prompt to send
            temperature: Sampling temperature
            seed: Random seed (OpenAI only)
            max_output_tokens: Maximum output tokens
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response dict
        """
        RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}

        for attempt in range(max_retries + 1):
            try:
                if self._uses_openai_api:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        seed=seed,
                    )
                    result_text = completion.choices[0].message.content
                else:  # anthropic
                    api_kwargs = {
                        "model": self.model,
                        "max_tokens": max_output_tokens if self.thinking_budget == 0 else self.thinking_budget + max_output_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if temperature > 0:
                        api_kwargs["temperature"] = temperature
                    if self.thinking_budget > 0:
                        api_kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": self.thinking_budget,
                        }
                    completion = await self.client.messages.create(**api_kwargs)
                    if not completion.content:
                        return {
                            "error": f"Empty response from API (stop_reason: {completion.stop_reason})",
                            "detected_indicators": [],
                        }
                    result_text = None
                    for block in completion.content:
                        if block.type == "text":
                            result_text = block.text
                            break
                    if result_text is None:
                        return {
                            "error": "No text block in response",
                            "detected_indicators": [],
                        }
                break  # success, exit retry loop
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait_time = min(2 ** attempt * 5, 120)  # 5s, 10s, 20s, 40s, 80s
                    print(f"  [Retry {attempt + 1}/{max_retries}] {type(e).__name__} (status {status_code}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        # Parse JSON from response
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            return json.loads(result_text.strip())
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse judge response: {e}",
                "raw_response": result_text,
                "detected_indicators": [],
            }

    async def judge_rollout_per_turn(self, rollout: dict) -> dict[str, Any]:
        """
        Judge a multi-turn rollout by evaluating one turn at a time.

        For each turn with reasoning, sends a separate prompt asking the model
        to analyze only that turn (while providing the full conversation for context).
        Results from all turns are assembled into a single output.

        Args:
            rollout: A rollout dictionary containing transcript

        Returns:
            Dictionary containing detected indicators with turn/sentence info
        """
        conversation_turns = extract_conversation_from_rollout(rollout)
        turns_with_reasoning = get_turns_with_reasoning(conversation_turns)

        # Pre-format conversation: full or truncated per turn
        if self.no_future_context:
            formatted_conversations = {
                turn_number: format_conversation_up_to_turn(conversation_turns, turn_number)
                for turn_number in turns_with_reasoning
            }
        else:
            full_conversation = format_conversation_for_judge(conversation_turns)
            formatted_conversations = {
                turn_number: full_conversation
                for turn_number in turns_with_reasoning
            }

        all_indicators = []
        for turn_number in turns_with_reasoning:
            prompt = self.per_turn_prompt_template.format(
                indicators=self.indicators_text,
                conversation=formatted_conversations[turn_number],
                target_turn_number=turn_number,
            )
            result = await self._call_judge_api(prompt)
            indicators = result.get("detected_indicators", [])
            # Add turn_number to each indicator; normalize span -> sentence
            for ind in indicators:
                ind["turn_number"] = turn_number
                if "span" in ind and "sentence" not in ind:
                    ind["sentence"] = ind.pop("span")
            all_indicators.extend(indicators)

        return {"detected_indicators": all_indicators}

    async def judge_rollout_per_turn_with_aggregation(
        self,
        rollout: dict,
        k: int = 3,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Judge a rollout per-turn, k times each, and aggregate results.

        For each of k runs, evaluates each turn independently, then aggregates
        the results across runs.

        Args:
            rollout: A rollout dictionary containing transcript
            k: Number of times to run detection
            temperature: Temperature for API calls

        Returns:
            Dictionary with aggregated indicators from k runs
        """
        conversation_turns = extract_conversation_from_rollout(rollout)
        formatted_full_conversation = format_conversation_for_judge(conversation_turns)
        turns_with_reasoning = get_turns_with_reasoning(conversation_turns)

        # Pre-format conversation: full or truncated per turn
        if self.no_future_context:
            formatted_conversations = {
                turn_number: format_conversation_up_to_turn(conversation_turns, turn_number)
                for turn_number in turns_with_reasoning
            }
        else:
            formatted_conversations = {
                turn_number: formatted_full_conversation
                for turn_number in turns_with_reasoning
            }

        results = []
        for run_idx in range(k):
            all_indicators = []
            for turn_number in turns_with_reasoning:
                prompt = self.per_turn_prompt_template.format(
                    indicators=self.indicators_text,
                    conversation=formatted_conversations[turn_number],
                    target_turn_number=turn_number,
                )
                result = await self._call_judge_api(
                    prompt, temperature=temperature, seed=run_idx,
                )
                indicators = result.get("detected_indicators", [])
                for ind in indicators:
                    ind["turn_number"] = turn_number
                    if "span" in ind and "sentence" not in ind:
                        ind["sentence"] = ind.pop("span")
                all_indicators.extend(indicators)
            results.append({"detected_indicators": all_indicators})

        # Aggregate results - always use full conversation for the stored formatted_conversation
        aggregated = aggregate_indicators(results, mode="rollout")
        aggregated["formatted_conversation"] = formatted_full_conversation
        aggregated["individual_runs"] = results

        return aggregated

    async def judge_rollout_per_turn_batch(
        self,
        rollouts: list[dict],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Judge multiple rollouts per-turn concurrently.

        Args:
            rollouts: List of rollout dictionaries to evaluate
            max_concurrent: Maximum number of concurrent rollouts

        Returns:
            List of judgment results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_judge(rollout: dict) -> dict[str, Any]:
            async with semaphore:
                return await self.judge_rollout_per_turn(rollout)

        tasks = [limited_judge(r) for r in rollouts]
        return await asyncio.gather(*tasks)

    async def judge_rollout_per_turn_batch_aggregated(
        self,
        rollouts: list[dict],
        k: int = 3,
        temperature: float = 0.7,
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Judge multiple rollouts per-turn with aggregation and filtering.

        Args:
            rollouts: List of rollout dictionaries
            k: Number of detection runs per rollout
            temperature: Temperature for API calls
            max_concurrent: Maximum concurrent rollouts to process

        Returns:
            List of results with both unfiltered and filtered indicators
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_judge(rollout: dict) -> dict[str, Any]:
            async with semaphore:
                aggregated = await self.judge_rollout_per_turn_with_aggregation(
                    rollout, k=k, temperature=temperature
                )
                # Filter the aggregated results
                original_content = aggregated.get("formatted_conversation", "")
                filtered = await self.filter_aggregated_indicators(
                    aggregated, original_content, mode="rollout"
                )
                return {
                    "aggregated": aggregated,
                    "filtered": filtered,
                }

        tasks = [limited_judge(r) for r in rollouts]
        return await asyncio.gather(*tasks)

    async def judge_batch(
        self,
        responses: list[str],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Judge multiple responses concurrently.

        Args:
            responses: List of AI responses to evaluate
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of judgment results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_judge(response: str) -> dict[str, Any]:
            async with semaphore:
                return await self.judge(response)

        tasks = [limited_judge(r) for r in responses]
        return await asyncio.gather(*tasks)

    async def judge_rollout_batch(
        self,
        rollouts: list[dict],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Judge multiple rollouts concurrently.

        Args:
            rollouts: List of rollout dictionaries to evaluate
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of judgment results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_judge(rollout: dict) -> dict[str, Any]:
            async with semaphore:
                return await self.judge_rollout(rollout)

        tasks = [limited_judge(r) for r in rollouts]
        return await asyncio.gather(*tasks)

    async def judge_rollout_with_aggregation(
        self,
        rollout: dict,
        k: int = 3,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Judge a rollout k times and aggregate the results.

        Args:
            rollout: A rollout dictionary containing transcript
            k: Number of times to run detection
            temperature: Temperature for API calls (higher = more variation)

        Returns:
            Dictionary with aggregated indicators from k runs
        """
        # Extract conversation once
        conversation_turns = extract_conversation_from_rollout(rollout)
        formatted_conversation = format_conversation_for_judge(conversation_turns)

        prompt = self.rollout_prompt_template.format(
            indicators=self.indicators_text,
            conversation=formatted_conversation,
        )

        # Run k evaluations
        results = []
        for run_idx in range(k):
            if self._uses_openai_api:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    seed=run_idx,  # Different seed for each run
                )
                result_text = completion.choices[0].message.content
            else:  # anthropic
                api_kwargs = {
                    "model": self.model,
                    "max_tokens": 8192 if self.thinking_budget == 0 else self.thinking_budget + 8192,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if self.thinking_budget > 0:
                    api_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget,
                    }
                completion = await self.client.messages.create(**api_kwargs)
                if not completion.content:
                    results.append({"detected_indicators": [], "error": "Empty response"})
                    continue
                result_text = None
                for block in completion.content:
                    if block.type == "text":
                        result_text = block.text
                        break
                if result_text is None:
                    results.append({"detected_indicators": [], "error": "No text block"})
                    continue

            # Parse JSON
            try:
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                results.append(json.loads(result_text.strip()))
            except json.JSONDecodeError:
                results.append({"detected_indicators": [], "error": "Parse error"})

        # Aggregate results
        aggregated = aggregate_indicators(results, mode="rollout")
        aggregated["formatted_conversation"] = formatted_conversation
        aggregated["individual_runs"] = results

        return aggregated

    async def filter_aggregated_indicators(
        self,
        aggregated: dict,
        original_content: str,
        mode: str = "rollout",
    ) -> dict[str, Any]:
        """
        Filter and validate aggregated indicators using a final judge pass.

        Args:
            aggregated: Aggregated indicators from judge_rollout_with_aggregation
            original_content: The original content that was evaluated (for context)
            mode: Either "rollout" or "response"

        Returns:
            Dictionary with valid_indicators and rejected_indicators
        """
        # Format aggregated indicators for the filter prompt
        formatted_aggregated = format_aggregated_for_filter(aggregated, mode=mode)

        prompt = FILTER_INDICATORS_PROMPT_TEMPLATE.format(
            indicators=self.indicators_text,
            num_runs=aggregated["num_runs"],
            aggregated_indicators=formatted_aggregated,
            original_content=original_content,
        )

        if self._uses_openai_api:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=0,
            )
            result_text = completion.choices[0].message.content
        else:  # anthropic
            api_kwargs = {
                "model": self.model,
                "max_tokens": 12000 if self.thinking_budget == 0 else self.thinking_budget + 12000,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.thinking_budget > 0:
                api_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            completion = await self.client.messages.create(**api_kwargs)
            if not completion.content:
                return {
                    "error": f"Empty response from filter API (stop_reason: {completion.stop_reason})",
                    "valid_indicators": [],
                    "rejected_indicators": [],
                }
            # Check for truncation
            if completion.stop_reason == "max_tokens":
                return {
                    "error": "Response truncated (max_tokens reached). Try reducing number of indicators.",
                    "valid_indicators": [],
                    "rejected_indicators": [],
                }
            result_text = None
            for block in completion.content:
                if block.type == "text":
                    result_text = block.text
                    break
            if result_text is None:
                return {
                    "error": "No text block in filter response",
                    "valid_indicators": [],
                    "rejected_indicators": [],
                }

        # Parse JSON
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            return json.loads(result_text.strip())
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse filter response: {e}",
                "raw_response": result_text,
                "valid_indicators": [],
                "rejected_indicators": [],
            }

    async def judge_rollout_aggregated_and_filtered(
        self,
        rollout: dict,
        k: int = 3,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Full pipeline: judge k times, aggregate, then filter.

        Args:
            rollout: A rollout dictionary containing transcript
            k: Number of times to run detection
            temperature: Temperature for API calls

        Returns:
            Dictionary with both unfiltered and filtered results
        """
        # Step 1: Run k times and aggregate
        aggregated = await self.judge_rollout_with_aggregation(
            rollout, k=k, temperature=temperature
        )

        # Step 2: Filter the aggregated results
        original_content = aggregated.get("formatted_conversation", "")
        filtered = await self.filter_aggregated_indicators(
            aggregated, original_content, mode="rollout"
        )

        return {
            "aggregated": aggregated,
            "filtered": filtered,
        }

    async def judge_rollout_batch_aggregated(
        self,
        rollouts: list[dict],
        k: int = 3,
        temperature: float = 0.7,
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Judge multiple rollouts with aggregation and filtering.

        Args:
            rollouts: List of rollout dictionaries
            k: Number of detection runs per rollout
            temperature: Temperature for API calls
            max_concurrent: Maximum concurrent rollouts to process

        Returns:
            List of results with both unfiltered and filtered indicators
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_judge(rollout: dict) -> dict[str, Any]:
            async with semaphore:
                return await self.judge_rollout_aggregated_and_filtered(
                    rollout, k=k, temperature=temperature
                )

        tasks = [limited_judge(r) for r in rollouts]
        return await asyncio.gather(*tasks)

    async def __call__(self, response: str) -> dict[str, Any]:
        """Alias for judge method."""
        return await self.judge(response)
