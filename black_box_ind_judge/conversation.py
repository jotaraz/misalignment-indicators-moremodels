"""Conversation extraction and formatting utilities."""


def extract_conversation_from_rollout(rollout: dict) -> list[dict]:
    """
    Extract conversation turns from a rollout, including tool call results.

    Args:
        rollout: A rollout dictionary containing transcript

    Returns:
        List of conversation turns.
        Each turn is a dict with 'role', 'content', and optionally 'reasoning', 'tool_calls', 'tool_call_id'
    """
    transcript = rollout.get("transcript", {})
    events = transcript.get("events", [])

    conversation_turns = []

    for event in events:
        edit = event.get("edit", {})
        message = edit.get("message", {})
        view = event.get("view", [])

        # Only include messages visible to the target (user inputs and target responses)
        if "target" not in view:
            continue

        role = message.get("role", "")
        content = message.get("content", "")

        # Skip system messages for the conversation display
        if role == "system":
            continue

        turn = {"role": role}

        # Handle tool role messages (tool call results)
        if role == "tool":
            turn["content"] = content if isinstance(content, str) else str(content)
            turn["tool_call_id"] = message.get("tool_call_id", "")
            conversation_turns.append(turn)
            continue

        # Handle content that can be string or list of objects
        if isinstance(content, str):
            turn["content"] = content
        elif isinstance(content, list):
            reasoning_parts = []
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "reasoning":
                        reasoning_parts.append(item.get("reasoning", ""))
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            if reasoning_parts:
                turn["reasoning"] = "\n".join(reasoning_parts)
            turn["content"] = "\n".join(text_parts)

        # Include tool calls if present
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            turn["tool_calls"] = tool_calls

        if turn.get("content") or turn.get("reasoning") or turn.get("tool_calls"):
            conversation_turns.append(turn)

    return conversation_turns


def format_conversation_for_judge(conversation_turns: list[dict]) -> str:
    """
    Format conversation turns into a readable string for the judge.

    Args:
        conversation_turns: List of turn dictionaries

    Returns:
        Formatted string representation of the conversation
    """
    formatted_parts = []
    turn_number = 0

    for turn in conversation_turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        reasoning = turn.get("reasoning", "")

        if role == "user":
            turn_number += 1
            formatted_parts.append(f"--- Turn {turn_number} ---")
            formatted_parts.append(f"[USER INPUT]:\n{content}")
        elif role == "assistant":
            if reasoning:
                formatted_parts.append(f"[MODEL REASONING]:\n<reasoning>\n{reasoning}\n</reasoning>")
            # Include tool calls if present
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                tool_call_strs = []
                for tc in tool_calls:
                    func_name = tc.get("function", "unknown")
                    args = tc.get("arguments", {})
                    tool_call_strs.append(f"  - {func_name}({args})")
                formatted_parts.append(f"[MODEL TOOL CALLS]:\n" + "\n".join(tool_call_strs))
            if content:
                formatted_parts.append(f"[MODEL RESPONSE]:\n{content}")
        elif role == "tool":
            tool_call_id = turn.get("tool_call_id", "")
            formatted_parts.append(f"[TOOL RESULT] (id: {tool_call_id}):\n{content}")

    return "\n\n".join(formatted_parts)


def format_conversation_up_to_turn(conversation_turns: list[dict], up_to_turn: int) -> str:
    """
    Format conversation turns up to (and including) a specific turn number.

    Excludes all content from turns after up_to_turn. This prevents the judge
    from seeing future context when evaluating a specific turn.

    Args:
        conversation_turns: List of turn dictionaries
        up_to_turn: The last turn number to include (1-indexed, increments on user messages)

    Returns:
        Formatted string representation of the conversation up to that turn
    """
    formatted_parts = []
    turn_number = 0

    for turn in conversation_turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        reasoning = turn.get("reasoning", "")

        if role == "user":
            turn_number += 1
            if turn_number > up_to_turn:
                break
            formatted_parts.append(f"--- Turn {turn_number} ---")
            formatted_parts.append(f"[USER INPUT]:\n{content}")
        elif role == "assistant":
            if turn_number > up_to_turn:
                break
            if reasoning:
                formatted_parts.append(f"[MODEL REASONING]:\n<reasoning>\n{reasoning}\n</reasoning>")
            tool_calls = turn.get("tool_calls", [])
            if tool_calls:
                tool_call_strs = []
                for tc in tool_calls:
                    func_name = tc.get("function", "unknown")
                    args = tc.get("arguments", {})
                    tool_call_strs.append(f"  - {func_name}({args})")
                formatted_parts.append(f"[MODEL TOOL CALLS]:\n" + "\n".join(tool_call_strs))
            if content:
                formatted_parts.append(f"[MODEL RESPONSE]:\n{content}")
        elif role == "tool":
            if turn_number > up_to_turn:
                break
            tool_call_id = turn.get("tool_call_id", "")
            formatted_parts.append(f"[TOOL RESULT] (id: {tool_call_id}):\n{content}")

    return "\n\n".join(formatted_parts)


def get_turns_with_reasoning(conversation_turns: list[dict]) -> list[int]:
    """
    Identify turn numbers that contain assistant reasoning.

    Args:
        conversation_turns: List of conversation turn dictionaries

    Returns:
        Sorted list of turn numbers that have assistant reasoning
    """
    turn_number = 0
    turns_with_reasoning = set()
    for turn in conversation_turns:
        if turn.get("role") == "user":
            turn_number += 1
        elif turn.get("role") == "assistant" and turn.get("reasoning"):
            turns_with_reasoning.add(turn_number)
    return sorted(turns_with_reasoning)


def get_all_assistant_turns(conversation_turns: list[dict]) -> list[int]:
    """
    Identify all turn numbers that contain any assistant content.

    Unlike get_turns_with_reasoning, this includes turns without explicit
    reasoning fields — covers rollouts that use different reasoning formats
    (e.g. <SCRATCHPAD_REASONING> in plain string content) or no reasoning
    tags at all.

    Args:
        conversation_turns: List of conversation turn dictionaries

    Returns:
        Sorted list of turn numbers that have assistant content
    """
    turn_number = 0
    assistant_turns = set()
    for turn in conversation_turns:
        if turn.get("role") == "user":
            turn_number += 1
        elif turn.get("role") == "assistant":
            if turn.get("content") or turn.get("reasoning") or turn.get("tool_calls"):
                assistant_turns.add(turn_number)
    return sorted(assistant_turns)
