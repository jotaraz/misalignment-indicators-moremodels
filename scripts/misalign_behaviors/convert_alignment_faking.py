"""
Convert alignment_faking_public transcript files to bloom rollout format.

Input:
    JSON transcript file(s) with conversation data structured as:
    - system prompt + list of messages with role/content
    - Messages may contain XML tags: <GOAL>, <INFO>, <INPUT>,
      <SCRATCHPAD_REASONING>, <SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL>, <OUTPUT>

    Supports multiple input formats:
    1. Single JSON file with a list of conversations
    2. Single JSON file with a dict mapping IDs to conversations
    3. Directory containing multiple JSON transcript files

    Each conversation should have:
    - "system" or "system_prompt": str (the system prompt)
    - "messages": list of {"role": str, "content": str}

Output:
    rollout.json - Bloom-format transcripts with scratchpad as reasoning blocks

Note: No automatic judgment.json is generated — alignment faking detection
requires manual analysis.

Usage:
    python convert_alignment_faking.py transcripts.json
    python convert_alignment_faking.py /path/to/transcripts/ --output_dir /tmp/output
"""

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def make_event(role: str, content, view=None):
    """Create a transcript event in bloom format."""
    if view is None:
        view = ["target", "combined"]
    return {
        "type": "transcript_event",
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "edit": {
            "operation": "add",
            "message": {
                "role": role,
                "id": str(uuid.uuid4()),
                "content": content,
            },
        },
    }


def parse_assistant_content(text: str):
    """Parse assistant message with XML tags into structured content blocks.

    Extracts <SCRATCHPAD_REASONING*> as reasoning blocks and <OUTPUT> as text blocks.
    Returns a list of content blocks or plain text if no tags found.
    """
    # Check if message contains alignment faking XML tags
    has_scratchpad = "<SCRATCHPAD_REASONING" in text
    has_output = "<OUTPUT>" in text

    if not has_scratchpad and not has_output:
        return text

    content_blocks = []

    # Extract all scratchpad reasoning blocks
    scratchpad_pattern = r"<SCRATCHPAD_REASONING[^>]*>(.*?)</SCRATCHPAD_REASONING[^>]*>"
    scratchpad_matches = re.findall(scratchpad_pattern, text, re.DOTALL)
    for reasoning in scratchpad_matches:
        reasoning = reasoning.strip()
        if reasoning:
            content_blocks.append({
                "type": "reasoning",
                "reasoning": reasoning,
            })

    # Also check for strategy tag
    strategy_pattern = r"<SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL>(.*?)</SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL>"
    strategy_matches = re.findall(strategy_pattern, text, re.DOTALL)
    for reasoning in strategy_matches:
        reasoning = reasoning.strip()
        if reasoning:
            content_blocks.insert(0, {
                "type": "reasoning",
                "reasoning": reasoning,
            })

    # Extract OUTPUT block
    output_pattern = r"<OUTPUT>(.*?)</OUTPUT>"
    output_matches = re.findall(output_pattern, text, re.DOTALL)
    for output in output_matches:
        output = output.strip()
        # Strip inner response_for_user tags if present
        inner_pattern = r"<response_for_user>(.*?)</response_for_user>"
        inner_match = re.search(inner_pattern, output, re.DOTALL)
        if inner_match:
            output = inner_match.group(1).strip()
        if output:
            content_blocks.append({
                "type": "text",
                "text": output,
            })

    return content_blocks if content_blocks else text


def convert_conversation(conversation: dict, variation_number: int) -> dict:
    """Convert a single conversation to a bloom rollout entry."""
    # Extract system prompt
    system_prompt = conversation.get("system") or conversation.get("system_prompt", "")
    messages = conversation.get("messages", [])

    # Build variation description from system prompt (truncated)
    desc = system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
    desc = desc.replace("\n", " ")

    events = []

    # System prompt event
    if system_prompt:
        events.append(make_event("system", system_prompt))

    # Process messages
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip empty messages and transition markers
        if not content or content.strip() in ("<NEXT/>", "<NEW_EPISODE/>"):
            continue

        if role == "assistant":
            parsed_content = parse_assistant_content(content)
            events.append(make_event("assistant", parsed_content))
        elif role == "user":
            events.append(make_event("user", content))
        elif role == "system":
            events.append(make_event("system", content))

    return {
        "variation_number": variation_number,
        "variation_description": desc,
        "repetition_number": 1,
        "modality": "alignment-faking",
        "transcript": {
            "metadata": {
                "source": "alignment_faking_public",
            },
            "events": events,
        },
    }


def load_conversations(path: Path) -> list[dict]:
    """Load conversations from a file or directory.

    Supports:
    - JSON file with list of conversations
    - JSON file with dict mapping IDs to conversations
    - Directory of JSON files
    """
    if path.is_dir():
        conversations = []
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                conversations.extend(data)
            elif isinstance(data, dict):
                if "messages" in data:
                    # Single conversation
                    conversations.append(data)
                else:
                    # Dict of conversations
                    for conv_id, conv in data.items():
                        if isinstance(conv, dict):
                            conv.setdefault("id", conv_id)
                            conversations.append(conv)
            print(f"  Loaded {json_file.name}")
        return conversations

    # Single file
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "messages" in data:
            return [data]
        # Dict mapping IDs to conversations
        conversations = []
        for conv_id, conv in data.items():
            if isinstance(conv, dict):
                conv.setdefault("id", conv_id)
                conversations.append(conv)
        return conversations

    return []


def convert_transcripts(input_path: str, output_dir: str | None = None, max_samples: int | None = None):
    input_path = Path(input_path)

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    elif input_path.is_file():
        out_dir = input_path.parent / "bloom_rollout"
    else:
        out_dir = input_path / "bloom_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading transcripts from: {input_path}")
    conversations = load_conversations(input_path)

    if not conversations:
        print("Warning: No conversations found")
        return

    print(f"Found {len(conversations)} conversation(s)")

    if max_samples is not None and len(conversations) > max_samples:
        conversations = conversations[:max_samples]
        print(f"Sampling first {max_samples} conversation(s)")

    bloom_rollouts = []
    for i, conv in enumerate(conversations, start=1):
        rollout = convert_conversation(conv, i)
        bloom_rollouts.append(rollout)

    # Write rollout.json
    rollout_out = {
        "metadata": {
            "source": str(input_path),
            "modality": "alignment-faking",
            "num_rollouts": len(bloom_rollouts),
        },
        "rollouts": bloom_rollouts,
    }
    rollout_path = out_dir / "rollout.json"
    with open(rollout_path, "w") as f:
        json.dump(rollout_out, f, indent=2)

    print(f"Converted {len(bloom_rollouts)} transcripts")
    print(f"  rollout.json -> {rollout_path}")
    print(f"  (No judgment.json — alignment faking requires manual analysis)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert alignment_faking_public transcripts to bloom format"
    )
    parser.add_argument(
        "input",
        help="Path to transcript JSON file or directory of JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: auto)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of transcripts to convert (default: all)",
    )
    args = parser.parse_args()

    convert_transcripts(args.input, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
