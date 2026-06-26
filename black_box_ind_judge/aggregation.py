"""Aggregation and filtering format helpers for indicator results."""


def aggregate_indicators(
    results: list[dict],
    mode: str = "rollout",
) -> dict:
    """
    Aggregate indicators from multiple evaluation runs.

    Args:
        results: List of evaluation results from multiple runs
        mode: Either "rollout" (has turn_number/sentence) or "response" (simpler format)

    Returns:
        Dictionary with aggregated indicators and their detection counts
    """
    # Track indicators by a unique key
    indicator_map: dict[str, dict] = {}

    for run_idx, result in enumerate(results):
        detected = result.get("detected_indicators", [])
        for ind in detected:
            name = ind.get("indicator_name", "Unknown")

            if mode == "rollout":
                # Create key from indicator name, turn, and sentence
                turn = ind.get("turn_number")
                sentence = ind.get("sentence", "")
                # Normalize sentence for comparison (first 100 chars)
                sentence_key = sentence[:100].strip().lower() if sentence else ""
                key = f"{name}|{turn}|{sentence_key}"
            else:
                # For response mode, use indicator name and evidence
                evidence = ind.get("evidence", "")
                evidence_key = evidence[:100].strip().lower() if evidence else ""
                key = f"{name}|{evidence_key}"

            if key not in indicator_map:
                indicator_map[key] = {
                    "indicator_name": name,
                    "detections": [],
                    "count": 0,
                }
                if mode == "rollout":
                    indicator_map[key]["turn_number"] = turn
                    indicator_map[key]["sentence"] = sentence

            indicator_map[key]["detections"].append({
                "run_index": run_idx,
                "evidence": ind.get("evidence", ""),
                "confidence": ind.get("confidence", 0),
            })
            indicator_map[key]["count"] += 1

    # Convert to list and sort by count (descending)
    aggregated = list(indicator_map.values())
    aggregated.sort(key=lambda x: -x["count"])

    return {
        "num_runs": len(results),
        "aggregated_indicators": aggregated,
        "total_unique_indicators": len(aggregated),
    }


def format_aggregated_for_filter(aggregated: dict, mode: str = "rollout") -> str:
    """Format aggregated indicators for the filtering prompt."""
    lines = []
    for i, ind in enumerate(aggregated["aggregated_indicators"], 1):
        lines.append(f"### Candidate {i}: {ind['indicator_name']}")
        lines.append(f"- Detection count: {ind['count']} / {aggregated['num_runs']} runs")

        if mode == "rollout" and ind.get("turn_number"):
            lines.append(f"- Turn number: {ind['turn_number']}")
        if mode == "rollout" and ind.get("sentence"):
            lines.append(f"- Sentence: \"{ind['sentence']}\"")

        lines.append("- Evidence from each detection:")
        for det in ind["detections"]:
            conf = det.get("confidence", "N/A")
            lines.append(f"  - Run {det['run_index'] + 1} (confidence: {conf}): \"{det['evidence']}\"")
        lines.append("")

    return "\n".join(lines)
