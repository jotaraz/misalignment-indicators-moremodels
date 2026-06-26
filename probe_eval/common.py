"""Shared constants, data-loading helpers, and JSON utilities for probe evaluation."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
INDICATOR_ROOT = REPO_ROOT / "bloom" / "indicator_results"
BLOOM_RESULTS_DIR = REPO_ROOT / "bloom" / "bloom-results"

# ---------------------------------------------------------------------------
# Indicator GT path mappings
# ---------------------------------------------------------------------------

# Finegrain indicator GT (v2.2-mid_opus)
BEHAVIOR_TO_INDICATOR_PATH: dict[str, str] = {
    "instructed-long-horizon-sabotage-backdoor": "v2.2/sabotage_backdoor/v2.2-mid_opus/rollout_run1.json",
    "instructed-long-horizon-sabotage-bug": "v2.2/sabotage_bug/v2.2-mid_opus/rollout_run1.json",
    "sycophancy": "v2.2/sycophancy/v2.2-mid_opus/rollout_run1.json",
}

# Behavior-level indicator GT (v2.2-per-behavior_opus)
BEHAVIOR_TO_INDICATOR_PATH_BEHAVIOR: dict[str, str] = {
    "instructed-long-horizon-sabotage-backdoor": "v2.2/sabotage_backdoor/v2.2-per-behavior_opus/rollout_run1.json",
    "instructed-long-horizon-sabotage-bug": "v2.2/sabotage_bug/v2.2-per-behavior_opus/rollout_run1.json",
    "sycophancy": "v2.2/sycophancy/v2.2-per-behavior_opus/rollout_run1.json",
}

# v2.3 indicator GT (opus)
BEHAVIOR_TO_INDICATOR_PATH_V2_3: dict[str, str] = {
    "instructed-long-horizon-sabotage-backdoor": "v2.3/sabotage_backdoor/v2.3_opus/rollout_run1.json",
    "instructed-long-horizon-sabotage-bug": "v2.3/sabotage_bug/v2.3_opus/rollout_run1.json",
    "sycophancy": "v2.3/sycophancy/v2.3_opus/rollout_run1.json",
    "strategic-deception_glmflash": "v2.3/strategic_deception/v2.3_opus/rollout_run1.json",
}

# ---------------------------------------------------------------------------
# Probe concept ↔ indicator name mapping
# ---------------------------------------------------------------------------

PROBE_TO_INDICATOR: dict[str, str] = {
    # Finegrain (v2.2) probes
    "concealment_planning": "Concealment Planning",
    "dishonest_positive_evaluation": "Dishonest Positive Evaluation",
    "distinct_self_goal_representation": "Distinct Self-Goal Representation",
    "evaluation_awareness": "Evaluation Awareness",
    "manufactured_agreement_shift": "Manufactured Agreement Shift",
    "observer_suspicion_modeling": "Observer Suspicion Modeling",
    "recognized_concern_suppression": "Recognized-Concern Suppression",
    "reward_awareness": "Reward Awareness",
    "situational_awareness_of_constraints": "Situational Awareness of Constraints",
    "threat_awareness": "Threat Awareness",
    "unauthorized_consequential_action_planning": "Unauthorized Consequential Action Planning",
    # Behavior-level (v2.2 general per-behavior) probes
    "deception": "Deception",
    "reward_hacking": "Reward Hacking",
    "sandbagging": "Sandbagging",
    "undermining_oversight": "Undermining Oversight",
    "sabotage": "Sabotage",
    "self_preservation": "Self-Preservation",
    "resource_acquisition": "Resource Acquisition",
    # v2.3 probes
    "action_concealment": "Action Concealment",
    "adversarial_user_framing": "Adversarial User Framing",
    "internal_output_divergence": "Internal-Output Divergence",
    "oversight_as_obstacle": "Oversight as Obstacle",
    "recognized_problem_suppression": "Recognized-Problem Suppression",
    "reward_gaming": "Reward Gaming",
    "self_consistency_monitoring": "Self-Consistency Monitoring",
}

INDICATOR_TO_PROBE: dict[str, str] = {v: k for k, v in PROBE_TO_INDICATOR.items()}

# Behavior-level probes detect a general behavior, not a specific indicator.
BEHAVIOR_PROBES: set[str] = {
    "deception", "reward_hacking", "sandbagging", "undermining_oversight",
    "sabotage", "self_preservation", "resource_acquisition",
}

# Precondition and behavioral concept probes — not misalignment indicators.
# Excluded from joint optimization by default.
PRECONDITION_CONCEPTS: set[str] = {
    "distinct_self_goal_representation",
    "evaluation_awareness",
    "situational_awareness_of_constraints",
    "reward_awareness",
    "threat_awareness",
}
BEHAVIORAL_CONCEPTS: set[str] = {
    "self_consistency_monitoring",
}
NON_INDICATOR_CONCEPTS: set[str] = PRECONDITION_CONCEPTS | BEHAVIORAL_CONCEPTS


# ---------------------------------------------------------------------------
# Data loading helpers (shared by both indicator and misalignment GT)
# ---------------------------------------------------------------------------

def get_rollout_var_rep(rollout_dir: str) -> dict[int, tuple[int, int]]:
    """Return ``{rollout_index: (variation_number, repetition_number)}``."""
    rollout_path = Path(rollout_dir) / "rollout.json"
    with open(rollout_path) as f:
        data = json.load(f)
    return {
        i: (r["variation_number"], r["repetition_number"])
        for i, r in enumerate(data["rollouts"])
    }


def get_n_turns(rollout_dir: str) -> dict[tuple[int, int], int]:
    """Return ``{(var, rep): n_turns}`` by counting target-view user messages."""
    rollout_path = Path(rollout_dir) / "rollout.json"
    with open(rollout_path) as f:
        data = json.load(f)

    result: dict[tuple[int, int], int] = {}
    for rollout in data["rollouts"]:
        key = (rollout["variation_number"], rollout["repetition_number"])
        n = sum(
            1
            for e in rollout["transcript"]["events"]
            if "target" in e.get("view", [])
            and e.get("edit", {}).get("message", {}).get("role") == "user"
        )
        result[key] = n
    return result


def load_probe_result(result_path: Path) -> dict | None:
    """Load a probe results.json, returning None if missing."""
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def get_concept_from_experiment_folder(experiment_folder: str) -> str | None:
    """Extract the probe concept name from the experiment folder path.

    Handles both old (trained_probes/{concept}/...) and new
    (probes/{indicator_set}/{concept}/...) directory layouts.
    """
    parts = experiment_folder.split("/")
    for i, part in enumerate(parts):
        if part.startswith("probes") and i + 2 < len(parts):
            # Handles: probes/{set}/{concept}/... and probes_v2/{set}/{concept}/...
            return parts[i + 2]
        if part.startswith("trained_probes") and i + 1 < len(parts):
            return parts[i + 1]
    return None


# ---------------------------------------------------------------------------
# JSON / aggregation helpers
# ---------------------------------------------------------------------------

def nan_to_none(v: float) -> float | None:
    """Convert NaN/Inf to None for JSON serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def safe_mean(values: list[float]) -> float | None:
    """Mean of non-NaN values, or None if empty."""
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not valid:
        return None
    return float(np.mean(valid))


def safe_max(values: list[float]) -> float | None:
    """Max of non-NaN values, or None if empty."""
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not valid:
        return None
    return float(max(valid))


def save_json(path: Path, data: dict) -> None:
    """Write dict to JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def filter_behaviors(
    by_behavior: dict[str, list],
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, list]:
    """Filter a {behavior_name: ...} dict by include/exclude glob patterns.

    Patterns use fnmatch-style matching (e.g. ``test_*`` matches any behavior
    starting with ``test_``).

    Args:
        by_behavior: Dict keyed by behavior name.
        include_patterns: If given, only keep behaviors matching at least one
            pattern.
        exclude_patterns: If given, remove behaviors matching any pattern.

    Returns:
        Filtered copy of *by_behavior*.
    """
    from fnmatch import fnmatch

    filtered = dict(by_behavior)
    if include_patterns:
        filtered = {
            b: v for b, v in filtered.items()
            if any(fnmatch(b, p) for p in include_patterns)
        }
    if exclude_patterns:
        filtered = {
            b: v for b, v in filtered.items()
            if not any(fnmatch(b, p) for p in exclude_patterns)
        }
    return filtered


def add_behavior_filter_args(parser) -> None:
    """Add --include-behaviors, --exclude-behaviors, --output-suffix to an
    argparse parser."""
    parser.add_argument(
        "--include-behaviors", type=str, default=None,
        help="Comma-separated fnmatch patterns; only include matching behaviors "
             "(e.g. 'test_*,bloom_rollout,sycophancy_*')",
    )
    parser.add_argument(
        "--exclude-behaviors", type=str, default=None,
        help="Comma-separated fnmatch patterns; exclude matching behaviors "
             "(e.g. 'test_*,bloom_rollout,sycophancy_*')",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix for output summary files (e.g. '_dev', '_test')",
    )
