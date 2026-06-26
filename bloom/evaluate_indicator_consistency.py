"""
Evaluate consistency between multiple indicator classification runs and the
final filtered result.

Metrics computed:
1. Variation-level binary agreement (any indicators detected or not)
   - Full agreement rate, pairwise Cohen's kappa
2. Indicator-name-level agreement per variation (Jaccard similarity)
   - Also per-turn variant (stricter)
3. Sentence-level agreement per variation (Jaccard similarity)
4. Confidence consistency for matched (variation, sentence, indicator_name)
5. Filtered vs. individual runs & majority vote

Usage:
    python evaluate_indicator_consistency.py <results_dir>
    python evaluate_indicator_consistency.py indicator_results/sabotage-backdoor-multiturn
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa for two binary label lists."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 1.0
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o = agree / n
    p_a = sum(labels_a) / n
    p_b = sum(labels_b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if abs(1 - p_e) < 1e-10:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def normalize_sentence(s: str) -> str:
    """Normalize a sentence for matching across runs."""
    return " ".join(s.strip().split())


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_run(path: Path) -> dict:
    """Load a run JSON and return {variation_number: [indicator_dicts]}."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for rollout in data["evaluated_rollouts"]:
        vn = rollout["variation_number"]
        result[vn] = rollout.get("detected_indicators", [])
    return result


def load_filtered(path: Path) -> dict:
    """Load filtered JSON and return {variation_number: [indicator_dicts]}."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for rollout in data["evaluated_rollouts"]:
        vn = rollout["variation_number"]
        # The filtered file may use either 'filtered_indicators' or 'detected_indicators'
        indicators = rollout.get("filtered_indicators", rollout.get("detected_indicators", []))
        result[vn] = indicators
    return result


# ──────────────────────────────────────────────────────────────────────
# Per-variation extraction
# ──────────────────────────────────────────────────────────────────────

def get_binary(indicators: list) -> int:
    return 1 if len(indicators) > 0 else 0


def get_indicator_names(indicators: list) -> set:
    return {ind["indicator_name"] for ind in indicators}


def get_indicator_names_at_turn(indicators: list, turn_number: int) -> set:
    return {ind["indicator_name"] for ind in indicators if ind.get("turn_number") == turn_number}


def get_sentences(indicators: list) -> set:
    return {normalize_sentence(ind["sentence"]) for ind in indicators}


def get_sentence_indicator_pairs(indicators: list) -> set:
    """Return set of (normalized_sentence, indicator_name) tuples."""
    return {
        (normalize_sentence(ind["sentence"]), ind["indicator_name"])
        for ind in indicators
    }


def get_confidence_map(indicators: list) -> dict:
    """Return {(normalized_sentence, indicator_name): confidence}."""
    result = {}
    for ind in indicators:
        key = (normalize_sentence(ind["sentence"]), ind["indicator_name"])
        # If duplicate, keep max confidence
        result[key] = max(result.get(key, 0), ind["confidence"])
    return result


# ──────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_consistency(results_dir: str) -> dict:
    results_path = Path(results_dir)

    # Discover run files
    run_files = sorted(results_path.glob("rollout_run*.json"))
    n_runs = len(run_files)
    if n_runs < 2:
        print(f"ERROR: Need at least 2 run files, found {n_runs} in {results_dir}")
        sys.exit(1)

    # Load runs
    runs = []
    for rf in run_files:
        runs.append(load_run(rf))
        print(f"Loaded {rf.name}")

    # Load filtered result
    filtered_path = results_path / "rollout_aggregated.json"
    if not filtered_path.exists():
        filtered_path = results_path / "rollout_filtered.json"
    filtered = load_filtered(filtered_path)
    print(f"Loaded filtered: {filtered_path.name}")

    # Get all variation numbers
    all_variations = sorted(set().union(*(r.keys() for r in runs), filtered.keys()))
    n_vars = len(all_variations)
    print(f"\nVariations: {all_variations}")
    print(f"Number of runs: {n_runs}")
    print()

    R = lambda x: round(float(x), 4)  # round helper for JSON
    report = {}

    # ── 1. Variation-level binary agreement ──────────────────────────
    print("=" * 70)
    print("1. VARIATION-LEVEL BINARY AGREEMENT (any indicators detected?)")
    print("=" * 70)

    binary_per_run = []
    for run in runs:
        binary_per_run.append([get_binary(run.get(v, [])) for v in all_variations])

    # Print per-variation breakdown
    print(f"\n{'Var':>4}", end="")
    for i in range(n_runs):
        print(f"  Run{i+1}", end="")
    print("  Filtered  Agree?")
    print("-" * (10 + 7 * n_runs + 12))

    full_agree_count = 0
    for j, v in enumerate(all_variations):
        run_vals = [binary_per_run[i][j] for i in range(n_runs)]
        filt_val = get_binary(filtered.get(v, []))
        agree = "YES" if len(set(run_vals)) == 1 else "NO"
        if agree == "YES":
            full_agree_count += 1
        print(f"  {v:>2}", end="")
        for rv in run_vals:
            print(f"  {'  +' if rv else '  -'}", end="  ")
        print(f"  {'  +' if filt_val else '  -'}      {agree}")

    full_agree_rate = full_agree_count / n_vars
    print(f"\nFull agreement rate (all {n_runs} runs agree): {full_agree_rate:.3f} ({full_agree_count}/{n_vars})")

    # Pairwise Cohen's kappa
    pairwise_kappas = []
    for (i, j) in combinations(range(n_runs), 2):
        ck = cohen_kappa(binary_per_run[i], binary_per_run[j])
        pairwise_kappas.append(ck)
        print(f"  Cohen's kappa (Run{i+1} vs Run{j+1}): {ck:.4f}")

    report["binary_agreement"] = {
        "full_agreement_rate": R(full_agree_rate),
        "pairwise_cohen_kappa": {
            f"run{i+1}_vs_run{j+1}": R(cohen_kappa(binary_per_run[i], binary_per_run[j]))
            for (i, j) in combinations(range(n_runs), 2)
        },
    }

    # ── 1b. Per-turn binary agreement ────────────────────────────────
    print("\n" + "=" * 70)
    print("1b. PER-TURN BINARY AGREEMENT (any indicators at each turn?)")
    print("=" * 70)

    # Find max turn across all runs and variations
    max_turn = 0
    for run in runs:
        for v in all_variations:
            for ind in run.get(v, []):
                max_turn = max(max_turn, ind.get("turn_number", 0))

    n_all_pairs = 0
    n_nontrivial_pairs = 0
    agree_all = 0
    agree_nontrivial = 0
    for v in all_variations:
        for turn_num in range(1, max_turn + 1):
            binaries = [1 if any(ind.get("turn_number") == turn_num for ind in run.get(v, [])) else 0 for run in runs]
            n_all_pairs += 1
            all_agree = len(set(binaries)) == 1
            if all_agree:
                agree_all += 1
            # Non-trivial: at least one run found indicators
            if not all(b == 0 for b in binaries):
                n_nontrivial_pairs += 1
                if all_agree:
                    agree_nontrivial += 1

    per_turn_agree_rate = agree_all / n_all_pairs if n_all_pairs > 0 else 1.0
    per_turn_agree_rate_nontrivial = agree_nontrivial / n_nontrivial_pairs if n_nontrivial_pairs > 0 else 1.0
    print(f"  All (variation, turn) pairs: {n_all_pairs}")
    print(f"  Agreement rate (all pairs):        {per_turn_agree_rate:.3f} ({agree_all}/{n_all_pairs})")
    print(f"  Non-trivial pairs (>=1 run has indicators): {n_nontrivial_pairs}")
    print(f"  Agreement rate (non-trivial only): {per_turn_agree_rate_nontrivial:.3f} ({agree_nontrivial}/{n_nontrivial_pairs})")

    report["binary_agreement_per_turn"] = {
        "n_all_pairs": n_all_pairs,
        "full_agreement_rate": R(per_turn_agree_rate),
        "n_nontrivial_pairs": n_nontrivial_pairs,
        "nontrivial_agreement_rate": R(per_turn_agree_rate_nontrivial),
    }

    # ── 2. Indicator-name-level Jaccard ──────────────────────────────
    print("\n" + "=" * 70)
    print("2. INDICATOR-NAME SET AGREEMENT (Jaccard similarity)")
    print("=" * 70)

    name_jaccards_pairwise = defaultdict(list)
    for v in all_variations:
        name_sets = [get_indicator_names(run.get(v, [])) for run in runs]
        for (i, j) in combinations(range(n_runs), 2):
            jac = jaccard(name_sets[i], name_sets[j])
            name_jaccards_pairwise[(i, j)].append(jac)

    for (i, j), jacs in sorted(name_jaccards_pairwise.items()):
        mean_jac = np.mean(jacs)
        print(f"  Run{i+1} vs Run{j+1}: mean Jaccard = {mean_jac:.4f}  (per-var: {[f'{x:.2f}' for x in jacs]})")

    overall_name_jaccard = np.mean([v for vals in name_jaccards_pairwise.values() for v in vals])
    print(f"\n  Overall mean Jaccard (indicator names): {overall_name_jaccard:.4f}")

    report["indicator_name_jaccard"] = {
        f"run{i+1}_vs_run{j+1}": R(np.mean(jacs))
        for (i, j), jacs in name_jaccards_pairwise.items()
    }
    report["indicator_name_jaccard"]["overall_mean"] = R(overall_name_jaccard)

    # ── 2b. Per-turn indicator-name Jaccard ──────────────────────────
    print("\n" + "=" * 70)
    print("2b. PER-TURN INDICATOR-NAME AGREEMENT (Jaccard similarity)")
    print("=" * 70)

    # All pairs (including all-empty turns, Jaccard=1.0 for both empty)
    per_turn_jac_all_pairwise = defaultdict(list)
    # Non-trivial pairs only (at least one run has indicators at this turn)
    per_turn_jac_nontrivial_pairwise = defaultdict(list)
    for v in all_variations:
        for turn_num in range(1, max_turn + 1):
            name_sets = [get_indicator_names_at_turn(run.get(v, []), turn_num) for run in runs]
            nontrivial = not all(len(s) == 0 for s in name_sets)
            for (i, j) in combinations(range(n_runs), 2):
                jac = jaccard(name_sets[i], name_sets[j])
                per_turn_jac_all_pairwise[(i, j)].append(jac)
                if nontrivial:
                    per_turn_jac_nontrivial_pairwise[(i, j)].append(jac)

    print("  Including all-empty turns:")
    for (i, j), jacs in sorted(per_turn_jac_all_pairwise.items()):
        print(f"    Run{i+1} vs Run{j+1}: mean Jaccard = {np.mean(jacs):.4f}  ({len(jacs)} pairs)")
    overall_per_turn_name_jaccard = np.mean(
        [v for vals in per_turn_jac_all_pairwise.values() for v in vals]
    ) if per_turn_jac_all_pairwise else 1.0
    print(f"    Overall: {overall_per_turn_name_jaccard:.4f}")

    print("  Non-trivial only (>=1 run has indicators):")
    for (i, j), jacs in sorted(per_turn_jac_nontrivial_pairwise.items()):
        print(f"    Run{i+1} vs Run{j+1}: mean Jaccard = {np.mean(jacs):.4f}  ({len(jacs)} pairs)")
    overall_per_turn_name_jaccard_nontrivial = np.mean(
        [v for vals in per_turn_jac_nontrivial_pairwise.values() for v in vals]
    ) if per_turn_jac_nontrivial_pairwise else 1.0
    print(f"    Overall: {overall_per_turn_name_jaccard_nontrivial:.4f}")

    report["indicator_name_jaccard_per_turn"] = {
        f"run{i+1}_vs_run{j+1}": R(np.mean(jacs))
        for (i, j), jacs in per_turn_jac_all_pairwise.items()
    }
    report["indicator_name_jaccard_per_turn"]["overall_mean"] = R(overall_per_turn_name_jaccard)
    report["indicator_name_jaccard_per_turn"]["overall_mean_nontrivial"] = R(overall_per_turn_name_jaccard_nontrivial)

    # ── 3. Sentence-level Jaccard ────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. SENTENCE-LEVEL AGREEMENT (Jaccard similarity)")
    print("=" * 70)

    sent_jaccards_pairwise = defaultdict(list)
    for v in all_variations:
        sent_sets = [get_sentences(run.get(v, [])) for run in runs]
        for (i, j) in combinations(range(n_runs), 2):
            jac = jaccard(sent_sets[i], sent_sets[j])
            sent_jaccards_pairwise[(i, j)].append(jac)

    for (i, j), jacs in sorted(sent_jaccards_pairwise.items()):
        mean_jac = np.mean(jacs)
        print(f"  Run{i+1} vs Run{j+1}: mean Jaccard = {mean_jac:.4f}  (per-var: {[f'{x:.2f}' for x in jacs]})")

    overall_sent_jaccard = np.mean([v for vals in sent_jaccards_pairwise.values() for v in vals])
    print(f"\n  Overall mean Jaccard (sentences): {overall_sent_jaccard:.4f}")

    report["sentence_jaccard"] = {
        f"run{i+1}_vs_run{j+1}": R(np.mean(jacs))
        for (i, j), jacs in sent_jaccards_pairwise.items()
    }
    report["sentence_jaccard"]["overall_mean"] = R(overall_sent_jaccard)

    # ── 3b. (sentence, indicator_name) pair Jaccard ──────────────────
    print("\n" + "=" * 70)
    print("3b. (SENTENCE, INDICATOR_NAME) PAIR AGREEMENT (Jaccard)")
    print("=" * 70)

    pair_jaccards_pairwise = defaultdict(list)
    for v in all_variations:
        pair_sets = [get_sentence_indicator_pairs(run.get(v, [])) for run in runs]
        for (i, j) in combinations(range(n_runs), 2):
            jac = jaccard(pair_sets[i], pair_sets[j])
            pair_jaccards_pairwise[(i, j)].append(jac)

    for (i, j), jacs in sorted(pair_jaccards_pairwise.items()):
        mean_jac = np.mean(jacs)
        print(f"  Run{i+1} vs Run{j+1}: mean Jaccard = {mean_jac:.4f}")

    overall_pair_jaccard = np.mean([v for vals in pair_jaccards_pairwise.values() for v in vals])
    print(f"\n  Overall mean Jaccard (sentence+indicator pairs): {overall_pair_jaccard:.4f}")

    report["sentence_indicator_pair_jaccard"] = {
        f"run{i+1}_vs_run{j+1}": R(np.mean(jacs))
        for (i, j), jacs in pair_jaccards_pairwise.items()
    }
    report["sentence_indicator_pair_jaccard"]["overall_mean"] = R(overall_pair_jaccard)

    # ── 4. Confidence consistency ────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. CONFIDENCE CONSISTENCY (matched sentence+indicator pairs)")
    print("=" * 70)

    all_conf_diffs = []
    all_conf_values = defaultdict(list)  # key -> list of confidences from different runs

    for v in all_variations:
        conf_maps = [get_confidence_map(run.get(v, [])) for run in runs]
        # Find all keys that appear in at least 2 runs
        all_keys = set()
        for cm in conf_maps:
            all_keys.update(cm.keys())

        for key in all_keys:
            confs = [cm[key] for cm in conf_maps if key in cm]
            if len(confs) >= 2:
                all_conf_values[key] = confs
                # Pairwise absolute differences
                for ci, cj in combinations(confs, 2):
                    all_conf_diffs.append(abs(ci - cj))

    if all_conf_diffs:
        mean_diff = np.mean(all_conf_diffs)
        median_diff = np.median(all_conf_diffs)
        std_diff = np.std(all_conf_diffs)
        max_diff = np.max(all_conf_diffs)
        print(f"  Matched pairs found: {len(all_conf_values)}")
        print(f"  Pairwise confidence abs differences:")
        print(f"    Mean:   {mean_diff:.4f}")
        print(f"    Median: {median_diff:.4f}")
        print(f"    Std:    {std_diff:.4f}")
        print(f"    Max:    {max_diff:.4f}")

        # Per-key std
        per_key_stds = []
        for key, confs in all_conf_values.items():
            if len(confs) >= 2:
                per_key_stds.append(np.std(confs))
        if per_key_stds:
            print(f"\n  Per-indicator confidence std (across runs):")
            print(f"    Mean std: {np.mean(per_key_stds):.4f}")
            print(f"    Max std:  {np.max(per_key_stds):.4f}")

        report["confidence_consistency"] = {
            "n_matched_pairs": len(all_conf_values),
            "mean_abs_diff": R(mean_diff),
            "median_abs_diff": R(median_diff),
            "max_abs_diff": R(max_diff),
            "mean_per_indicator_std": R(np.mean(per_key_stds)) if per_key_stds else None,
        }
    else:
        print("  No matched indicator pairs found across runs.")
        report["confidence_consistency"] = {"n_matched_pairs": 0}

    # ── 5. Filtered vs runs consistency ──────────────────────────────
    print("\n" + "=" * 70)
    print("5. FILTERED RESULT vs INDIVIDUAL RUNS")
    print("=" * 70)

    filtered_binary = [get_binary(filtered.get(v, [])) for v in all_variations]

    for i in range(n_runs):
        run_binary = binary_per_run[i]
        ck = cohen_kappa(run_binary, filtered_binary)
        agree = sum(a == b for a, b in zip(run_binary, filtered_binary)) / n_vars

        # Sentence-level Jaccard
        sent_jacs = []
        name_jacs = []
        for v in all_variations:
            run_sents = get_sentences(runs[i].get(v, []))
            filt_sents = get_sentences(filtered.get(v, []))
            sent_jacs.append(jaccard(run_sents, filt_sents))

            run_names = get_indicator_names(runs[i].get(v, []))
            filt_names = get_indicator_names(filtered.get(v, []))
            name_jacs.append(jaccard(run_names, filt_names))

        print(f"\n  Run{i+1} vs Filtered:")
        print(f"    Binary agreement:        {agree:.3f}")
        print(f"    Cohen's kappa:           {ck:.4f}")
        print(f"    Mean sentence Jaccard:   {np.mean(sent_jacs):.4f}")
        print(f"    Mean ind-name Jaccard:   {np.mean(name_jacs):.4f}")

    # Majority vote comparison
    print(f"\n  Majority Vote vs Filtered:")
    majority_binary = []
    for j in range(n_vars):
        votes = sum(binary_per_run[i][j] for i in range(n_runs))
        majority_binary.append(1 if votes > n_runs / 2 else 0)

    mv_agree = sum(a == b for a, b in zip(majority_binary, filtered_binary)) / n_vars
    mv_kappa = cohen_kappa(majority_binary, filtered_binary)
    print(f"    Binary agreement:        {mv_agree:.3f}")
    print(f"    Cohen's kappa:           {mv_kappa:.4f}")

    # Check disagreements
    disagreements = []
    for j, v in enumerate(all_variations):
        if majority_binary[j] != filtered_binary[j]:
            disagreements.append(v)
    if disagreements:
        print(f"    Disagreement on variations: {disagreements}")
    else:
        print(f"    No disagreements between majority vote and filtered result.")

    report["filtered_vs_runs"] = {
        f"run{i+1}": {
            "binary_agreement": R(sum(a == b for a, b in zip(binary_per_run[i], filtered_binary)) / n_vars),
            "cohen_kappa": R(cohen_kappa(binary_per_run[i], filtered_binary)),
        }
        for i in range(n_runs)
    }
    report["filtered_vs_runs"]["majority_vote"] = {
        "binary_agreement": R(mv_agree),
        "cohen_kappa": R(mv_kappa),
        "disagreement_variations": disagreements,
    }

    # ── 6. Per-variation detail summary ──────────────────────────────
    print("\n" + "=" * 70)
    print("6. PER-VARIATION DETAIL SUMMARY")
    print("=" * 70)

    per_variation = {}
    for v in all_variations:
        run_counts = [len(runs[i].get(v, [])) for i in range(n_runs)]
        filt_count = len(filtered.get(v, []))
        run_names = [get_indicator_names(runs[i].get(v, [])) for i in range(n_runs)]
        filt_names = get_indicator_names(filtered.get(v, []))
        all_names = sorted(set().union(*run_names))

        # Pairwise sentence Jaccard for this variation
        sent_sets = [get_sentences(run.get(v, [])) for run in runs]
        var_sent_jacs = [
            jaccard(sent_sets[i], sent_sets[j])
            for (i, j) in combinations(range(n_runs), 2)
        ]

        per_variation[str(v)] = {
            "indicator_counts": {"runs": run_counts, "filtered": filt_count},
            "binary": {"runs": [int(c > 0) for c in run_counts], "filtered": int(filt_count > 0)},
            "indicator_types": {
                name: {
                    "runs": [name in run_names[i] for i in range(n_runs)],
                    "filtered": name in filt_names,
                }
                for name in all_names
            },
            "mean_sentence_jaccard": R(np.mean(var_sent_jacs)) if var_sent_jacs else None,
        }

        print(f"\n  Variation {v}:")
        print(f"    Indicator counts: {' / '.join(str(c) for c in run_counts)} (runs)  |  {filt_count} (filtered)")
        if all_names:
            print(f"    Indicator types detected across runs:")
            for name in all_names:
                presence = ["X" if name in run_names[i] else "." for i in range(n_runs)]
                filt_mark = "X" if name in filt_names else "."
                print(f"      {' '.join(presence)} | {filt_mark}  {name}")
        else:
            print(f"    No indicators detected in any run.")

    report["per_variation"] = per_variation

    # ── 7. Overall summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Binary agreement (all runs):               {full_agree_rate:.3f}")
    print(f"  Binary agreement (per-turn, all):           {per_turn_agree_rate:.3f}")
    print(f"  Binary agreement (per-turn, non-trivial):   {per_turn_agree_rate_nontrivial:.3f}")
    print(f"  Mean indicator-name Jaccard:                {overall_name_jaccard:.4f}")
    print(f"  Mean ind-name Jaccard (per-turn, all):      {overall_per_turn_name_jaccard:.4f}")
    print(f"  Mean ind-name Jaccard (per-turn, non-triv): {overall_per_turn_name_jaccard_nontrivial:.4f}")
    print(f"  Mean sentence Jaccard:                      {overall_sent_jaccard:.4f}")
    print(f"  Mean sent+ind pair Jaccard:                 {overall_pair_jaccard:.4f}")
    if all_conf_diffs:
        print(f"  Mean confidence abs diff:                   {np.mean(all_conf_diffs):.4f}")
    print(f"  Majority vote vs filtered agree:            {mv_agree:.3f}")
    print(f"  Majority vote vs filtered kappa:            {mv_kappa:.4f}")

    report["summary"] = {
        "n_variations": n_vars,
        "n_runs": n_runs,
        "binary_agreement_rate": R(full_agree_rate),
        "binary_agreement_rate_per_turn": R(per_turn_agree_rate),
        "binary_agreement_rate_per_turn_nontrivial": R(per_turn_agree_rate_nontrivial),
        "mean_indicator_name_jaccard": R(overall_name_jaccard),
        "mean_indicator_name_jaccard_per_turn": R(overall_per_turn_name_jaccard),
        "mean_indicator_name_jaccard_per_turn_nontrivial": R(overall_per_turn_name_jaccard_nontrivial),
        "mean_sentence_jaccard": R(overall_sent_jaccard),
        "mean_sentence_indicator_pair_jaccard": R(overall_pair_jaccard),
        "mean_confidence_abs_diff": R(np.mean(all_conf_diffs)) if all_conf_diffs else None,
        "majority_vote_vs_filtered_agreement": R(mv_agree),
        "majority_vote_vs_filtered_kappa": R(mv_kappa),
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate indicator classification consistency across runs")
    parser.add_argument("results_dir", help="Directory containing rollout_run*.json and rollout_filtered.json/rollout_aggregated.json")
    args = parser.parse_args()

    report = evaluate_consistency(args.results_dir)

    # Always save to the results folder
    out_path = Path(args.results_dir) / "consistency_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
