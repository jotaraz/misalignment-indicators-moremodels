"""
Generate an interactive HTML dashboard for sentence-average probe scores
against misalignment turn-level ground truth.

Pre-computes sentence-level averages server-side so per-token scores are NOT
embedded in the HTML, resulting in much smaller file sizes.

Usage:
    python probe_eval/visualize_misalignment_sentence.py
    python probe_eval/visualize_misalignment_sentence.py --results-subdir v2_4_combined_v3_span
"""

import argparse
import json
import re
from pathlib import Path

import probe_eval.visualize_misalignment as _vm
from probe_eval.visualize_misalignment import (
    HTML_TEMPLATE as _BASE_TEMPLATE,
    build_data as _build_data_base,
)

# Same sentence splitting regex used by probe training and evaluation
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


# ---------------------------------------------------------------------------
# Server-side sentence helpers
# ---------------------------------------------------------------------------

def _segment_tokens_py(tokens: list[str]) -> list[tuple[str, int, int]]:
    """Parse token list by role markers. Returns [(role, start_idx, end_idx)]."""
    segments: list[tuple[str, int, int]] = []
    role = None
    start = 0
    for i, t in enumerate(tokens):
        if t in ('<|system|>', '<|user|>', '<|assistant|>'):
            if role is not None:
                segments.append((role, start, i))
            role = t[2:-2]
            start = i + 1
    if role is not None:
        segments.append((role, start, len(tokens)))
    return segments


def _sentence_boundaries(tokens: list[str]) -> list[list[int]]:
    """Compute sentence boundary groups for a token list.

    Returns list of groups where each group is a list of token indices
    (relative to the input list). Mirrors the JS getSentenceBoundaries.
    """
    if not tokens:
        return []
    text = "".join(tokens)
    split_positions = sorted(
        set(m.end() for m in _SENTENCE_SPLIT_RE.finditer(text))
    )
    groups: list[list[int]] = [[]]
    group_tokens: list[list[str]] = [[]]
    char_pos = 0
    split_idx = 0
    for i, tok in enumerate(tokens):
        tok_mid = char_pos + len(tok) / 2
        while split_idx < len(split_positions) and tok_mid >= split_positions[split_idx]:
            split_idx += 1
            groups.append([])
            group_tokens.append([])
        groups[-1].append(i)
        group_tokens[-1].append(tok)
        char_pos += len(tok)

    # Merge short sentences (< 5 chars stripped) into adjacent ones
    i = 0
    while i < len(groups):
        sent_text = "".join(group_tokens[i]).strip()
        if len(sent_text) < 5:
            if i > 0:
                groups[i - 1].extend(groups[i])
                group_tokens[i - 1].extend(group_tokens[i])
                groups.pop(i)
                group_tokens.pop(i)
            elif i < len(groups) - 1:
                groups[i + 1] = groups[i] + groups[i + 1]
                group_tokens[i + 1] = group_tokens[i] + group_tokens[i + 1]
                groups.pop(i)
                group_tokens.pop(i)
            else:
                i += 1
        else:
            i += 1

    return [g for g in groups if g]


def _sentence_avg(scores: list, indices: list[int]) -> float | None:
    """Average score for given token indices, skipping None values."""
    vals = [scores[i] for i in indices if i < len(scores) and scores[i] is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Data builder — pre-computes sentence averages, drops per-token scores
# ---------------------------------------------------------------------------

def build_data(base_dir: Path, results_subdir: str) -> dict:
    """Build data with pre-computed sentence-level scores (no per-token scores)."""
    data = _build_data_base(base_dir, results_subdir)

    # sentence_scores[probe][beh][ri_str][seg_start_str] = [avg0, avg1, ...]
    sentence_scores: dict = {}
    sentence_extents: dict[str, list[float]] = {}

    for probe in data["probes"]:
        sentence_scores[probe] = {}
        sent_min = float("inf")
        sent_max = float("-inf")

        for beh in data["behaviors"]:
            beh_tok_scores = data["scores"].get(probe, {}).get(beh, {})
            sentence_scores[probe][beh] = {}

            for idx_str, tok_scores in beh_tok_scores.items():
                tokens = data["tokens"].get(beh, {}).get(idx_str, [])
                if not tokens:
                    continue
                segs = _segment_tokens_py(tokens)
                seg_avgs: dict[str, list] = {}
                for role, seg_s, seg_e in segs:
                    if role != "assistant":
                        continue
                    seg_tokens = tokens[seg_s:seg_e]
                    seg_scores = tok_scores[seg_s:seg_e]
                    boundaries = _sentence_boundaries(seg_tokens)
                    avgs = []
                    for group in boundaries:
                        avg = _sentence_avg(seg_scores, group)
                        if avg is not None:
                            sent_min = min(sent_min, avg)
                            sent_max = max(sent_max, avg)
                            avgs.append(round(avg, 2))
                        else:
                            avgs.append(None)
                    seg_avgs[str(seg_s)] = avgs
                sentence_scores[probe][beh][idx_str] = seg_avgs

        if sent_min == float("inf"):
            sent_min, sent_max = -1.0, 1.0
        sentence_extents[probe] = [round(sent_min, 4), round(sent_max, 4)]

    # Drop per-token scores (the bulk of file size)
    del data["scores"]
    data["sentence_scores"] = sentence_scores
    data["sentence_extents"] = sentence_extents

    # Round per_turn_scores in results
    for probe_res in data["results"].values():
        for beh_res in probe_res.values():
            for rollout in beh_res.get("per_rollout", []):
                if rollout.get("probe_score") is not None:
                    rollout["probe_score"] = round(rollout["probe_score"], 4)
                for ts in rollout.get("per_turn_scores", []):
                    if ts.get("score") is not None:
                        ts["score"] = round(ts["score"], 2)

    # Load per-indicator val_clean_label_f1 thresholds for FP/FN analysis
    summary_path = (
        base_dir / "probe_eval" / "results" / results_subdir
        / "misalignment_gt_summary_dev.json"
    )
    fp_thresholds: dict[str, float] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        per_concept = (
            summary.get("per_layer", {}).get(_vm.LAYER, {}).get("per_concept", {})
        )
        for concept, cdata in per_concept.items():
            vt = cdata.get("val_clean_label_f1", {})
            t = vt.get("threshold")
            if t is not None:
                fp_thresholds[concept] = round(t, 6)
    data["fp_thresholds"] = fp_thresholds

    # Override per-turn thresholds (used by getProbeDetection in the base
    # template for turn-level FP/FN badges) with val_clean_label_f1 thresholds
    # so both sentence-level and turn-level FP/FN use the same threshold source.
    if fp_thresholds:
        from probe_eval.visualize_misalignment import EVAL_MODES
        for mode in EVAL_MODES:
            data["thresholds"][mode] = dict(fp_thresholds)

    return data


# ---------------------------------------------------------------------------
# Build modified HTML template via targeted string replacements
# ---------------------------------------------------------------------------

_SENTENCE_JS = r"""
const SENT_RE = /(?<=[.!?])\s+|\n+/g;

function getSentenceBoundaries(tokens) {
    if (!tokens.length) return [];
    const text = tokens.join('');
    const splitPositions = [];
    let m;
    SENT_RE.lastIndex = 0;
    while ((m = SENT_RE.exec(text)) !== null) {
        splitPositions.push(m.index + m[0].length);
    }
    const uniq = [...new Set(splitPositions)].sort((a,b) => a-b);
    const groups = [[]];
    let charPos = 0, spIdx = 0;
    for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        const mid = charPos + tok.length / 2;
        while (spIdx < uniq.length && mid >= uniq[spIdx]) { spIdx++; groups.push([]); }
        groups[groups.length-1].push(i);
        charPos += tok.length;
    }
    for (let i = 0; i < groups.length; ) {
        const t = groups[i].map(j => tokens[j]).join('').trim();
        if (t.length < 5) {
            if (i > 0) { groups[i-1].push(...groups[i]); groups.splice(i,1); }
            else if (i < groups.length-1) { groups[i+1]=[...groups[i],...groups[i+1]]; groups.splice(i,1); }
            else { i++; }
        } else { i++; }
    }
    return groups.filter(g => g.length > 0);
}

function renderSentences(tokens, scores, segStart) {
    const boundaries = getSentenceBoundaries(tokens);
    const fpThresh = DATA.fp_thresholds || {};
    const segKey = String(segStart);
    const riKey = String(curRi);
    return boundaries.map((idxs, sentIdx) => {
        const text = idxs.map(i => tokens[i]).join('');
        // Look up pre-computed sentence average
        const score = DATA.sentence_scores?.[curProbe]?.[curBeh]?.[riKey]?.[segKey]?.[sentIdx] ?? null;
        // Check which indicators fire on this sentence
        const firing = [];
        for (const probe of DATA.probes) {
            const thresh = fpThresh[probe];
            if (thresh == null) continue;
            const avg = DATA.sentence_scores?.[probe]?.[curBeh]?.[riKey]?.[segKey]?.[sentIdx];
            if (avg != null && avg > thresh) firing.push({name:probe, avg, thresh});
        }
        const t = esc(text).replace(/\n/g,'<br>');
        let html = '';
        if (score == null) {
            html = `<span class="sent un">${t}</span>`;
        } else {
            const bg = scoreColor(score);
            html = `<span class="sent" style="background:${bg}" title="Sentence avg: ${score.toFixed(4)}">${t}</span>`;
        }
        if (firing.length > 0) {
            const badges = firing.map(f => {
                const label = f.name.replace(/_/g,' ');
                return `<span class="ind-fire" title="${label}: avg=${f.avg.toFixed(4)} > thresh=${f.thresh.toFixed(4)}">${label}</span>`;
            }).join('');
            html += `<span class="ind-fires">[${badges}]</span>`;
        }
        return html;
    }).join('');
}

function getRolloutSentFP(beh, rolloutIndex) {
    const fpThresh = DATA.fp_thresholds || {};
    if (!Object.keys(fpThresh).length) return false;
    const tokens = DATA.tokens[beh]?.[String(rolloutIndex)] || [];
    if (!tokens.length) return false;
    const variation = rolloutIndex + 1;
    const nTurns = DATA.n_turns?.[beh]?.[String(variation)] || 20;
    const gtStatus = getTurnGtStatus(variation, nTurns);
    const turnMap = DATA.turn_maps?.[beh]?.[String(rolloutIndex)] || [];
    const dummyScores = new Array(tokens.length).fill(null);
    const segs = segmentTokens(tokens, dummyScores);
    const turns = groupTurns(segs, turnMap);
    const riKey = String(rolloutIndex);
    for (const turn of turns) {
        const tn = turn.turn;
        if (tn === 0) continue;
        const gt = gtStatus[tn] || {gt_pos: false};
        if (gt.gt_pos || gt.ambiguous) continue;
        for (const seg of turn.segments) {
            if (seg.role !== 'assistant') continue;
            const segKey = String(seg.start);
            const boundaries = getSentenceBoundaries(seg.tokens);
            for (let si = 0; si < boundaries.length; si++) {
                for (const probe of DATA.probes) {
                    const thresh = fpThresh[probe];
                    if (thresh == null) continue;
                    const avg = DATA.sentence_scores?.[probe]?.[beh]?.[riKey]?.[segKey]?.[si];
                    if (avg != null && avg > thresh) return true;
                }
            }
        }
    }
    return false;
}

"""

_t = _BASE_TEMPLATE

# 1. Add .sent CSS alongside .tok styles
_t = _t.replace(
    ".tok.un{color:#999}",
    ".tok.un{color:#999}\n"
    ".sent{display:inline;padding:2px 1px;font-family:monospace;"
    "font-size:.78rem;cursor:default;white-space:pre-wrap;border-radius:2px}"
    "\n.sent.un{color:#999}"
    "\n.ind-fires{display:inline;margin-left:1px;font-size:0}"
    "\n.ind-fire{display:inline-block;padding:0 3px;margin:0 1px;"
    "border-radius:2px;font-size:.55rem;background:#fdd;color:#c00;"
    "border:1px solid #eaa;cursor:default;vertical-align:super;"
    "line-height:1.2;max-width:90px;overflow:hidden;text-overflow:ellipsis;"
    "white-space:nowrap}",
)

# 2. Update legend
_t = _t.replace("<span>Low score</span>", "<span>Low sentence avg</span>")
_t = _t.replace("<span>High score</span>", "<span>High sentence avg</span>")

# 3. Update subtitle
_t = _t.replace(
    "Layer 30 probe scores with misalignment turn-level ground truth. "
    "Hover tokens for scores.",
    "Layer 30 sentence-average probe scores with misalignment ground truth. "
    "Hover sentences for average scores.",
)

# 4. Insert sentence JS functions before renderProbeTabs
_t = _t.replace(
    "function renderProbeTabs()",
    _SENTENCE_JS + "function renderProbeTabs()",
)

# 5. Use sentence extents for color scale
_t = _t.replace(
    "DATA.extents[curProbe]",
    "DATA.sentence_extents[curProbe]",
)

# 6. Replace per-token score loading with dummy scores (tokens still needed for text)
_t = _t.replace(
    "const scores = DATA.scores[curProbe]?.[curBeh]?.[String(curRi)] || [];",
    "const scores = new Array(tokens.length).fill(null);",
)

# 7. Update threshold bar label
_t = _t.replace(
    "(joint optimized F1)",
    "(val_clean_label_f1)",
)

# 8. Use renderSentences for assistant content (pass seg.start for segment key lookup)
_t = _t.replace(
    "renderTokens(seg.tokens, seg.scores)",
    "renderSentences(seg.tokens, seg.scores, seg.start)",
)

# 8. Add sentence-level FP badge on rollout tabs
_t = _t.replace(
    """const fpfn = (hasFP ? ' <span class="tab-fp">FP</span>' : '') + (hasFN ? ' <span class="tab-fn">FN</span>' : '');""",
    """const hasSentFP = getRolloutSentFP(curBeh, r.rollout_index);
        const fpfn = (hasFP ? ' <span class="tab-fp">FP</span>' : '') + (hasFN ? ' <span class="tab-fn">FN</span>' : '') + (hasSentFP ? ' <span class="tab-fp">Sent-FP</span>' : '');""",
)

# 9. Add per-turn sentence FP label in turn header
_t = _t.replace(
    "if (firedHtml) html += `<div style=\"padding:2px 12px\">${firedHtml}</div>`;",
    """if (firedHtml) html += `<div style="padding:2px 12px">${firedHtml}</div>`;

        // Sentence-level FP check for this turn
        if (!gt.gt_pos && !gt.ambiguous) {
            const turnSentFPs = [];
            for (const seg of turn.segments) {
                if (seg.role !== 'assistant') continue;
                const segKey = String(seg.start);
                const boundaries = getSentenceBoundaries(seg.tokens);
                for (let si = 0; si < boundaries.length; si++) {
                    const sentFiring = [];
                    for (const probe of DATA.probes) {
                        const thresh = (DATA.fp_thresholds||{})[probe];
                        if (thresh == null) continue;
                        const avg = DATA.sentence_scores?.[probe]?.[curBeh]?.[String(curRi)]?.[segKey]?.[si];
                        if (avg != null && avg > thresh) sentFiring.push(probe.replace(/_/g,' '));
                    }
                    if (sentFiring.length) turnSentFPs.push(sentFiring);
                }
            }
            if (turnSentFPs.length > 0) {
                const allFired = [...new Set(turnSentFPs.flat())].sort();
                html += `<div style="padding:2px 12px;font-size:.72rem"><span class="det-badge fp">Sent-FP</span> <span style="color:#c00">${turnSentFPs.length} sentence(s): ${allFired.join(', ')}</span></div>`;
            }
        }""",
)

# 10. Override getProbeDetection to use sentence-level scores from fp_thresholds
#     instead of per-turn scores with joint_optimized thresholds.
#     The function now receives (beh, rolloutIndex, turn) and checks sentence_scores.
_t = _t.replace(
    """function getProbeDetection(turnScores) {
    const modeThresholds = DATA.thresholds[curMode] || {};
    const fired = [];
    for (const probe of DATA.probes) {
        const thresh = modeThresholds[probe];
        if (thresh == null) continue;
        const score = turnScores[probe];
        if (score != null && score > thresh) {
            fired.push(probe);
        }
    }
    return {detected: fired.length > 0, fired_probes: fired};
}""",
    r"""function getProbeDetection(turnScores, turnSegs) {
    const fpThresh = DATA.fp_thresholds || {};
    const riKey = String(curRi);
    const fired = [];
    if (turnSegs) {
        // Sentence-based detection: check if any sentence in this turn fires
        for (const seg of turnSegs) {
            if (seg.role !== 'assistant') continue;
            const segKey = String(seg.start);
            const boundaries = getSentenceBoundaries(seg.tokens);
            for (let si = 0; si < boundaries.length; si++) {
                for (const probe of DATA.probes) {
                    const thresh = fpThresh[probe];
                    if (thresh == null) continue;
                    const avg = DATA.sentence_scores?.[probe]?.[curBeh]?.[riKey]?.[segKey]?.[si];
                    if (avg != null && avg > thresh && !fired.includes(probe)) {
                        fired.push(probe);
                    }
                }
            }
        }
    } else {
        // Fallback to per-turn scores (should not happen in sentence dashboard)
        for (const probe of DATA.probes) {
            const thresh = fpThresh[probe];
            if (thresh == null) continue;
            const score = turnScores[probe];
            if (score != null && score > thresh) fired.push(probe);
        }
    }
    return {detected: fired.length > 0, fired_probes: fired};
}""",
)

# 11. Override getRolloutFpFn to use sentence-based detection
_t = _t.replace(
    """function getRolloutFpFn(beh, rolloutIndex) {
    const variation = getRolloutVariation(beh, rolloutIndex);
    const nTurns = DATA.n_turns?.[beh]?.[String(variation)] || 20;
    const gtStatus = getTurnGtStatus(variation, nTurns);
    // Build per-turn scores across all probes for this rollout
    const allProbeTurnScores = {};
    for (const probe of DATA.probes) {
        const probeRes = DATA.results[probe]?.[beh];
        if (!probeRes) continue;
        const probeRollout = probeRes.per_rollout.find(r => r.rollout_index === rolloutIndex);
        if (!probeRollout) continue;
        for (const ts of (probeRollout.per_turn_scores || [])) {
            if (!allProbeTurnScores[ts.turn]) allProbeTurnScores[ts.turn] = {};
            allProbeTurnScores[ts.turn][probe] = ts.score;
        }
    }
    let hasFP = false, hasFN = false;
    for (let tn = 1; tn <= nTurns; tn++) {
        const gt = gtStatus[tn] || {gt_pos: false, ambiguous: false};
        if (gt.ambiguous) continue;
        const det = getProbeDetection(allProbeTurnScores[tn] || {});
        if (!gt.gt_pos && det.detected) hasFP = true;
        if (gt.gt_pos && !det.detected) hasFN = true;
    }
    return {hasFP, hasFN};
}""",
    r"""function getRolloutFpFn(beh, rolloutIndex) {
    const variation = getRolloutVariation(beh, rolloutIndex);
    const nTurns = DATA.n_turns?.[beh]?.[String(variation)] || 20;
    const gtStatus = getTurnGtStatus(variation, nTurns);
    const fpThresh = DATA.fp_thresholds || {};
    const riKey = String(rolloutIndex);
    const tokens = DATA.tokens[beh]?.[riKey] || [];
    let hasFP = false, hasFN = false;
    if (!tokens.length) return {hasFP, hasFN};
    const dummyScores = new Array(tokens.length).fill(null);
    const segs = segmentTokens(tokens, dummyScores);
    const turnMap = DATA.turn_maps?.[beh]?.[riKey] || [];
    const turns = groupTurns(segs, turnMap);
    for (const turn of turns) {
        const tn = turn.turn;
        if (tn === 0) continue;
        const gt = gtStatus[tn] || {gt_pos: false, ambiguous: false};
        if (gt.ambiguous) continue;
        // Check if any sentence in this turn fires any probe
        let detected = false;
        for (const seg of turn.segments) {
            if (seg.role !== 'assistant') continue;
            const segKey = String(seg.start);
            const boundaries = getSentenceBoundaries(seg.tokens);
            for (let si = 0; si < boundaries.length; si++) {
                for (const probe of DATA.probes) {
                    const thresh = fpThresh[probe];
                    if (thresh == null) continue;
                    const avg = DATA.sentence_scores?.[probe]?.[beh]?.[riKey]?.[segKey]?.[si];
                    if (avg != null && avg > thresh) { detected = true; break; }
                }
                if (detected) break;
            }
            if (detected) break;
        }
        if (!gt.gt_pos && detected) hasFP = true;
        if (gt.gt_pos && !detected) hasFN = true;
    }
    return {hasFP, hasFN};
}""",
)

# 12. Update getProbeDetection call site in renderContent to pass turn segments
_t = _t.replace(
    "const det = getProbeDetection(allProbeTurnScores[tn] || {});",
    "const det = getProbeDetection(allProbeTurnScores[tn] || {}, turn.segments);",
)

# 13. Update threshold display bar to use fp_thresholds
_t = _t.replace(
    "const modeThresholds = DATA.thresholds[curMode] || {};\n"
    "    html += `<div class=\"thresh-bar\">",
    "const modeThresholds = DATA.fp_thresholds || {};\n"
    "    html += `<div class=\"thresh-bar\">",
)

HTML_TEMPLATE = _t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate sentence-average misalignment probe dashboard",
    )
    parser.add_argument(
        "--results-subdir", default="v2_3",
        help="Subdirectory under probe_eval/results/",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory")
    parser.add_argument("--layer", default=None, help="Layer to visualize (e.g. 27). Auto-detected if not set.")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument(
        "--ignore-probes", nargs="*", default=None,
        help="Probe names to exclude from the dashboard (e.g. internal_output_divergence)",
    )
    parser.add_argument(
        "--include-behaviors", default=None,
        help="Comma-separated fnmatch patterns for behaviors to include (default: all)",
    )
    parser.add_argument(
        "--exclude-behaviors", default=None,
        help="Comma-separated fnmatch patterns for behaviors to exclude",
    )
    parser.add_argument(
        "--tuned-thresholds", default=None,
        help="Path to tuned_thresholds.json file to use for FP/FN thresholds",
    )
    parser.add_argument(
        "--tuned-threshold-version", default=None,
        help="Version name in tuned thresholds file (e.g. fpr_0.02)",
    )
    args = parser.parse_args()

    if args.layer is not None:
        _vm.LAYER = f"layer{args.layer}"

    base_dir = Path(args.base_dir)
    print(f"Building data (with pre-computed sentence averages, layer={args.layer or 'default'})...")
    data = build_data(base_dir, args.results_subdir)

    # Override thresholds from tuned_thresholds.json if provided
    if args.tuned_thresholds:
        tuned_path = Path(args.tuned_thresholds)
        if tuned_path.exists():
            with open(tuned_path) as f:
                tuned_data = json.load(f)
            version = args.tuned_threshold_version
            if version is None:
                # Use first version
                version = next(iter(tuned_data.get("versions", {})), None)
            if version and version in tuned_data.get("versions", {}):
                layer_key = _vm.LAYER
                tuned_thresholds = (
                    tuned_data["versions"][version]
                    .get("per_layer", {})
                    .get(layer_key, {})
                    .get("thresholds", {})
                )
                if tuned_thresholds:
                    data["fp_thresholds"] = {
                        k: round(v, 6) for k, v in tuned_thresholds.items()
                    }
                    # Also override per-mode thresholds
                    from probe_eval.visualize_misalignment import EVAL_MODES
                    for mode in EVAL_MODES:
                        data["thresholds"][mode] = dict(data["fp_thresholds"])
                    print(f"  Loaded tuned thresholds: {version} ({len(tuned_thresholds)} probes)")
                else:
                    print(f"  WARNING: No thresholds for {layer_key} in {version}")
            else:
                print(f"  WARNING: Version '{version}' not found in {tuned_path}")
        else:
            print(f"  WARNING: Tuned thresholds file not found: {tuned_path}")

    # Filter out ignored probes
    if args.ignore_probes:
        ignored = set(args.ignore_probes)
        data["probes"] = [p for p in data["probes"] if p not in ignored]
        for key in ("sentence_scores", "sentence_extents", "results", "fp_thresholds"):
            if key in data and isinstance(data[key], dict):
                for p in ignored:
                    data[key].pop(p, None)
        # Also remove from thresholds (per eval mode)
        if "thresholds" in data:
            for mode in list(data["thresholds"].keys()):
                if isinstance(data["thresholds"][mode], dict):
                    for p in ignored:
                        data["thresholds"][mode].pop(p, None)
        print(f"  Ignored {len(ignored)} probes: {', '.join(sorted(ignored))}")

    # Filter behaviors
    if args.include_behaviors or args.exclude_behaviors:
        import fnmatch as _fnm
        inc_pats = args.include_behaviors.split(",") if args.include_behaviors else None
        exc_pats = args.exclude_behaviors.split(",") if args.exclude_behaviors else None

        def _keep_beh(name):
            if inc_pats and not any(_fnm.fnmatch(name, p) for p in inc_pats):
                return False
            if exc_pats and any(_fnm.fnmatch(name, p) for p in exc_pats):
                return False
            return True

        orig_behs = list(data["behaviors"])
        data["behaviors"] = [b for b in data["behaviors"] if _keep_beh(b)]
        kept = set(data["behaviors"])
        for key in ("tokens", "turn_maps"):
            if key in data:
                data[key] = {b: v for b, v in data[key].items() if b in kept}
        for key in ("sentence_scores", "results"):
            if key in data and isinstance(data[key], dict):
                for probe in list(data[key].keys()):
                    data[key][probe] = {b: v for b, v in data[key][probe].items() if b in kept}
        if "ground_truth" in data:
            data["ground_truth"] = {b: v for b, v in data["ground_truth"].items() if b in kept}
        if "n_turns" in data:
            data["n_turns"] = {b: v for b, v in data["n_turns"].items() if b in kept}
        removed = set(orig_behs) - kept
        if removed:
            print(f"  Filtered behaviors: kept {len(kept)}, removed {len(removed)}")

    print(f"  {len(data['probes'])} probes, {len(data['behaviors'])} behaviors")
    n_tokens = sum(
        len(v) for beh_data in data["tokens"].values() for v in beh_data.values()
    )
    print(f"  {n_tokens} total token arrays")
    n_sentences = sum(
        len(avgs)
        for beh_data in data["sentence_scores"].get(data["probes"][0], {}).values()
        for ri_data in beh_data.values()
        for avgs in ri_data.values()
    )
    print(f"  ~{n_sentences} sentences (per probe)")

    data_json = json.dumps(data, separators=(",", ":"))

    title = f"Misalignment Sentence Dashboard \u2014 {args.results_subdir}"
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            base_dir / "probe_eval" / "results" / args.results_subdir
            / "misalignment_sentence_dashboard.html"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Generated {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
