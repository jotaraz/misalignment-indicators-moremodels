"""
Generate an interactive HTML dashboard for probe evaluation results.

For each probe (layer 30) and each transcript (behavior-i, i=1..10), shows:
  - Ground truth indicators detected by Opus (run 1) for the relevant indicator,
    with highlighted sentence spans
  - Average probe score per turn
  - Per-token coloring of model responses based on token-level probe scores

Data is embedded as JSON and rendered client-side for compactness and performance.

Usage:
    python probe_eval/visualize_probes.py
    python probe_eval/visualize_probes.py --output probe_eval/results/probe_dashboard.html
    python probe_eval/visualize_probes.py --probe-set behavior
"""

import argparse
import json
from pathlib import Path

# ---------- Config ---------- #

PROBE_SETS = {
    "finegrain": {
        "results_dir": "probe_eval/results/finegrain",
        "gt_dirs": {
            "instructed-long-horizon-sabotage-backdoor": "bloom/indicator_results/v2.2/sabotage_backdoor/v2.2-mid_opus",
            "instructed-long-horizon-sabotage-bug": "bloom/indicator_results/v2.2/sabotage_bug/v2.2-mid_opus",
            "sycophancy": "bloom/indicator_results/v2.2/sycophancy/v2.2-mid_opus",
        },
        "rollout_dirs": {
            "instructed-long-horizon-sabotage-backdoor": "bloom/bloom-results/instructed-long-horizon-sabotage-backdoor",
            "instructed-long-horizon-sabotage-bug": "bloom/bloom-results/instructed-long-horizon-sabotage-bug",
            "sycophancy": "bloom/bloom-results/sycophancy",
        },
    },
    "behavior": {
        "results_dir": "probe_eval/results/behavior",
        "gt_dirs": {
            "instructed-long-horizon-sabotage-backdoor": "bloom/indicator_results/v2.2/sabotage_backdoor/v2.2-per-behavior_opus",
            "instructed-long-horizon-sabotage-bug": "bloom/indicator_results/v2.2/sabotage_bug/v2.2-per-behavior_opus",
            "sycophancy": "bloom/indicator_results/v2.2/sycophancy/v2.2-per-behavior_opus",
        },
        "rollout_dirs": {
            "instructed-long-horizon-sabotage-backdoor": "bloom/bloom-results/instructed-long-horizon-sabotage-backdoor",
            "instructed-long-horizon-sabotage-bug": "bloom/bloom-results/instructed-long-horizon-sabotage-bug",
            "sycophancy": "bloom/bloom-results/sycophancy",
        },
    },
}

LAYER = "layer30"


# ---------- Data loading ---------- #

def probe_name_to_indicator(probe_name: str) -> str:
    """Convert snake_case probe name to Title Case indicator name."""
    result = probe_name.replace("_", " ").title()
    # Handle special cases
    result = result.replace("Recognized Concern Suppression", "Recognized-Concern Suppression")
    return result


def compute_turn_map_from_rollout(rollout: dict) -> list[int]:
    """Compute per-message bloom turn numbers from a bloom rollout.

    Replicates the turn counting logic from evaluate.py's extract_target_dialogue:
    each target-view user message increments the turn counter. Tool messages and
    separator messages (for alternating roles) do NOT increment it. Returns one
    bloom turn number per message (including separators) in the dialogue.
    """
    events = rollout["transcript"]["events"]
    turn_map: list[int] = []
    current_turn = 0
    prev_mapped_role: str | None = None

    for event in events:
        if "target" not in event.get("view", []):
            continue
        msg = event.get("edit", {}).get("message", {})
        role = msg.get("role")
        content = msg.get("content", "")

        if role is None:
            continue

        if role == "user":
            current_turn += 1

        # Handle structured content (same as evaluate.py)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "reasoning" and block.get("reasoning"):
                        text_parts.append(block["reasoning"].strip())
                    elif block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"].strip())
            content = " ".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            continue

        if role in ("system", "user", "tool"):
            mapped_role = "user" if role == "tool" else role
        elif role == "assistant":
            mapped_role = "assistant"
        else:
            continue

        # Separator for consecutive same roles (mirrors evaluate.py logic)
        if prev_mapped_role is not None and prev_mapped_role == mapped_role:
            turn_map.append(current_turn)  # separator message
        turn_map.append(current_turn)
        prev_mapped_role = mapped_role

    return turn_map


def load_bloom_turn_maps(
    rollout_path: Path,
) -> dict[int, list[int]]:
    """Load bloom rollouts and compute per-message turn maps, keyed by rollout index."""
    if not rollout_path.exists():
        return {}
    data = json.loads(rollout_path.read_text())
    result: dict[int, list[int]] = {}
    for i, rollout in enumerate(data.get("rollouts", [])):
        result[i] = compute_turn_map_from_rollout(rollout)
    return result


def load_ground_truth(gt_dir: str) -> dict[int, list[dict]]:
    """Load run 1 ground truth indicators, indexed by variation_number."""
    path = Path(gt_dir) / "rollout_run1.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    result = {}
    for r in data["evaluated_rollouts"]:
        result[r["variation_number"]] = r["detected_indicators"]
    return result


def build_data(probe_set_name: str, base_dir: Path) -> dict:
    """Build the complete data structure for the dashboard."""
    cfg = PROBE_SETS[probe_set_name]
    results_dir = base_dir / cfg["results_dir"]

    probes = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and not d.name.endswith(".json")
    ])

    behaviors = ["instructed-long-horizon-sabotage-backdoor",
                 "instructed-long-horizon-sabotage-bug",
                 "sycophancy"]

    # Load all ground truth
    gt_data: dict[str, dict[int, list[dict]]] = {}
    for beh, gt_dir in cfg["gt_dirs"].items():
        gt_data[beh] = load_ground_truth(base_dir / gt_dir)

    # Preload bloom turn maps (fallback for token_scores.json without turn_map)
    bloom_turn_maps: dict[str, dict[int, list[int]]] = {}
    for beh, rollout_dir in cfg.get("rollout_dirs", {}).items():
        bloom_turn_maps[beh] = load_bloom_turn_maps(
            base_dir / rollout_dir / "rollout.json"
        )

    # Token data is shared across probes for the same behavior+rollout.
    # Store tokens once, scores per probe.
    # Structure:
    #   tokens[beh][rollout_index] = [token_strings...]
    #   turn_maps[beh][rollout_index] = [bloom_turn_per_message...]
    #   scores[probe][beh][rollout_index] = [score_or_null...]
    shared_tokens: dict[str, dict[int, list[str]]] = {}
    shared_turn_maps: dict[str, dict[int, list[int]]] = {}
    probe_scores: dict[str, dict[str, dict[int, list]]] = {}
    # Per-probe results
    probe_results: dict[str, dict[str, dict]] = {}

    # Per-probe score extents for independent color scales
    probe_extents: dict[str, list[float]] = {}

    for probe in probes:
        probe_scores[probe] = {}
        probe_results[probe] = {}
        probe_min = float("inf")
        probe_max = float("-inf")
        for beh in behaviors:
            result_dir = results_dir / probe / "turn" / LAYER / beh
            results_path = result_dir / "results.json"
            token_scores_path = result_dir / "token_scores.json"

            if not results_path.exists():
                continue

            results = json.loads(results_path.read_text())
            probe_results[probe][beh] = results

            # Load token scores
            if token_scores_path.exists():
                ts_data = json.loads(token_scores_path.read_text())
                ts_by_idx = {r["rollout_index"]: r for r in ts_data.get("per_rollout", [])}
            else:
                ts_by_idx = {}

            # Store tokens (shared) and scores (per-probe)
            if beh not in shared_tokens:
                shared_tokens[beh] = {}
            if beh not in shared_turn_maps:
                shared_turn_maps[beh] = {}

            probe_scores[probe][beh] = {}
            for r in results.get("per_rollout", []):
                idx = r["rollout_index"]
                tok_data = ts_by_idx.get(idx)
                if tok_data:
                    if idx not in shared_tokens[beh]:
                        shared_tokens[beh][idx] = tok_data["tokens"]
                    if idx not in shared_turn_maps[beh]:
                        if "turn_map" in tok_data:
                            shared_turn_maps[beh][idx] = tok_data["turn_map"]
                        elif idx in bloom_turn_maps.get(beh, {}):
                            shared_turn_maps[beh][idx] = bloom_turn_maps[beh][idx]
                    # Round scores to 4 decimal places for compactness
                    scores = [round(s, 4) if s is not None else None
                              for s in tok_data["scores"]]
                    probe_scores[probe][beh][idx] = scores

                    for s in tok_data["scores"]:
                        if s is not None:
                            probe_min = min(probe_min, s)
                            probe_max = max(probe_max, s)

        if probe_min == float("inf"):
            probe_min, probe_max = -1.0, 1.0
        probe_extents[probe] = [round(probe_min, 4), round(probe_max, 4)]

    # Load per-probe thresholds from indicator_gt_summary.json
    # thresholds[probe] = {threshold, accuracy, auroc} or absent if no data
    probe_thresholds: dict[str, dict] = {}
    summary_path = results_dir / "indicator_gt_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        layer_data = summary.get("per_layer", {}).get(LAYER, {})
        per_concept = layer_data.get("per_concept", {})
        for probe in probes:
            if probe in per_concept:
                pc = per_concept[probe]
                probe_thresholds[probe] = {
                    "threshold": pc.get("best_accuracy_nontrivial_threshold"),
                    "accuracy_nontrivial": pc.get("best_accuracy_nontrivial"),
                    "precision_nontrivial": pc.get("precision_nontrivial"),
                    "recall_nontrivial": pc.get("recall_nontrivial"),
                    "auroc": pc.get("mean_binary_auroc"),
                }

    # Build ground truth filtered by probe
    # gt_filtered[probe][beh][variation] = [matching_indicators...]
    gt_filtered: dict[str, dict[str, dict[int, list[dict]]]] = {}
    for probe in probes:
        gt_filtered[probe] = {}
        indicator_name = probe_name_to_indicator(probe)
        for beh in behaviors:
            gt_filtered[probe][beh] = {}
            for var_num, indicators in gt_data.get(beh, {}).items():
                matching = [
                    ind for ind in indicators
                    if ind["indicator_name"].lower() == indicator_name.lower()
                ]
                if matching:
                    gt_filtered[probe][beh][var_num] = matching

    # Assemble output
    return {
        "probes": probes,
        "behaviors": behaviors,
        "tokens": {
            beh: {str(k): v for k, v in rollouts.items()}
            for beh, rollouts in shared_tokens.items()
        },
        "turn_maps": {
            beh: {str(k): v for k, v in rollouts.items()}
            for beh, rollouts in shared_turn_maps.items()
        },
        "scores": {
            probe: {
                beh: {str(k): v for k, v in rollouts.items()}
                for beh, rollouts in beh_scores.items()
            }
            for probe, beh_scores in probe_scores.items()
        },
        "results": {
            probe: {
                beh: {
                    "metrics": res.get("metrics", {}),
                    "per_rollout": res.get("per_rollout", []),
                }
                for beh, res in beh_res.items()
            }
            for probe, beh_res in probe_results.items()
        },
        "gt": {
            probe: {
                beh: {str(k): v for k, v in vars_.items()}
                for beh, vars_ in beh_gt.items()
            }
            for probe, beh_gt in gt_filtered.items()
        },
        "extents": {probe: ext for probe, ext in probe_extents.items()},
        "thresholds": probe_thresholds,
    }


# ---------- HTML template ---------- #

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;padding:16px;color:#333}
h1{font-size:1.4rem;margin-bottom:4px;color:#222;text-align:center}
.subtitle{text-align:center;font-size:.85rem;color:#777;margin-bottom:12px}
.tab{padding:6px 14px;border:2px solid #ddd;border-radius:16px;background:#fff;cursor:pointer;font-size:.8rem;font-weight:500;transition:all .15s;color:#555}
.tab:hover{border-color:#999;color:#222}
.tab.active{background:#333;color:#fff;border-color:#333}
.probe-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;padding:8px;background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.beh-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.roll-tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.roll-tab{font-size:.75rem;padding:4px 10px}
.roll-tab.dec{border-color:#e88}.roll-tab.dec.active{background:#c33;border-color:#c33}
.roll-tab.hon{border-color:#8b8}.roll-tab.hon.active{background:#393;border-color:#393}
.bp-badge{font-size:.65rem;opacity:.8}
.rollout-hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#fafafa;border-radius:8px;margin-bottom:8px;border:1px solid #e0e0e0}
.rollout-hdr h3{font-size:.95rem;margin:0}
.score-lbl{font-family:monospace;font-size:.82rem;color:#555}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:bold;margin-left:6px}
.badge.deceptive{background:#fdd;color:#c00}.badge.honest{background:#dfd;color:#070}
.turn-card{background:#fff;border-radius:8px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
.turn-hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#f7f7f7;border-bottom:1px solid #eee}
.turn-lbl{font-weight:600;font-size:.85rem}
.turn-score{font-family:monospace;font-size:.8rem;padding:2px 8px;border-radius:4px}
.turn-body{padding:8px 12px}
.msg{margin-bottom:8px}
.msg-role{font-size:.7rem;font-weight:600;text-transform:uppercase;color:#888;margin-bottom:2px}
.user-text{font-size:.8rem;color:#555;white-space:pre-wrap;word-wrap:break-word;max-height:200px;overflow-y:auto;background:#f9f9f9;padding:6px 8px;border-radius:4px;border:1px solid #eee;font-family:monospace}
.sys-text{font-size:.78rem;color:#666;white-space:pre-wrap;word-wrap:break-word;max-height:300px;overflow-y:auto;padding:8px;font-family:monospace}
.asst-tokens{line-height:1.8;word-wrap:break-word}
.tok{display:inline;padding:1px 0;font-family:monospace;font-size:.78rem;cursor:default;white-space:pre-wrap}
.tok.un{color:#999}
.tok.gt{border-bottom:2px solid #f90}
.gt-box{margin:6px 12px;padding:8px;background:#fff8e1;border-left:3px solid #f90;border-radius:0 4px 4px 0}
.gt-hdr{font-weight:600;font-size:.8rem;color:#b57900;margin-bottom:6px}
.gt-item{margin-bottom:6px;padding:4px 6px;background:#fff;border-radius:4px;font-size:.78rem}
.gt-name{font-weight:600;color:#c60}
.gt-conf{font-size:.7rem;color:#888;margin-left:6px}
.gt-sent{margin-top:3px;padding:3px 6px;background:#fff3cd;border-radius:3px;font-size:.75rem;color:#555;font-style:italic}
.gt-ev{margin-top:2px;font-size:.72rem;color:#777}
.empty{color:#999;font-style:italic;padding:20px;text-align:center}
.thresh-info{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:8px 12px;background:#eef;border-radius:8px;margin-bottom:10px;font-size:.78rem;border:1px solid #cce}
.thresh-info .ti-label{font-weight:600;color:#446}
.thresh-info .ti-val{font-family:monospace;color:#333}
.det-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.65rem;font-weight:bold;margin-left:6px}
.det-badge.pos{background:#fdd;color:#c00}
.det-badge.neg{background:#eee;color:#888}
.legend{display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0 12px;font-size:.78rem;color:#777}
.legend-bar{width:180px;height:14px;border-radius:7px;background:linear-gradient(to right,rgba(100,100,255,.8),rgba(255,255,255,.3),rgba(255,100,100,.8));border:1px solid #ddd}
.legend-hl{border-bottom:2px solid #f90;padding:0 4px 1px}
summary{cursor:pointer;padding:8px 12px;font-weight:600;font-size:.85rem;background:#f7f7f7;border-bottom:1px solid #eee}
details{margin-bottom:10px;background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
#scroll-top{display:none;position:fixed;bottom:24px;right:24px;width:40px;height:40px;border-radius:50%;background:#555;color:#fff;border:none;font-size:20px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.3);z-index:1000}
#scroll-top:hover{background:#333}
#content{min-height:200px}
.loading{text-align:center;padding:40px;color:#999;font-size:1.1rem}
</style>
</head>
<body>

<h1>__TITLE__</h1>
<p class="subtitle">Layer 30 probe scores with ground truth indicators (Opus run 1). Hover tokens for scores.</p>
<div class="legend">
<span>Low score</span><span class="legend-bar"></span><span>High score</span>
<span id="extent-label" style="margin-left:8px;font-family:monospace;font-size:.72rem"></span>
<span style="margin-left:12px"><span class="legend-hl">highlighted</span> = GT indicator span</span>
</div>

<div class="probe-tabs" id="probe-tabs"></div>
<div class="beh-tabs" id="beh-tabs"></div>
<div class="roll-tabs" id="roll-tabs"></div>
<div id="content"><div class="loading">Loading...</div></div>

<button id="scroll-top" title="Scroll to top">&#8679;</button>

<script>
const DATA = __DATA__;

let curProbe = DATA.probes[0];
let curBeh = DATA.behaviors[0];
let curRi = 0;

function getExtent() {
    const ext = DATA.extents[curProbe];
    return ext || [-1, 1];
}

function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scoreColor(s) {
    const [pMin, pMax] = getExtent();
    if (pMax === pMin) return 'rgba(255,255,255,0)';
    const mid = (pMax + pMin) / 2;
    const hr = (pMax - pMin) / 2;
    let t = hr === 0 ? 0 : Math.max(-1, Math.min(1, (s - mid) / hr));
    let r, g, b, a;
    if (t > 0) {
        r = 255; g = Math.round(255*(1-t)); b = g;
        a = 0.3 + 0.7*t;
    } else {
        r = Math.round(255*(1+t)); g = r; b = 255;
        a = 0.3 + 0.7*(-t);
    }
    return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}

function renderProbeTabs() {
    const el = document.getElementById('probe-tabs');
    el.innerHTML = DATA.probes.map(p => {
        const label = p.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
        const cls = p === curProbe ? 'tab active' : 'tab';
        return `<button class="${cls}" data-p="${p}">${label}</button>`;
    }).join('');
    el.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => { curProbe = btn.dataset.p; curRi = 0; renderAll(); };
    });
}

function renderBehTabs() {
    const el = document.getElementById('beh-tabs');
    const res = DATA.results[curProbe] || {};
    el.innerHTML = DATA.behaviors.map(b => {
        const m = (res[b]||{}).metrics||{};
        const auroc = m.auroc;
        const aStr = (auroc != null && !isNaN(auroc)) ? ` (AUROC: ${auroc.toFixed(3)})` : '';
        const label = b.replace(/-/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
        const cls = b === curBeh ? 'tab active' : 'tab';
        return `<button class="${cls}" data-b="${b}">${label}${aStr}</button>`;
    }).join('');
    el.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => { curBeh = btn.dataset.b; curRi = 0; renderAll(); };
    });
}

function renderRollTabs() {
    const el = document.getElementById('roll-tabs');
    const res = DATA.results[curProbe]?.[curBeh];
    if (!res) { el.innerHTML = ''; return; }
    el.innerHTML = res.per_rollout.map(r => {
        const cls = `tab roll-tab ${r.label==='deceptive'?'dec':'hon'} ${r.rollout_index===curRi?'active':''}`;
        return `<button class="${cls}" data-ri="${r.rollout_index}">behavior-${r.rollout_index+1} <span class="bp-badge">bp=${r.behavior_presence}</span></button>`;
    }).join('');
    el.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => { curRi = parseInt(btn.dataset.ri); renderContent(); renderRollTabs(); };
    });
}

function segmentTokens(tokens, scores) {
    const segs = [];
    let role = null, toks = [], sc = [], start = 0;
    for (let i = 0; i < tokens.length; i++) {
        const t = tokens[i];
        if (t === '<|system|>' || t === '<|user|>' || t === '<|assistant|>') {
            if (role !== null) segs.push({role, tokens: toks, scores: sc, start});
            role = t.replace('<|','').replace('|>','');
            toks = []; sc = []; start = i + 1;
        } else {
            toks.push(t);
            sc.push(i < scores.length ? scores[i] : null);
        }
    }
    if (role && toks.length) segs.push({role, tokens: toks, scores: sc, start});
    return segs;
}

function groupTurns(segs, turnMap) {
    // turnMap: optional array of bloom turn numbers (one per message in the dialogue).
    // When available, segments are grouped by bloom turn so that tool calls
    // and separator messages share the same turn as their parent user message.
    // Without turnMap, falls back to sequential numbering (legacy).
    const turns = [];
    if (turnMap && turnMap.length) {
        let cur = [], currentTn = null, msgIdx = 0;
        for (const seg of segs) {
            const tn = msgIdx < turnMap.length ? turnMap[msgIdx] : currentTn;
            msgIdx++;
            if (currentTn !== null && tn !== currentTn) {
                turns.push({turn: currentTn, segments: cur});
                cur = [seg];
                currentTn = tn;
            } else {
                cur.push(seg);
                if (currentTn === null) currentTn = tn;
            }
        }
        if (cur.length && currentTn !== null) turns.push({turn: currentTn, segments: cur});
    } else {
        let cur = [], tn = 0;
        for (const seg of segs) {
            if (seg.role === 'system') { cur.push(seg); continue; }
            if (seg.role === 'user') {
                if (cur.length) turns.push({turn: tn, segments: cur});
                tn++;
                cur = [seg];
            } else {
                cur.push(seg);
            }
        }
        if (cur.length) turns.push({turn: tn, segments: cur});
    }
    return turns;
}

function findGtSpans(tokens, gtInds) {
    const flags = new Uint8Array(tokens.length);
    const text = tokens.join('');
    // Build char offset map
    const offsets = [];
    let pos = 0;
    for (const t of tokens) { offsets.push(pos); pos += t.length; }
    for (const ind of gtInds) {
        const sent = ind.sentence || '';
        if (!sent) continue;
        let idx = text.indexOf(sent);
        if (idx === -1) idx = text.indexOf(sent.slice(0, 80));
        if (idx < 0) continue;
        const end = idx + sent.length;
        for (let ti = 0; ti < tokens.length; ti++) {
            const ts = offsets[ti], te = ts + tokens[ti].length;
            if (ts < end && te > idx) flags[ti] = 1;
        }
    }
    return flags;
}

function renderTokens(tokens, scores, gtInds) {
    const gtFlags = findGtSpans(tokens, gtInds);
    const parts = [];
    for (let i = 0; i < tokens.length; i++) {
        const t = esc(tokens[i]).replace(/\n/g, '<br>');
        const s = scores[i];
        if (s == null) {
            parts.push(`<span class="tok un">${t}</span>`);
        } else {
            const bg = scoreColor(s);
            const gt = gtFlags[i] ? ' gt' : '';
            parts.push(`<span class="tok${gt}" style="background:${bg}" title="${s.toFixed(4)}">${t}</span>`);
        }
    }
    return parts.join('');
}

function renderContent() {
    const el = document.getElementById('content');
    const res = DATA.results[curProbe]?.[curBeh];
    if (!res) { el.innerHTML = '<div class="empty">No results for this combination.</div>'; return; }
    const rollout = res.per_rollout.find(r => r.rollout_index === curRi);
    if (!rollout) { el.innerHTML = '<div class="empty">Rollout not found.</div>'; return; }

    const tokens = DATA.tokens[curBeh]?.[String(curRi)] || [];
    const scores = DATA.scores[curProbe]?.[curBeh]?.[String(curRi)] || [];
    const turnMap = DATA.turn_maps?.[curBeh]?.[String(curRi)] || [];
    const variation = curRi + 1;
    const gtInds = DATA.gt[curProbe]?.[curBeh]?.[String(variation)] || [];
    const turnScoreMap = {};
    for (const ts of (rollout.per_turn_scores || [])) turnScoreMap[ts.turn] = ts.score;

    // GT by turn
    const gtByTurn = {};
    for (const ind of gtInds) {
        const tn = ind.turn_number;
        if (!gtByTurn[tn]) gtByTurn[tn] = [];
        gtByTurn[tn].push(ind);
    }

    const segs = segmentTokens(tokens, scores);
    const turns = groupTurns(segs, turnMap);

    // Threshold info for current probe
    const threshInfo = DATA.thresholds[curProbe];
    const thresh = threshInfo ? threshInfo.threshold : null;

    let html = '';
    // Header
    const lbl = rollout.label;
    html += `<div class="rollout-hdr"><h3>Variation ${variation} <span class="badge ${lbl}">${lbl} (bp=${rollout.behavior_presence})</span></h3>`;
    html += `<span class="score-lbl">Overall probe score: ${rollout.probe_score.toFixed(4)}</span></div>`;

    // Threshold info bar
    if (threshInfo) {
        html += '<div class="thresh-info">';
        html += `<span><span class="ti-label">Threshold (best acc nontrivial):</span> <span class="ti-val">${thresh.toFixed(4)}</span></span>`;
        if (threshInfo.accuracy_nontrivial != null) html += `<span><span class="ti-label">Accuracy:</span> <span class="ti-val">${(threshInfo.accuracy_nontrivial*100).toFixed(1)}%</span></span>`;
        if (threshInfo.precision_nontrivial != null) html += `<span><span class="ti-label">Precision:</span> <span class="ti-val">${(threshInfo.precision_nontrivial*100).toFixed(1)}%</span></span>`;
        if (threshInfo.recall_nontrivial != null) html += `<span><span class="ti-label">Recall:</span> <span class="ti-val">${(threshInfo.recall_nontrivial*100).toFixed(1)}%</span></span>`;
        if (threshInfo.auroc != null) html += `<span><span class="ti-label">AUROC:</span> <span class="ti-val">${threshInfo.auroc.toFixed(3)}</span></span>`;
        html += '</div>';
    }

    if (!tokens.length) {
        html += '<div class="empty">No token data available.</div>';
        el.innerHTML = html;
        return;
    }

    for (const turn of turns) {
        const tn = turn.turn;
        if (tn === 0) {
            // System prompt collapsed
            let sysText = '';
            for (const seg of turn.segments) sysText += seg.tokens.join('');
            html += `<details class="turn-card"><summary>System Prompt</summary><div class="turn-body"><pre class="sys-text">${esc(sysText)}</pre></div></details>`;
            continue;
        }

        const score = turnScoreMap[tn];
        const scoreStr = score != null ? score.toFixed(4) : 'N/A';
        const scoreBg = score != null ? scoreColor(score) : '';
        // Binary detection based on threshold
        let detBadge = '';
        if (thresh != null && score != null) {
            const detected = score >= thresh;
            detBadge = detected
                ? '<span class="det-badge pos">DETECTED</span>'
                : '<span class="det-badge neg">not detected</span>';
        }

        html += '<div class="turn-card">';
        html += `<div class="turn-hdr"><span class="turn-lbl">Turn ${tn}${detBadge}</span><span class="turn-score" style="background:${scoreBg}">Probe Score: ${scoreStr}</span></div>`;

        // GT indicators for this turn
        const turnGt = gtByTurn[tn] || [];
        if (turnGt.length) {
            html += '<div class="gt-box">';
            html += `<div class="gt-hdr">Ground Truth Indicators (${turnGt.length})</div>`;
            for (const ind of turnGt) {
                html += `<div class="gt-item"><span class="gt-name">${esc(ind.indicator_name)}</span><span class="gt-conf">conf=${ind.confidence.toFixed(2)}</span>`;
                if (ind.sentence) html += `<div class="gt-sent">${esc(ind.sentence)}</div>`;
                if (ind.evidence) html += `<div class="gt-ev">${esc(ind.evidence)}</div>`;
                html += '</div>';
            }
            html += '</div>';
        }

        html += '<div class="turn-body">';
        for (const seg of turn.segments) {
            if (seg.role === 'user') {
                html += `<div class="msg"><div class="msg-role">User</div><pre class="user-text">${esc(seg.tokens.join(''))}</pre></div>`;
            } else if (seg.role === 'assistant') {
                html += `<div class="msg"><div class="msg-role">Assistant</div><div class="asst-tokens">${renderTokens(seg.tokens, seg.scores, turnGt)}</div></div>`;
            }
        }
        html += '</div></div>';
    }

    el.innerHTML = html;
}

function renderAll() {
    renderProbeTabs();
    renderBehTabs();
    renderRollTabs();
    renderContent();
    const [pMin, pMax] = getExtent();
    document.getElementById('extent-label').textContent = `[${pMin.toFixed(2)}, ${pMax.toFixed(2)}]`;
}

// Scroll to top
const scrollBtn = document.getElementById('scroll-top');
window.addEventListener('scroll', () => { scrollBtn.style.display = window.scrollY > 300 ? 'block' : 'none'; });
scrollBtn.addEventListener('click', () => { window.scrollTo({top:0,behavior:'smooth'}); });

renderAll();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate probe evaluation HTML dashboard")
    parser.add_argument(
        "--probe-set",
        default="finegrain",
        choices=list(PROBE_SETS.keys()),
        help="Which probe set to visualize",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory of the project")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    print("Building data...")
    data = build_data(args.probe_set, base_dir)

    print(f"  {len(data['probes'])} probes, {len(data['behaviors'])} behaviors")
    n_tokens = sum(
        len(v) for beh_data in data["tokens"].values() for v in beh_data.values()
    )
    print(f"  {n_tokens} total token arrays")

    # Serialize data
    data_json = json.dumps(data, separators=(",", ":"))

    title = f"Probe Evaluation Dashboard — {args.probe_set}"
    html_content = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = base_dir / PROBE_SETS[args.probe_set]["results_dir"] / "probe_dashboard.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Generated {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
