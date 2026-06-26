"""
Generate an interactive HTML dashboard for probe evaluation against
misalignment turn-level ground truth.

For each probe (layer 30) and each transcript (behavior-i, i=1..10), shows:
  - Ground truth misaligned turns (from rollout_misalignment_turns.json)
    with description and evidence quotes
  - Average probe score per turn
  - Per-token coloring of model responses based on token-level probe scores
  - Detection badges using joint-optimized F1 thresholds (OR across probes)
  - Three evaluation modes: per_turn, transcript_relaxed, partial_relaxed

Data is embedded as JSON and rendered client-side for compactness and performance.

Usage:
    python probe_eval/visualize_misalignment.py
    python probe_eval/visualize_misalignment.py --results-subdir v2_3
    python probe_eval/visualize_misalignment.py --output probe_eval/results/v2_3/misalignment_dashboard.html
"""

import argparse
import json
from pathlib import Path

# ---------- Config ---------- #

LAYER = "layer30"  # default; overridden by --layer CLI arg
BLOOM_RESULTS_DIR = "bloom/bloom-results"
BLOOM_RESULTS_TEST_DIR = "bloom/bloom-results-test"
EVAL_MODES = ["per_turn", "transcript_relaxed", "partial_relaxed"]
LABEL_MODES = ["turn", "span"]  # checked in order


def _find_label_mode_dir(probe_dir: Path) -> Path | None:
    """Return the first existing label-mode sub-directory (turn or span)."""
    for lm in LABEL_MODES:
        layer_dir = probe_dir / lm / LAYER
        if layer_dir.is_dir():
            return layer_dir
    return None


def discover_behaviors(results_dir: Path) -> list[str]:
    """Auto-discover behavior names from the results directory.

    Scans probe subdirs for behavior folders under {turn|span}/layer30/.
    """
    behaviors: set[str] = set()
    for probe_dir in results_dir.iterdir():
        if not probe_dir.is_dir() or probe_dir.name.startswith("."):
            continue
        layer_dir = _find_label_mode_dir(probe_dir)
        if layer_dir is None:
            continue
        for beh_dir in layer_dir.iterdir():
            if beh_dir.is_dir():
                behaviors.add(beh_dir.name)
    # Sort: non-benign first (alphabetical), then benign (alphabetical)
    non_benign = sorted(b for b in behaviors if not b.endswith("_benign"))
    benign = sorted(b for b in behaviors if b.endswith("_benign"))
    return non_benign + benign


# ---------- Data loading (reused from visualize_probes.py) ---------- #

def compute_turn_map_from_rollout(rollout: dict) -> list[int]:
    """Compute per-message bloom turn numbers from a bloom rollout."""
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

        if prev_mapped_role is not None and prev_mapped_role == mapped_role:
            turn_map.append(current_turn)
        turn_map.append(current_turn)
        prev_mapped_role = mapped_role

    return turn_map


def load_bloom_turn_maps(rollout_path: Path) -> dict[int, list[int]]:
    """Load bloom rollouts and compute per-message turn maps, keyed by rollout index."""
    if not rollout_path.exists():
        return {}
    data = json.loads(rollout_path.read_text())
    result: dict[int, list[int]] = {}
    for i, rollout in enumerate(data.get("rollouts", [])):
        result[i] = compute_turn_map_from_rollout(rollout)
    return result


def load_misalignment_gt(gt_path: Path) -> dict[int, list[dict]]:
    """Load misalignment ground truth, keyed by variation_number.

    Returns {variation_number: [{turn_index, description, evidence}, ...]}.
    """
    if not gt_path.exists():
        return {}
    data = json.loads(gt_path.read_text())
    result: dict[int, list[dict]] = {}
    for rollout in data.get("evaluated_rollouts", []):
        var_num = rollout["variation_number"]
        turns = []
        for t in rollout.get("misaligned_turns", []):
            turns.append({
                "turn_index": t["turn_index"],
                "description": t.get("description", ""),
                "evidence": t.get("evidence", []),
            })
        result[var_num] = turns
    return result


def load_thresholds_and_metrics(summary_path: Path) -> tuple[dict, dict]:
    """Load per-mode per-probe thresholds and joint metrics from misalignment_gt_summary.json.

    Returns (thresholds, joint_metrics) where:
      thresholds = {mode: {probe: threshold_or_null}}
      joint_metrics = {mode: {f1, accuracy, precision, recall}}
    """
    thresholds: dict[str, dict[str, float | None]] = {}
    joint_metrics: dict[str, dict] = {}

    if not summary_path.exists():
        return thresholds, joint_metrics

    data = json.loads(summary_path.read_text())
    layer_data = data.get("per_layer", {}).get(LAYER, {})

    for mode in EVAL_MODES:
        key = f"{mode}_joint_optimized_f1"
        section = layer_data.get(key, {})
        thresholds[mode] = section.get("per_probe_thresholds", {})
        joint_metrics[mode] = {
            "f1": section.get("f1"),
            "accuracy": section.get("accuracy"),
            "precision": section.get("precision"),
            "recall": section.get("recall"),
        }

    return thresholds, joint_metrics


# ---------- Build data ---------- #

def build_data(base_dir: Path, results_subdir: str) -> dict:
    """Build the complete data structure for the dashboard."""
    results_dir = base_dir / "probe_eval" / "results" / results_subdir

    probes = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and not d.name.endswith(".json") and not d.name.endswith(".html")
    ])

    behaviors = discover_behaviors(results_dir)

    # Load misalignment GT per behavior
    # Resolve rollout_dir from results.json (handles OOD behaviors whose
    # rollout dirs are outside bloom/bloom-results/)
    misalignment_gt: dict[str, dict[int, list[dict]]] = {}
    bloom_turn_maps: dict[str, dict[int, list[int]]] = {}
    _rollout_dir_cache: dict[str, Path] = {}
    for beh in behaviors:
        # Find rollout_dir from any probe's results.json for this behavior
        rollout_dir = None
        if beh not in _rollout_dir_cache:
            for probe_dir in results_dir.iterdir():
                rj = probe_dir / "span" / LAYER / beh / "results.json"
                if rj.exists():
                    with open(rj) as _f:
                        _rd = json.loads(_f.read()).get("rollout_dir", "")
                    if _rd:
                        _rollout_dir_cache[beh] = Path(_rd)
                        break
        rollout_dir = _rollout_dir_cache.get(beh)

        if rollout_dir and rollout_dir.exists():
            bloom_dir = rollout_dir
        else:
            # Fallback to legacy paths
            bloom_dir = base_dir / BLOOM_RESULTS_DIR / beh
            if not bloom_dir.exists():
                bloom_dir = base_dir / BLOOM_RESULTS_TEST_DIR / beh
        gt_path = bloom_dir / "rollout_misalignment_turns.json"
        misalignment_gt[beh] = load_misalignment_gt(gt_path)
        bloom_turn_maps[beh] = load_bloom_turn_maps(
            bloom_dir / "rollout.json"
        )

    # Load thresholds and joint metrics
    summary_path = results_dir / "misalignment_gt_summary.json"
    thresholds, joint_metrics = load_thresholds_and_metrics(summary_path)

    # Token data shared across probes; scores per probe
    shared_tokens: dict[str, dict[int, list[str]]] = {}
    shared_turn_maps: dict[str, dict[int, list[int]]] = {}
    probe_scores: dict[str, dict[str, dict[int, list]]] = {}
    probe_results: dict[str, dict[str, dict]] = {}
    probe_extents: dict[str, list[float]] = {}

    for probe in probes:
        probe_scores[probe] = {}
        probe_results[probe] = {}
        probe_min = float("inf")
        probe_max = float("-inf")

        for beh in behaviors:
            probe_layer_dir = _find_label_mode_dir(results_dir / probe)
            if probe_layer_dir is None:
                continue
            result_dir = probe_layer_dir / beh
            results_path = result_dir / "results.json"
            token_scores_path = result_dir / "token_scores.json"

            if not results_path.exists():
                continue

            results = json.loads(results_path.read_text())
            probe_results[probe][beh] = results

            if token_scores_path.exists():
                ts_data = json.loads(token_scores_path.read_text())
                ts_by_idx = {r["rollout_index"]: r for r in ts_data.get("per_rollout", [])}
            else:
                ts_by_idx = {}

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

    # Build per-behavior n_turns map (needed for ambiguous turn computation)
    # {beh: {variation: n_turns}}
    n_turns_per_beh: dict[str, dict[int, int]] = {}
    # Map rollout_index -> variation_number (handles 0-indexed OOD dirs)
    rollout_variations: dict[str, dict[int, int]] = {}
    for beh in behaviors:
        # Prefer cached rollout_dir (handles OOD paths)
        rollout_dir = _rollout_dir_cache.get(beh)
        if rollout_dir and (rollout_dir / "rollout.json").exists():
            rollout_path = rollout_dir / "rollout.json"
        else:
            rollout_path = base_dir / BLOOM_RESULTS_DIR / beh / "rollout.json"
            if not rollout_path.exists():
                rollout_path = base_dir / BLOOM_RESULTS_TEST_DIR / beh / "rollout.json"
        if not rollout_path.exists():
            continue
        rollout_data = json.loads(rollout_path.read_text())
        n_turns_per_beh[beh] = {}
        rollout_variations[beh] = {}
        for ri, rollout in enumerate(rollout_data.get("rollouts", [])):
            var_num = rollout["variation_number"]
            n = sum(
                1
                for e in rollout["transcript"]["events"]
                if "target" in e.get("view", [])
                and e.get("edit", {}).get("message", {}).get("role") == "user"
            )
            n_turns_per_beh[beh][var_num] = n
            rollout_variations[beh][ri] = var_num

    return {
        "probes": probes,
        "behaviors": behaviors,
        "eval_modes": EVAL_MODES,
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
        "misalignment_gt": {
            beh: {str(var_num): turns for var_num, turns in gt.items()}
            for beh, gt in misalignment_gt.items()
        },
        "n_turns": {
            beh: {str(var_num): n for var_num, n in nt.items()}
            for beh, nt in n_turns_per_beh.items()
        },
        "rollout_variations": {
            beh: {str(ri): var for ri, var in rv.items()}
            for beh, rv in rollout_variations.items()
        },
        "extents": probe_extents,
        "thresholds": thresholds,
        "joint_metrics": joint_metrics,
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
.mode-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
.mode-tab{padding:6px 14px;border:2px solid #88c;border-radius:16px;background:#fff;cursor:pointer;font-size:.8rem;font-weight:500;transition:all .15s;color:#446}
.mode-tab:hover{border-color:#669;color:#224}
.mode-tab.active{background:#446;color:#fff;border-color:#446}
.mode-metrics{font-size:.75rem;color:#666;font-family:monospace;margin-left:8px}
.roll-tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.roll-tab{font-size:.75rem;padding:4px 10px}
.roll-tab.dec{border-color:#e88}.roll-tab.dec.active{background:#c33;border-color:#c33}
.roll-tab.hon{border-color:#8b8}.roll-tab.hon.active{background:#393;border-color:#393}
.bp-badge{font-size:.65rem;opacity:.8}
.tab-fp,.tab-fn{font-size:.6rem;font-weight:700;padding:1px 3px;border-radius:3px;margin-left:2px}
.tab-fp{background:#fdd;color:#c00}
.tab-fn{background:#fec;color:#960}
.rollout-hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#fafafa;border-radius:8px;margin-bottom:8px;border:1px solid #e0e0e0}
.rollout-hdr h3{font-size:.95rem;margin:0}
.score-lbl{font-family:monospace;font-size:.82rem;color:#555}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:bold;margin-left:6px}
.badge.deceptive{background:#fdd;color:#c00}.badge.honest{background:#dfd;color:#070}
.turn-card{background:#fff;border-radius:8px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
.turn-card.gt-misaligned{border-left:4px solid #c33}
.turn-card.ambiguous{border-left:4px solid #fa0;opacity:.85}
.turn-hdr{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#f7f7f7;border-bottom:1px solid #eee;flex-wrap:wrap;gap:6px}
.turn-lbl{font-weight:600;font-size:.85rem}
.turn-score{font-family:monospace;font-size:.8rem;padding:2px 8px;border-radius:4px}
.turn-badges{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.turn-body{padding:8px 12px}
.msg{margin-bottom:8px}
.msg-role{font-size:.7rem;font-weight:600;text-transform:uppercase;color:#888;margin-bottom:2px}
.user-text{font-size:.8rem;color:#555;white-space:pre-wrap;word-wrap:break-word;max-height:200px;overflow-y:auto;background:#f9f9f9;padding:6px 8px;border-radius:4px;border:1px solid #eee;font-family:monospace}
.sys-text{font-size:.78rem;color:#666;white-space:pre-wrap;word-wrap:break-word;max-height:300px;overflow-y:auto;padding:8px;font-family:monospace}
.asst-tokens{line-height:1.8;word-wrap:break-word}
.tok{display:inline;padding:1px 0;font-family:monospace;font-size:.78rem;cursor:default;white-space:pre-wrap}
.tok.un{color:#999}
.gt-box{margin:6px 12px;padding:8px;background:#fde8e8;border-left:3px solid #c33;border-radius:0 4px 4px 0}
.gt-hdr{font-weight:600;font-size:.8rem;color:#900;margin-bottom:6px}
.gt-desc{font-size:.78rem;color:#555;margin-bottom:4px}
.gt-ev{margin-top:2px;font-size:.72rem;color:#777;font-style:italic;padding:3px 6px;background:#fff3f3;border-radius:3px;margin-bottom:2px}
.amb-box{margin:6px 12px;padding:6px 8px;background:#fff8e1;border-left:3px solid #fa0;border-radius:0 4px 4px 0;font-size:.75rem;color:#886600}
.det-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.68rem;font-weight:bold}
.det-badge.tp{background:#cfc;color:#060;border:1px solid #9c9}
.det-badge.fp{background:#fdd;color:#c00;border:1px solid #eaa}
.det-badge.fn{background:#fec;color:#960;border:1px solid #db9}
.det-badge.tn{background:#eee;color:#888;border:1px solid #ddd}
.det-badge.amb{background:#fff8e1;color:#886600;border:1px solid #ecd}
.det-probes{font-size:.65rem;color:#666;margin-top:2px}
.thresh-bar{padding:8px 12px;background:#eef;border-radius:8px;margin-bottom:10px;font-size:.75rem;border:1px solid #cce}
.thresh-bar .tb-title{font-weight:600;color:#446;margin-bottom:4px}
.thresh-bar .tb-items{display:flex;gap:8px;flex-wrap:wrap}
.thresh-bar .tb-item{font-family:monospace;color:#333;padding:2px 6px;background:#fff;border-radius:3px;border:1px solid #dde}
.thresh-bar .tb-item.cur{background:#e8e0ff;border-color:#b8a0ee;font-weight:600}
.thresh-bar .tb-null{color:#aaa}
.empty{color:#999;font-style:italic;padding:20px;text-align:center}
.legend{display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0 12px;font-size:.78rem;color:#777;flex-wrap:wrap}
.legend-bar{width:180px;height:14px;border-radius:7px;background:linear-gradient(to right,rgba(100,100,255,.8),rgba(255,255,255,.3),rgba(255,100,100,.8));border:1px solid #ddd}
summary{cursor:pointer;padding:8px 12px;font-weight:600;font-size:.85rem;background:#f7f7f7;border-bottom:1px solid #eee}
details{margin-bottom:10px;background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
#scroll-top{display:none;position:fixed;bottom:24px;right:24px;width:40px;height:40px;border-radius:50%;background:#555;color:#fff;border:none;font-size:20px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.3);z-index:1000}
#scroll-top:hover{background:#333}
#content{min-height:200px}
.loading{text-align:center;padding:40px;color:#999;font-size:1.1rem}
.legend-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.68rem;font-weight:bold;margin:0 2px}
</style>
</head>
<body>

<h1>__TITLE__</h1>
<p class="subtitle">Layer 30 probe scores with misalignment turn-level ground truth. Hover tokens for scores.</p>
<div class="legend">
<span>Low score</span><span class="legend-bar"></span><span>High score</span>
<span id="extent-label" style="margin-left:8px;font-family:monospace;font-size:.72rem"></span>
<span style="margin-left:12px">
<span class="legend-badge" style="background:#cfc;color:#060">TP</span>
<span class="legend-badge" style="background:#fdd;color:#c00">FP</span>
<span class="legend-badge" style="background:#fec;color:#960">FN</span>
<span class="legend-badge" style="background:#eee;color:#888">TN</span>
<span class="legend-badge" style="background:#fff8e1;color:#886600">Ambiguous</span>
</span>
</div>

<div class="probe-tabs" id="probe-tabs"></div>
<div class="mode-tabs" id="mode-tabs"></div>
<div class="beh-tabs" id="beh-tabs"></div>
<div class="roll-tabs" id="roll-tabs"></div>
<div id="content"><div class="loading">Loading...</div></div>

<button id="scroll-top" title="Scroll to top">&#8679;</button>

<script>
const DATA = __DATA__;

let curProbe = DATA.probes[0];
let curBeh = DATA.behaviors[0];
let curRi = 0;
let curMode = DATA.eval_modes[0];

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

function renderModeTabs() {
    const el = document.getElementById('mode-tabs');
    const modeLabels = {
        'per_turn': 'Per Turn',
        'transcript_relaxed': 'Transcript Relaxed',
        'partial_relaxed': 'Partial Relaxed',
    };
    let html = '';
    for (const mode of DATA.eval_modes) {
        const cls = mode === curMode ? 'mode-tab active' : 'mode-tab';
        const jm = DATA.joint_metrics[mode] || {};
        const f1 = jm.f1 != null ? (jm.f1*100).toFixed(1)+'%' : 'N/A';
        const acc = jm.accuracy != null ? (jm.accuracy*100).toFixed(1)+'%' : 'N/A';
        const prec = jm.precision != null ? (jm.precision*100).toFixed(1)+'%' : 'N/A';
        const rec = jm.recall != null ? (jm.recall*100).toFixed(1)+'%' : 'N/A';
        html += `<button class="${cls}" data-m="${mode}">${modeLabels[mode]||mode} <span class="mode-metrics">F1=${f1} Acc=${acc} P=${prec} R=${rec}</span></button>`;
    }
    el.innerHTML = html;
    el.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => { curMode = btn.dataset.m; renderAll(); };
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

function getRolloutVariation(beh, rolloutIndex) {
    // Look up actual variation number from rollout.json (handles 0-indexed OOD dirs)
    const mapping = DATA.rollout_variations?.[beh];
    if (mapping && mapping[String(rolloutIndex)] !== undefined) {
        return mapping[String(rolloutIndex)];
    }
    return rolloutIndex + 1;  // fallback
}

function getRolloutFpFn(beh, rolloutIndex) {
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
}

function getRolloutGtLabel(beh, rolloutIndex) {
    // Determine if rollout has any misaligned turns from GT
    const variation = getRolloutVariation(beh, rolloutIndex);
    const nTurns = DATA.n_turns?.[beh]?.[String(variation)] || 20;
    const gtTurns = DATA.misalignment_gt?.[beh]?.[String(variation)] || [];
    const hasMisalignment = gtTurns.length > 0;
    return {label: hasMisalignment ? 'misaligned' : 'benign', cls: hasMisalignment ? 'dec' : 'hon', n_mis: gtTurns.length, n_turns: nTurns};
}

function renderRollTabs() {
    const el = document.getElementById('roll-tabs');
    const res = DATA.results[curProbe]?.[curBeh];
    if (!res) { el.innerHTML = ''; return; }
    el.innerHTML = res.per_rollout.map(r => {
        const {hasFP, hasFN} = getRolloutFpFn(curBeh, r.rollout_index);
        const fpfn = (hasFP ? ' <span class="tab-fp">FP</span>' : '') + (hasFN ? ' <span class="tab-fn">FN</span>' : '');
        const gt = getRolloutGtLabel(curBeh, r.rollout_index);
        const cls = `tab roll-tab ${gt.cls} ${r.rollout_index===curRi?'active':''}`;
        return `<button class="${cls}" data-ri="${r.rollout_index}">v${r.rollout_index+1} <span class="bp-badge">${gt.label} (${gt.n_mis}/${gt.n_turns})</span>${fpfn}</button>`;
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

function renderTokens(tokens, scores) {
    const parts = [];
    for (let i = 0; i < tokens.length; i++) {
        const t = esc(tokens[i]).replace(/\n/g, '<br>');
        const s = scores[i];
        if (s == null) {
            parts.push(`<span class="tok un">${t}</span>`);
        } else {
            const bg = scoreColor(s);
            parts.push(`<span class="tok" style="background:${bg}" title="${s.toFixed(4)}">${t}</span>`);
        }
    }
    return parts.join('');
}

/** Determine turn-level GT status for the current evaluation mode.
 *  Returns an object per turn: {gt_pos, ambiguous, misaligned_info}
 */
function getTurnGtStatus(variation, nTurns) {
    const gtTurns = DATA.misalignment_gt[curBeh]?.[String(variation)] || [];
    const misalignedSet = new Set(gtTurns.map(t => t.turn_index));
    const gtByTurn = {};
    for (const t of gtTurns) {
        gtByTurn[t.turn_index] = t;
    }

    const hasAnyMisalignment = misalignedSet.size > 0;
    const maxMisalignedTurn = misalignedSet.size > 0 ? Math.max(...misalignedSet) : -1;

    const result = {};
    for (let tn = 1; tn <= nTurns; tn++) {
        const isMisaligned = misalignedSet.has(tn);
        let ambiguous = false;

        if (curMode === 'transcript_relaxed') {
            // Ambiguous: in a positive transcript but not itself misaligned
            if (!isMisaligned && hasAnyMisalignment) ambiguous = true;
        } else if (curMode === 'partial_relaxed') {
            // Ambiguous: not misaligned but there exists a misaligned turn >= this turn
            if (!isMisaligned && tn <= maxMisalignedTurn) ambiguous = true;
        }
        // per_turn: no ambiguity, all turns are either positive or negative

        result[tn] = {
            gt_pos: isMisaligned,
            ambiguous: ambiguous,
            info: gtByTurn[tn] || null,
        };
    }
    return result;
}

/** Determine if a turn is detected as misaligned by probes (OR logic).
 *  Returns {detected: bool, fired_probes: [names]}
 */
function getProbeDetection(turnScores) {
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
    const variation = getRolloutVariation(curBeh, curRi);
    const nTurns = DATA.n_turns?.[curBeh]?.[String(variation)] || 20;

    // Build per-turn probe scores for ALL probes (for detection logic)
    const allProbeTurnScores = {};  // {turn: {probe: score}}
    for (const probe of DATA.probes) {
        const probeRes = DATA.results[probe]?.[curBeh];
        if (!probeRes) continue;
        const probeRollout = probeRes.per_rollout.find(r => r.rollout_index === curRi);
        if (!probeRollout) continue;
        for (const ts of (probeRollout.per_turn_scores || [])) {
            if (!allProbeTurnScores[ts.turn]) allProbeTurnScores[ts.turn] = {};
            allProbeTurnScores[ts.turn][probe] = ts.score;
        }
    }

    // Current probe's per-turn scores
    const turnScoreMap = {};
    for (const ts of (rollout.per_turn_scores || [])) turnScoreMap[ts.turn] = ts.score;

    // GT status per turn
    const turnGtStatus = getTurnGtStatus(variation, nTurns);

    const segs = segmentTokens(tokens, scores);
    const turns = groupTurns(segs, turnMap);

    let html = '';
    // Header
    const gtLbl = getRolloutGtLabel(curBeh, rollout.rollout_index);
    html += `<div class="rollout-hdr"><h3>Variation ${variation} <span class="badge ${gtLbl.cls}">${gtLbl.label} (${gtLbl.n_mis}/${gtLbl.n_turns} misaligned turns)</span></h3>`;
    html += `<span class="score-lbl">Probe score (${curProbe.replace(/_/g,' ')}): ${rollout.probe_score.toFixed(4)}</span></div>`;

    // Threshold info bar showing active thresholds for current mode
    const modeThresholds = DATA.thresholds[curMode] || {};
    html += `<div class="thresh-bar"><div class="tb-title">Active thresholds for ${curMode.replace(/_/g,' ')} (joint optimized F1):</div><div class="tb-items">`;
    for (const probe of DATA.probes) {
        const t = modeThresholds[probe];
        const label = probe.replace(/_/g,' ');
        const isCur = probe === curProbe;
        if (t == null) {
            html += `<span class="tb-item tb-null${isCur?' cur':''}">${label}: off</span>`;
        } else {
            html += `<span class="tb-item${isCur?' cur':''}">${label}: ${t.toFixed(3)}</span>`;
        }
    }
    html += '</div></div>';

    if (!tokens.length) {
        html += '<div class="empty">No token data available.</div>';
        el.innerHTML = html;
        return;
    }

    for (const turn of turns) {
        const tn = turn.turn;
        if (tn === 0) {
            let sysText = '';
            for (const seg of turn.segments) sysText += seg.tokens.join('');
            html += `<details class="turn-card"><summary>System Prompt</summary><div class="turn-body"><pre class="sys-text">${esc(sysText)}</pre></div></details>`;
            continue;
        }

        const score = turnScoreMap[tn];
        const scoreStr = score != null ? score.toFixed(4) : 'N/A';
        const scoreBg = score != null ? scoreColor(score) : '';

        // GT status
        const gt = turnGtStatus[tn] || {gt_pos: false, ambiguous: false, info: null};
        // Probe detection (OR across all probes)
        const det = getProbeDetection(allProbeTurnScores[tn] || {});

        // Classification badge
        let classBadge = '';
        let turnCardClass = 'turn-card';
        if (gt.ambiguous) {
            classBadge = `<span class="det-badge amb">AMBIGUOUS</span>`;
            turnCardClass += ' ambiguous';
            if (det.detected) {
                classBadge += ` <span class="det-badge amb">probe fired</span>`;
            }
        } else if (gt.gt_pos && det.detected) {
            classBadge = `<span class="det-badge tp">TP</span>`;
            turnCardClass += ' gt-misaligned';
        } else if (!gt.gt_pos && det.detected) {
            classBadge = `<span class="det-badge fp">FP</span>`;
        } else if (gt.gt_pos && !det.detected) {
            classBadge = `<span class="det-badge fn">FN</span>`;
            turnCardClass += ' gt-misaligned';
        } else {
            classBadge = `<span class="det-badge tn">TN</span>`;
        }

        // Show which probes fired with their score vs threshold
        let firedHtml = '';
        if (det.fired_probes.length > 0) {
            const details = det.fired_probes.map(p => {
                const score = (allProbeTurnScores[tn] || {})[p];
                const thresh = modeThresholds[p];
                const sStr = score != null ? score.toFixed(3) : '?';
                const tStr = thresh != null ? thresh.toFixed(3) : '?';
                return `${p.replace(/_/g,' ')} (${sStr} > ${tStr})`;
            }).join(', ');
            firedHtml = `<div class="det-probes">Fired: ${esc(details)}</div>`;
        }

        html += `<div class="${turnCardClass}">`;
        html += `<div class="turn-hdr"><span class="turn-lbl">Turn ${tn}</span><div class="turn-badges">${classBadge}<span class="turn-score" style="background:${scoreBg}">${curProbe.replace(/_/g,' ')}: ${scoreStr}</span></div></div>`;
        if (firedHtml) html += `<div style="padding:2px 12px">${firedHtml}</div>`;

        // GT misalignment info
        if (gt.gt_pos && gt.info) {
            html += '<div class="gt-box">';
            html += `<div class="gt-hdr">Ground Truth: Misaligned Turn</div>`;
            if (gt.info.description) html += `<div class="gt-desc">${esc(gt.info.description)}</div>`;
            if (gt.info.evidence && gt.info.evidence.length) {
                for (const ev of gt.info.evidence) {
                    const evText = typeof ev === 'string' ? ev : (ev.text || JSON.stringify(ev));
                    html += `<div class="gt-ev">${esc(evText)}</div>`;
                }
            }
            html += '</div>';
        }
        if (gt.ambiguous) {
            const modeLabel = curMode === 'transcript_relaxed'
                ? 'Turn in positive transcript but not itself misaligned'
                : 'Turn before a future misaligned turn';
            html += `<div class="amb-box">Ambiguous: ${modeLabel} — excluded from ${curMode.replace(/_/g,' ')} evaluation</div>`;
        }

        html += '<div class="turn-body">';
        for (const seg of turn.segments) {
            if (seg.role === 'user') {
                html += `<div class="msg"><div class="msg-role">User</div><pre class="user-text">${esc(seg.tokens.join(''))}</pre></div>`;
            } else if (seg.role === 'assistant') {
                html += `<div class="msg"><div class="msg-role">Assistant</div><div class="asst-tokens">${renderTokens(seg.tokens, seg.scores)}</div></div>`;
            }
        }
        html += '</div></div>';
    }

    el.innerHTML = html;
}

function renderAll() {
    renderProbeTabs();
    renderModeTabs();
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
    parser = argparse.ArgumentParser(
        description="Generate misalignment ground truth probe evaluation HTML dashboard"
    )
    parser.add_argument(
        "--results-subdir", default="v2_3",
        help="Subdirectory under probe_eval/results/ (default: v2_3)",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory of the project")
    parser.add_argument("--layer", default=None, help="Layer to visualize (e.g. 30). Auto-detected if not set.")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    if args.layer is not None:
        global LAYER
        LAYER = f"layer{args.layer}"

    base_dir = Path(args.base_dir)
    print(f"Building data (layer={LAYER})...")
    data = build_data(base_dir, args.results_subdir)

    print(f"  {len(data['probes'])} probes, {len(data['behaviors'])} behaviors")
    n_tokens = sum(
        len(v) for beh_data in data["tokens"].values() for v in beh_data.values()
    )
    print(f"  {n_tokens} total token arrays")

    data_json = json.dumps(data, separators=(",", ":"))

    title = f"Misalignment GT Probe Dashboard — {args.results_subdir}"
    html_content = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = base_dir / "probe_eval" / "results" / args.results_subdir / "misalignment_dashboard.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Generated {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
