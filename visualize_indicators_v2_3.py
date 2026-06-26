"""Generate an HTML visualization for v2.3 misalignment indicators.

Produces a single HTML file with two sections (tabbed):
  1. Tree Structure — shows indicators grouped by behavior, cross-cutting,
     preconditions, and behavioral concepts.
  2. Indicator Browser — interactive list, click to expand definitions,
     examples, and non-examples.
"""

import json
from indicators.misalignment_indicators_v2_3 import (
    MISALIGNMENT_INDICATORS_V2 as INDICATORS,
    MISALIGNMENT_PRECONDITIONS as PRECONDITIONS,
    BEHAVIORAL_CONCEPTS as CONCEPTS,
    DeceptionIndicatorCategory,
    RewardHackingIndicatorCategory,
    SandbaggingIndicatorCategory,
    UnderminingOversightIndicatorCategory,
    SabotageIndicatorCategory,
    PreservationIndicatorCategory,
    ResourceAcquisitionIndicatorCategory,
    RationalizationIndicatorCategory,
    InternalOutputDivergenceIndicatorCategory,
    AdversarialFramingIndicatorCategory,
    OversightResistanceIndicatorCategory,
    PreconditionCategory,
    BehavioralConceptCategory,
)


# ── Behavior colors ──────────────────────────────────────────────────────────
BEH = {
    "deception":       {"label": "Deception",              "color": "#c0392b"},
    "reward":          {"label": "Reward Hacking",         "color": "#e67e22"},
    "sandbagging":     {"label": "Sandbagging",            "color": "#8e44ad"},
    "oversight":       {"label": "Undermining Oversight",  "color": "#2c3e80"},
    "sabotage":        {"label": "Sabotage",               "color": "#922b21"},
    "preservation":    {"label": "Self-Preservation",      "color": "#1a7a5a"},
    "resource":        {"label": "Resource Acquisition",   "color": "#7d6608"},
    "rationalization": {"label": "Rationalization",        "color": "#b05020"},
    "divergence":      {"label": "Internal-Output Div.",   "color": "#2980b9"},
    "adversarial":     {"label": "Adversarial Framing",    "color": "#d35400"},
    "oversight_res":   {"label": "Oversight Resistance",   "color": "#6c3483"},
    "precondition":    {"label": "Precondition",           "color": "#7f8c8d"},
    "concept":         {"label": "Behavioral Concept",     "color": "#95a5a6"},
}

# Behavior-specific indicator groups (behavior key → category class)
BEHAVIOR_GROUPS = [
    {"name": "Deception",              "id": "deception",       "cls": DeceptionIndicatorCategory},
    {"name": "Reward Hacking",         "id": "reward",          "cls": RewardHackingIndicatorCategory},
    {"name": "Sandbagging",            "id": "sandbagging",     "cls": SandbaggingIndicatorCategory},
    {"name": "Undermining Oversight",  "id": "oversight",       "cls": UnderminingOversightIndicatorCategory},
    {"name": "Sabotage",               "id": "sabotage",        "cls": SabotageIndicatorCategory},
    {"name": "Self-Preservation",      "id": "preservation",    "cls": PreservationIndicatorCategory},
]

# Cross-cutting indicator groups
CROSS_GROUPS = [
    {"name": "Resource Acquisition",   "id": "resource",        "cls": ResourceAcquisitionIndicatorCategory},
    {"name": "Rationalization",        "id": "rationalization",  "cls": RationalizationIndicatorCategory},
    {"name": "Internal-Output Divergence", "id": "divergence",  "cls": InternalOutputDivergenceIndicatorCategory},
    {"name": "Adversarial User Framing",   "id": "adversarial", "cls": AdversarialFramingIndicatorCategory},
    {"name": "Oversight as Obstacle",  "id": "oversight_res",   "cls": OversightResistanceIndicatorCategory},
]


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _esc(s):
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def ind_to_dict(ind, max_examples=3):
    d = {
        "name": ind.name,
        "definition": ind.definition,
        "examples": ind.examples[:max_examples],
        "non_examples": ind.non_examples[:max_examples] if ind.non_examples else [],
    }
    if hasattr(ind, "note") and ind.note:
        d["note"] = ind.note
    return d


def concept_to_dict(c, max_examples=3):
    d = {
        "name": c.name,
        "definition": c.definition,
        "examples": c.examples[:max_examples],
        "non_examples": c.non_examples[:max_examples] if c.non_examples else [],
    }
    if hasattr(c, "note") and c.note:
        d["note"] = c.note
    return d


# ── Tree data ────────────────────────────────────────────────────────────────

def build_tree_data():
    """Build the tree hierarchy for the tree view."""
    tree = []

    # Behavior-specific groups
    for bg in BEHAVIOR_GROUPS:
        inds = [ind for ind in INDICATORS if isinstance(ind.category, bg["cls"])]
        if not inds:
            continue
        tree.append({
            "name": bg["name"],
            "beh": bg["id"],
            "level": "top",
            "count": len(inds),
            "children": [
                {"name": ind.name, "beh": bg["id"], "level": "mid",
                 "cat_label": ind.category.name.replace("_", " ").title()}
                for ind in inds
            ],
        })

    # Cross-cutting indicators
    cross_children = []
    for cg in CROSS_GROUPS:
        inds = [ind for ind in INDICATORS if isinstance(ind.category, cg["cls"])]
        for ind in inds:
            cross_children.append({
                "name": ind.name, "beh": cg["id"], "level": "mid",
            })
    if cross_children:
        tree.append({
            "name": "Cross-Cutting Indicators",
            "beh": "cross",
            "level": "top",
            "count": len(cross_children),
            "children": cross_children,
        })

    # Preconditions
    if PRECONDITIONS:
        tree.append({
            "name": "Preconditions",
            "beh": "precondition",
            "level": "top",
            "count": len(PRECONDITIONS),
            "children": [
                {"name": p.name, "beh": "precondition", "level": "mid"}
                for p in PRECONDITIONS
            ],
        })

    # Behavioral concepts
    if CONCEPTS:
        tree.append({
            "name": "Behavioral Concepts",
            "beh": "concept",
            "level": "top",
            "count": len(CONCEPTS),
            "children": [
                {"name": c.name, "beh": "concept", "level": "mid"}
                for c in CONCEPTS
            ],
        })

    return tree


def _tree_node_html(node):
    """Recursively render a tree node to HTML."""
    level = node.get("level", "top")
    beh = node.get("beh", "cross")
    children = node.get("children", [])
    count = node.get("count")
    cat_label = node.get("cat_label", "")

    lines = []
    lines.append(f'<div class="tree-row level-{level} beh-{beh}">')
    lines.append(f'  <div class="tree-node">{_esc(node["name"])}')
    if count is not None:
        lines.append(f'    <span class="count-badge">{count} indicators</span>')
    if cat_label:
        lines.append(f'    <span class="cat-label">{_esc(cat_label)}</span>')
    lines.append(f'  </div>')
    lines.append(f'</div>')

    if children:
        lines.append(f'<div class="tree-children">')
        for child in children:
            lines.append(_tree_node_html(child))
        lines.append(f'</div>')

    return "\n".join(lines)


def build_tree_html(tree):
    parts = ['<div class="tree">']
    for group in tree:
        parts.append('<div class="tree-level-0">')
        parts.append(_tree_node_html(group))
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


# ── Browser data ─────────────────────────────────────────────────────────────

def build_browser_data():
    # Behavior-specific
    behavior_specific = []
    for bg in BEHAVIOR_GROUPS:
        inds = [ind for ind in INDICATORS if isinstance(ind.category, bg["cls"])]
        if inds:
            behavior_specific.append({
                "name": bg["name"],
                "id": bg["id"],
                "color": BEH[bg["id"]]["color"],
                "indicators": [ind_to_dict(ind) for ind in inds],
            })

    # Cross-cutting (flat list)
    cross_cutting = []
    for cg in CROSS_GROUPS:
        inds = [ind for ind in INDICATORS if isinstance(ind.category, cg["cls"])]
        for ind in inds:
            cross_cutting.append({
                **ind_to_dict(ind),
                "color": BEH[cg["id"]]["color"],
            })

    # Preconditions (flat list)
    preconditions = []
    for p in PRECONDITIONS:
        preconditions.append({
            **ind_to_dict(p),
            "color": BEH["precondition"]["color"],
        })

    # Behavioral concepts (flat list)
    concepts = []
    for c in CONCEPTS:
        concepts.append({
            **concept_to_dict(c),
            "color": BEH["concept"]["color"],
        })

    return {
        "behavior_specific": behavior_specific,
        "cross_cutting": cross_cutting,
        "preconditions": preconditions,
        "concepts": concepts,
    }


# ── Full HTML ────────────────────────────────────────────────────────────────

def build_html() -> str:
    tree_data = build_tree_data()
    tree_html = build_tree_html(tree_data)
    browser_data = json.dumps(build_browser_data(), indent=2)

    n_indicators = len(INDICATORS)
    n_preconditions = len(PRECONDITIONS)
    n_concepts = len(CONCEPTS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Misalignment Indicators v2.3</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #fafafa;
    color: #333;
    padding: 24px 32px;
    line-height: 1.5;
    max-width: 1000px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #777; font-size: 13px; margin-bottom: 20px; }}

  /* ── Top-level section tabs ── */
  .section-tabs {{
    display: flex;
    gap: 0;
    border-bottom: 2px solid #ddd;
    margin-bottom: 24px;
  }}
  .section-tab {{
    padding: 10px 28px;
    cursor: pointer;
    font-size: 15px;
    font-weight: 600;
    color: #888;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.15s;
    user-select: none;
  }}
  .section-tab:hover {{ color: #555; }}
  .section-tab.active {{ color: #333; border-bottom-color: #333; }}
  .section-content {{ display: none; }}
  .section-content.active {{ display: block; }}

  /* ── Legend ── */
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
    font-size: 12px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  /* ── Tree ── */
  .tree {{ padding-left: 0; }}
  .tree-level-0 {{ margin-bottom: 24px; }}
  .tree-node {{
    position: relative;
    padding: 6px 14px;
    margin: 3px 0;
    border-radius: 6px;
    font-size: 13px;
    display: inline-block;
    border: 1.5px solid transparent;
    transition: all 0.2s;
  }}
  .tree-children {{
    margin-left: 28px;
    border-left: 2px solid #e0e0e0;
    padding-left: 16px;
  }}
  .tree-children > .tree-row::before {{
    content: '';
    position: absolute;
    left: -18px;
    top: 50%;
    width: 14px;
    border-top: 2px solid #e0e0e0;
  }}
  .tree-row {{ position: relative; }}
  .level-top .tree-node {{ font-size: 15px; font-weight: 600; padding: 8px 16px; }}
  .level-mid .tree-node {{ font-size: 13px; font-weight: 500; }}
  .count-badge {{
    display: inline-block;
    font-size: 10px;
    padding: 0 6px;
    border-radius: 8px;
    background: rgba(0,0,0,0.08);
    color: inherit;
    margin-left: 6px;
    font-weight: 400;
    font-style: normal;
  }}
  .cat-label {{
    display: inline-block;
    font-size: 10px;
    padding: 0 6px;
    border-radius: 8px;
    background: rgba(0,0,0,0.05);
    color: #999;
    margin-left: 6px;
    font-weight: 400;
    font-style: italic;
  }}

  /* Behavior colors for tree nodes */
  .beh-deception .tree-node       {{ background: #fdf2f2; color: {BEH["deception"]["color"]};       border-color: #e8b4b4; }}
  .beh-reward .tree-node          {{ background: #fef6ed; color: {BEH["reward"]["color"]};          border-color: #f0c8a0; }}
  .beh-sandbagging .tree-node     {{ background: #f8f0fc; color: {BEH["sandbagging"]["color"]};     border-color: #d0a8e0; }}
  .beh-oversight .tree-node       {{ background: #eef0f8; color: {BEH["oversight"]["color"]};       border-color: #a8b0d0; }}
  .beh-sabotage .tree-node        {{ background: #faf0ee; color: {BEH["sabotage"]["color"]};        border-color: #d4a8a2; }}
  .beh-preservation .tree-node    {{ background: #eef8f4; color: {BEH["preservation"]["color"]};    border-color: #a0d0c0; }}
  .beh-resource .tree-node        {{ background: #faf8ee; color: {BEH["resource"]["color"]};        border-color: #d4cc90; }}
  .beh-rationalization .tree-node {{ background: #fdf4ee; color: {BEH["rationalization"]["color"]}; border-color: #d4b0a0; }}
  .beh-divergence .tree-node      {{ background: #eef4fa; color: {BEH["divergence"]["color"]};      border-color: #a0c4e0; }}
  .beh-adversarial .tree-node     {{ background: #fdf2ee; color: {BEH["adversarial"]["color"]};     border-color: #e0b4a0; }}
  .beh-oversight_res .tree-node   {{ background: #f4eef8; color: {BEH["oversight_res"]["color"]};   border-color: #c0a0d4; }}
  .beh-cross .tree-node           {{ background: #f5f5f5; color: #555;                              border-color: #ccc; }}
  .beh-precondition .tree-node    {{ background: #f0f0f0; color: {BEH["precondition"]["color"]};    border-color: #c0c0c0; }}
  .beh-concept .tree-node         {{ background: #f4f4f4; color: {BEH["concept"]["color"]};         border-color: #ccc; }}

  /* ── v2.2 diff annotations ── */
  .v22-diff {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 6px;
    font-weight: 500;
    display: inline-block;
    vertical-align: middle;
  }}
  .v22-new       {{ background: #d4edda; color: #155724; }}
  .v22-renamed   {{ background: #fff3cd; color: #856404; }}
  .v22-merged    {{ background: #cce5ff; color: #004085; }}
  .v22-split     {{ background: #f8d7da; color: #721c24; }}

  /* ── Indicator Browser ── */
  .granularity-bar {{
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
    background: white;
  }}
  .granularity-btn {{
    flex: 1;
    padding: 12px 8px;
    border: none;
    background: white;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: #666;
    transition: all 0.15s;
    border-right: 1px solid #eee;
    text-align: center;
  }}
  .granularity-btn:last-child {{ border-right: none; }}
  .granularity-btn:hover {{ background: #f8f8f8; color: #333; }}
  .granularity-btn.active {{
    background: #2c3e50;
    color: white;
  }}
  .granularity-btn .g-label {{ display: block; }}
  .granularity-btn .g-count {{
    display: block;
    font-size: 11px;
    font-weight: 400;
    margin-top: 2px;
    opacity: 0.7;
  }}

  .behavior-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .behavior-tab {{
    padding: 10px 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.15s;
  }}
  .behavior-tab:hover {{ border-color: #999; }}
  .behavior-tab.active {{
    color: white;
    border-color: transparent;
  }}

  .indicator-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }}
  .indicator-btn {{
    padding: 10px 16px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    text-align: left;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .indicator-btn:hover {{
    border-color: #999;
    background: #fafafa;
  }}
  .indicator-btn.active {{
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, white);
    font-weight: 600;
  }}
  .indicator-btn .arrow {{
    margin-left: auto;
    color: #999;
    transition: transform 0.15s;
    flex-shrink: 0;
  }}
  .indicator-btn.active .arrow {{
    color: var(--accent);
    transform: rotate(90deg);
  }}
  .indicator-btn .color-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  .detail-panel {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 24px;
    display: none;
    animation: fadeIn 0.2s;
  }}
  .detail-panel.visible {{ display: block; }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .detail-panel h3 {{ font-size: 18px; margin-bottom: 12px; }}
  .detail-panel .definition {{
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 16px;
    color: #444;
  }}
  .detail-panel .note-text {{
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 16px;
    color: #666;
    padding: 10px 14px;
    background: #f0f4f8;
    border-radius: 6px;
    border-left: 3px solid #7f8c8d;
  }}
  .detail-panel .section-label {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }}
  .example {{
    font-size: 13px;
    line-height: 1.5;
    padding: 10px 14px;
    background: #f8f9fa;
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    margin-bottom: 6px;
    font-style: italic;
    color: #555;
  }}
  .non-example {{
    font-size: 13px;
    line-height: 1.5;
    padding: 10px 14px;
    background: #f8f9fa;
    border-left: 3px solid #ccc;
    border-radius: 0 4px 4px 0;
    margin-bottom: 6px;
    color: #888;
  }}
  .examples-section {{ margin-bottom: 16px; }}
  .placeholder {{
    color: #999;
    font-size: 14px;
    text-align: center;
    padding: 40px;
  }}
</style>
</head>
<body>

<h1>Misalignment Reasoning Indicators v2.3</h1>
<p class="subtitle">
  {n_indicators} indicators + {n_preconditions} preconditions + {n_concepts} behavioral concept
  &nbsp;&middot;&nbsp;
  Atomized from v2.2: compound indicators decomposed, overlapping merged
</p>

<!-- Top-level section tabs -->
<div class="section-tabs">
  <div class="section-tab active" data-section="tree">Tree Structure</div>
  <div class="section-tab" data-section="browser">Indicator Browser</div>
</div>

<!-- SECTION 1: Tree Structure -->
<div class="section-content active" id="section-tree">

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["deception"]["color"]}"></div>Deception</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["reward"]["color"]}"></div>Reward Hacking</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["sandbagging"]["color"]}"></div>Sandbagging</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["oversight"]["color"]}"></div>Undermining Oversight</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["sabotage"]["color"]}"></div>Sabotage</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["preservation"]["color"]}"></div>Self-Preservation</div>
  <div class="legend-item"><div class="legend-dot" style="background:#555"></div>Cross-Cutting</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["precondition"]["color"]}"></div>Preconditions</div>
  <div class="legend-item"><div class="legend-dot" style="background:{BEH["concept"]["color"]}"></div>Behavioral Concepts</div>
</div>

{tree_html}

</div>

<!-- SECTION 2: Indicator Browser -->
<div class="section-content" id="section-browser">

<div class="granularity-bar" id="granularityBar"></div>
<div class="behavior-tabs" id="behaviorTabs"></div>
<div class="indicator-list" id="indicatorList">
  <div class="placeholder">Select a category above</div>
</div>
<div class="detail-panel" id="detailPanel"></div>

</div>

<script>
// Section tab switching
document.querySelectorAll('.section-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.section-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('section-' + tab.dataset.section).classList.add('active');
  }});
}});

// Indicator Browser
const DATA = {browser_data};

const CATEGORIES = [
  {{ key: 'behavior_specific', label: 'Behavior-Specific',
     count: DATA.behavior_specific.reduce((s, b) => s + b.indicators.length, 0) + ' indicators' }},
  {{ key: 'cross_cutting', label: 'Cross-Cutting',
     count: DATA.cross_cutting.length + ' indicators' }},
  {{ key: 'preconditions', label: 'Preconditions',
     count: DATA.preconditions.length + ' indicators' }},
  {{ key: 'concepts', label: 'Behavioral Concepts',
     count: DATA.concepts.length + ' concepts' }},
];

let activeCategory = null;
let activeBehavior = null;
let activeIndicator = null;

function render() {{
  const gBar = document.getElementById('granularityBar');
  gBar.innerHTML = '';
  CATEGORIES.forEach(g => {{
    const btn = document.createElement('button');
    btn.className = 'granularity-btn' + (activeCategory === g.key ? ' active' : '');
    btn.innerHTML = '<span class="g-label">' + g.label + '</span><span class="g-count">' + g.count + '</span>';
    btn.onclick = () => {{
      if (activeCategory === g.key) return;
      activeCategory = g.key;
      activeBehavior = null;
      activeIndicator = null;
      render();
    }};
    gBar.appendChild(btn);
  }});

  const tabsEl = document.getElementById('behaviorTabs');
  const listEl = document.getElementById('indicatorList');
  const detailEl = document.getElementById('detailPanel');
  tabsEl.innerHTML = '';
  detailEl.className = 'detail-panel';
  detailEl.innerHTML = '';

  if (activeCategory === null) {{
    listEl.innerHTML = '<div class="placeholder">Select a category above</div>';
    return;
  }}

  if (activeCategory === 'behavior_specific') {{
    renderGrouped(DATA.behavior_specific, tabsEl, listEl, detailEl);
  }} else if (activeCategory === 'cross_cutting') {{
    renderFlatList(DATA.cross_cutting, listEl, detailEl);
  }} else if (activeCategory === 'preconditions') {{
    renderFlatList(DATA.preconditions, listEl, detailEl);
  }} else if (activeCategory === 'concepts') {{
    renderFlatList(DATA.concepts, listEl, detailEl);
  }}
}}

function renderGrouped(behaviors, tabsEl, listEl, detailEl) {{
  behaviors.forEach((b, i) => {{
    const btn = document.createElement('button');
    btn.className = 'behavior-tab' + (activeBehavior === i ? ' active' : '');
    btn.textContent = b.name;
    if (activeBehavior === i) {{
      btn.style.background = b.color;
      btn.style.borderColor = b.color;
    }}
    btn.onclick = () => {{
      activeBehavior = (activeBehavior === i) ? null : i;
      activeIndicator = null;
      render();
    }};
    tabsEl.appendChild(btn);
  }});

  if (activeBehavior === null) {{
    listEl.innerHTML = '<div class="placeholder">Select a behavior above</div>';
    return;
  }}

  const b = behaviors[activeBehavior];
  listEl.innerHTML = '';
  b.indicators.forEach((ind, j) => {{
    const btn = document.createElement('button');
    btn.className = 'indicator-btn' + (activeIndicator === j ? ' active' : '');
    btn.style.setProperty('--accent', b.color);
    btn.innerHTML = ind.name + '<span class="arrow">&#9654;</span>';
    btn.onclick = () => {{
      activeIndicator = (activeIndicator === j) ? null : j;
      render();
    }};
    listEl.appendChild(btn);
  }});

  if (activeIndicator !== null) {{
    renderDetail(detailEl, b.indicators[activeIndicator], b.color);
  }}
}}

function renderFlatList(items, listEl, detailEl) {{
  listEl.innerHTML = '';
  items.forEach((ind, j) => {{
    const btn = document.createElement('button');
    btn.className = 'indicator-btn' + (activeIndicator === j ? ' active' : '');
    btn.style.setProperty('--accent', ind.color);
    btn.innerHTML = '<span class="color-dot" style="background:' + ind.color + '"></span>' + ind.name + '<span class="arrow">&#9654;</span>';
    btn.onclick = () => {{
      activeIndicator = (activeIndicator === j) ? null : j;
      render();
    }};
    listEl.appendChild(btn);
  }});

  if (activeIndicator !== null && activeIndicator < items.length) {{
    const ind = items[activeIndicator];
    renderDetail(detailEl, ind, ind.color);
  }}
}}

function renderDetail(detailEl, ind, color) {{
  detailEl.style.setProperty('--accent', color);
  let h = '<h3>' + ind.name + '</h3>';
  h += '<div class="definition">' + ind.definition + '</div>';
  if (ind.note) {{
    h += '<div class="note-text"><strong>Note:</strong> ' + ind.note + '</div>';
  }}
  h += '<div class="examples-section">';
  h += '<div class="section-label" style="color: ' + color + '">Example reasoning traces</div>';
  ind.examples.forEach(ex => {{
    h += '<div class="example">&ldquo;' + ex + '&rdquo;</div>';
  }});
  h += '</div>';
  if (ind.non_examples && ind.non_examples.length > 0) {{
    h += '<div class="examples-section">';
    h += '<div class="section-label" style="color: #999">Not this indicator</div>';
    ind.non_examples.forEach(ne => {{
      h += '<div class="non-example">' + ne + '</div>';
    }});
    h += '</div>';
  }}
  detailEl.innerHTML = h;
  detailEl.className = 'detail-panel visible';
}}

render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    output_path = "indicators_v2_3.html"
    with open(output_path, "w") as f:
        f.write(build_html())
    print(f"Written to {output_path}")
