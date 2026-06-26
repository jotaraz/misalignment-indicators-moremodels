"""
Generate an HTML heatmap visualization for indicator consistency results.

Reads consistency_report.json from all v1/v2 result dirs and produces
a single interactive HTML file with behavior tabs and color-coded tables.

Usage:
    python bloom/generate_results_html.py
    python bloom/generate_results_html.py --output bloom/indicator_results/results.html
"""

import argparse
import json
from pathlib import Path

METRICS = [
    ("Binary agree", "binary_agreement_rate"),
    ("Binary agree (per-turn, all)", "binary_agreement_rate_per_turn"),
    ("Binary agree (per-turn, non-trivial)", "binary_agreement_rate_per_turn_nontrivial"),
    ("Ind-Name Jaccard", "mean_indicator_name_jaccard"),
    ("Ind-Name Jaccard (per-turn, all)", "mean_indicator_name_jaccard_per_turn"),
    ("Ind-Name Jaccard (per-turn, non-trivial)", "mean_indicator_name_jaccard_per_turn_nontrivial"),
]


def load_all_results(base_dir: Path, exclude_models: list[str] | None = None) -> dict:
    """Load all consistency reports grouped by behavior.

    Args:
        base_dir: Root directory containing version subdirectories.
        exclude_models: List of model name substrings to exclude (e.g. ["glm_flash"]).
    """
    exclude_models = exclude_models or []
    results = {}
    for version_dir in sorted(base_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        version = version_dir.name  # v1 or v2
        for behavior_dir in sorted(version_dir.iterdir()):
            if not behavior_dir.is_dir():
                continue
            behavior = behavior_dir.name
            if behavior not in results:
                results[behavior] = {}
            for config_dir in sorted(behavior_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                if any(ex in config_dir.name for ex in exclude_models):
                    continue
                report_path = config_dir / "consistency_report.json"
                if not report_path.exists():
                    continue
                report = json.loads(report_path.read_text())
                summary = report.get("summary", {})
                if not summary:
                    continue
                config_name = config_dir.name
                results[behavior][config_name] = summary
    return results


def value_to_color(val: float) -> str:
    """Map a 0-1 value to a green-yellow-red color scale."""
    if val is None:
        return "#f0f0f0"
    val = max(0.0, min(1.0, val))
    # Green (high) -> Yellow (mid) -> Red (low)
    if val >= 0.5:
        # Yellow to green
        t = (val - 0.5) * 2  # 0 to 1
        r = int(255 * (1 - t) + 76 * t)
        g = int(200 * (1 - t) + 175 * t)
        b = int(50 * (1 - t) + 80 * t)
    else:
        # Red to yellow
        t = val * 2  # 0 to 1
        r = int(220 * (1 - t) + 255 * t)
        g = int(80 * (1 - t) + 200 * t)
        b = int(60 * (1 - t) + 50 * t)
    return f"rgb({r},{g},{b})"


def text_color(val: float) -> str:
    """Return black or white text depending on background brightness."""
    if val is None:
        return "#999"
    return "#222"


def compute_average(results: dict) -> dict:
    """Compute per-config average across all behaviors."""
    # Collect all config names across behaviors
    all_configs = set()
    for configs in results.values():
        all_configs.update(configs.keys())

    avg = {}
    for config in sorted(all_configs):
        metric_sums = {}
        metric_counts = {}
        for behavior, configs in results.items():
            if config not in configs:
                continue
            summary = configs[config]
            for _, metric_key in METRICS:
                val = summary.get(metric_key)
                if val is not None:
                    metric_sums[metric_key] = metric_sums.get(metric_key, 0) + val
                    metric_counts[metric_key] = metric_counts.get(metric_key, 0) + 1
        avg[config] = {
            k: round(metric_sums[k] / metric_counts[k], 4)
            for k in metric_sums
            if metric_counts.get(k, 0) > 0
        }
    return avg


def generate_html(results: dict) -> str:
    behaviors = sorted(results.keys())

    # Add average across behaviors
    results_with_avg = dict(results)
    results_with_avg["average (all behaviors)"] = compute_average(results)
    all_tabs = behaviors + ["average (all behaviors)"]

    behavior_tabs = ""
    for i, b in enumerate(all_tabs):
        active = "active" if i == 0 else ""
        label = b.replace("_", " ").title()
        behavior_tabs += f'<button class="tab {active}" onclick="showBehavior(\'{b}\')">{label}</button>\n'

    tables_html = ""
    for i, behavior in enumerate(all_tabs):
        configs = results_with_avg[behavior]
        config_names = sorted(configs.keys())
        display = "block" if i == 0 else "none"

        table = f'<div class="behavior-table" id="table-{behavior}" style="display:{display}">\n'
        table += '<div class="table-wrapper"><table>\n<thead><tr><th class="metric-col">Metric</th>\n'
        for cn in config_names:
            label = cn.replace("_", " ")
            table += f'<th class="val-col"><div class="col-label">{label}</div></th>\n'
        table += '</tr></thead>\n<tbody>\n'

        for metric_label, metric_key in METRICS:
            table += f'<tr><td class="metric-cell">{metric_label}</td>\n'
            for cn in config_names:
                val = configs[cn].get(metric_key)
                if val is not None:
                    bg = value_to_color(val)
                    tc = text_color(val)
                    table += f'<td class="val-cell" style="background:{bg};color:{tc}">{val:.3f}</td>\n'
                else:
                    table += '<td class="val-cell empty">-</td>\n'
            table += '</tr>\n'

        table += '</tbody>\n</table></div></div>\n'
        tables_html += table

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indicator Judge — Inter-run Consistency</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    padding: 24px;
    color: #333;
}}
h1 {{
    font-size: 1.5rem;
    margin-bottom: 4px;
    color: #222;
}}
.subtitle {{
    font-size: 0.9rem;
    color: #777;
    margin-bottom: 20px;
}}
.tabs {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}}
.tab {{
    padding: 8px 18px;
    border: 2px solid #ddd;
    border-radius: 20px;
    background: white;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.15s;
    color: #555;
}}
.tab:hover {{
    border-color: #999;
    color: #222;
}}
.tab.active {{
    background: #333;
    color: white;
    border-color: #333;
}}
.table-wrapper {{
    overflow-x: auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 4px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82rem;
}}
thead th {{
    background: #fafafa;
    border-bottom: 2px solid #e0e0e0;
    padding: 8px 6px;
    text-align: center;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 1;
}}
.metric-col {{
    text-align: left !important;
    min-width: 240px;
    padding-left: 12px !important;
}}
.col-label {{
    writing-mode: horizontal-tb;
    font-size: 0.75rem;
    line-height: 1.2;
    max-width: 90px;
    word-wrap: break-word;
}}
.val-col {{
    min-width: 70px;
    max-width: 90px;
}}
tbody tr {{
    border-bottom: 1px solid #f0f0f0;
}}
tbody tr:hover {{
    outline: 2px solid #666;
    outline-offset: -1px;
    z-index: 2;
    position: relative;
}}
.metric-cell {{
    padding: 10px 12px;
    font-weight: 500;
    white-space: nowrap;
    background: #fafafa;
    border-right: 2px solid #e0e0e0;
}}
.val-cell {{
    padding: 10px 6px;
    text-align: center;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
    transition: transform 0.1s;
}}
.val-cell:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 0 2px #333;
    z-index: 3;
    position: relative;
}}
.val-cell.empty {{
    background: #f8f8f8;
    color: #ccc;
}}
.legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 16px;
    font-size: 0.8rem;
    color: #777;
}}
.legend-bar {{
    width: 200px;
    height: 14px;
    border-radius: 7px;
    background: linear-gradient(to right, rgb(220,80,60), rgb(255,200,50), rgb(76,175,80));
}}
.legend-label {{
    font-size: 0.75rem;
}}
</style>
</head>
<body>
<h1>Indicator Judge — Inter-run Consistency</h1>
<p class="subtitle">Consistency metrics across 3 independent runs (k=3, temperature=1). Higher is better.</p>

<div class="tabs">
{behavior_tabs}
</div>

{tables_html}

<div class="legend">
    <span class="legend-label">0.0</span>
    <div class="legend-bar"></div>
    <span class="legend-label">1.0</span>
    <span style="margin-left:8px; color:#999">← Color scale</span>
</div>

<script>
function showBehavior(behavior) {{
    document.querySelectorAll('.behavior-table').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('table-' + behavior).style.display = 'block';
    event.target.classList.add('active');
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization for consistency results")
    parser.add_argument("--base-dir", default="bloom/indicator_results", help="Base directory containing v1/ and v2/")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--exclude-models", nargs="*", default=["glm_flash"], help="Model name substrings to exclude (default: glm_flash)")
    parser.add_argument("--behaviors", nargs="*", default=None, help="Only include these behaviors (default: all)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    results = load_all_results(base, exclude_models=args.exclude_models)

    if not results:
        print("No results found!")
        return

    # Filter to requested behaviors
    if args.behaviors:
        results = {b: configs for b, configs in results.items() if b in args.behaviors}

    # Filter out behaviors with no configs
    results = {b: configs for b, configs in results.items() if configs}

    html = generate_html(results)
    out_path = Path(args.output) if args.output else base / "results.html"
    out_path.write_text(html)
    print(f"Generated {out_path}")
    for b, configs in sorted(results.items()):
        print(f"  {b}: {len(configs)} configs")


if __name__ == "__main__":
    main()
