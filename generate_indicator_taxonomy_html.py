#!/usr/bin/env python3
"""
Generate an HTML page displaying the general and fine-grained misalignment indicators taxonomy.

Usage:
    python generate_indicator_taxonomy_html.py -o indicator_taxonomy.html
"""

import argparse
import json
from pathlib import Path

from indicators.general_misalignment_indicators import (
    GENERAL_MISALIGNMENT_INDICATORS,
    MisalignmentCategory,
)
from indicators.misalignment_indicators import (
    MISALIGNMENT_INDICATORS,
    DeceptionIndicatorCategory,
    MisalignedGoalIndicatorCategory,
    RewardHackingIndicatorCategory,
)

# Mapping from general category to fine-grained category enum
CATEGORY_MAPPING = {
    MisalignmentCategory.DECEPTION: DeceptionIndicatorCategory,
    MisalignmentCategory.REWARD_HACKING: RewardHackingIndicatorCategory,
    MisalignmentCategory.MISALIGNED_GOAL_PURSUING: MisalignedGoalIndicatorCategory,
}

CATEGORY_DISPLAY_NAMES = {
    MisalignmentCategory.DECEPTION: "Deception",
    MisalignmentCategory.REWARD_HACKING: "Reward Hacking",
    MisalignmentCategory.MISALIGNED_GOAL_PURSUING: "Misaligned Goal Pursuing",
}

CATEGORY_COLORS = {
    MisalignmentCategory.DECEPTION: ("#e74c3c", "#fdedec"),
    MisalignmentCategory.REWARD_HACKING: ("#f39c12", "#fef9e7"),
    MisalignmentCategory.MISALIGNED_GOAL_PURSUING: ("#8e44ad", "#f4ecf7"),
}

SUBCATEGORY_DISPLAY_NAMES = {
    # Deception
    DeceptionIndicatorCategory.MODELING: "Modeling",
    DeceptionIndicatorCategory.THREAT_ASSESSMENT: "Threat Assessment",
    DeceptionIndicatorCategory.SIMULATION: "Simulation",
    DeceptionIndicatorCategory.CHECKING: "Checking",
    # Reward Hacking
    RewardHackingIndicatorCategory.MODELING: "Modeling",
    RewardHackingIndicatorCategory.REVERSE_MODELING: "Reverse Modeling",
    RewardHackingIndicatorCategory.DECISION: "Decision",
    # Misaligned Goal Pursuing
    MisalignedGoalIndicatorCategory.THREAT_ASSESSMENT: "Threat Assessment",
    MisalignedGoalIndicatorCategory.ACTION_FRAMING: "Action Framing",
    MisalignedGoalIndicatorCategory.ACHIEVEMENT_ASSESSMENT: "Achievement Assessment",
}


def get_fine_grained_for_category(category: MisalignmentCategory):
    """Get fine-grained indicators belonging to a general category."""
    cat_enum = CATEGORY_MAPPING[category]
    return [ind for ind in MISALIGNMENT_INDICATORS if isinstance(ind.category, type(cat_enum.THREAT_ASSESSMENT if hasattr(cat_enum, 'THREAT_ASSESSMENT') else list(cat_enum)[0])) and type(ind.category) == cat_enum]


def get_fine_grained_by_subcategory(category: MisalignmentCategory):
    """Group fine-grained indicators by subcategory within a general category."""
    cat_enum = CATEGORY_MAPPING[category]
    grouped = {}
    for ind in MISALIGNMENT_INDICATORS:
        if type(ind.category) == cat_enum:
            subcat = ind.category
            if subcat not in grouped:
                grouped[subcat] = []
            grouped[subcat].append(ind)
    return grouped


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


def generate_html() -> str:
    sections = []

    for cat in MisalignmentCategory:
        cat_name = CATEGORY_DISPLAY_NAMES[cat]
        accent, bg = CATEGORY_COLORS[cat]

        # General indicators for this category
        general_inds = [g for g in GENERAL_MISALIGNMENT_INDICATORS if g.category == cat]

        # Fine-grained grouped by subcategory
        fine_grouped = get_fine_grained_by_subcategory(cat)

        # Build general indicators cards
        general_cards = ""
        for g in general_inds:
            examples_html = "".join(
                f'<li>{escape_html(ex)}</li>' for ex in g.examples
            )
            general_cards += f"""
            <div class="general-card" style="border-left: 4px solid {accent};">
                <div class="card-name">{escape_html(g.name)}</div>
                <div class="card-def">{escape_html(g.definition)}</div>
                {"<ul class='card-examples'>" + examples_html + "</ul>" if examples_html else ""}
            </div>"""

        # Build fine-grained indicators grouped by subcategory
        fine_section = ""
        for subcat in CATEGORY_MAPPING[cat]:
            subcat_name = SUBCATEGORY_DISPLAY_NAMES.get(subcat, subcat.name)
            inds = fine_grouped.get(subcat, [])
            if not inds:
                continue

            ind_cards = ""
            for ind in inds:
                examples_html = "".join(
                    f'<li>{escape_html(ex)}</li>' for ex in ind.examples
                )
                ind_cards += f"""
                <div class="fine-card">
                    <div class="card-name">{escape_html(ind.name)}</div>
                    <div class="card-def">{escape_html(ind.definition)}</div>
                    {"<ul class='card-examples'>" + examples_html + "</ul>" if examples_html else ""}
                </div>"""

            fine_section += f"""
            <div class="subcategory">
                <div class="subcategory-header" style="background: {accent}22; border-left: 3px solid {accent};">
                    <span class="subcategory-label" style="color: {accent};">{escape_html(subcat_name)}</span>
                    <span class="subcategory-count">{len(inds)} indicator{"s" if len(inds) != 1 else ""}</span>
                </div>
                <div class="fine-cards">{ind_cards}
                </div>
            </div>"""

        total_fine = sum(len(v) for v in fine_grouped.values())
        sections.append(f"""
    <div class="category-section">
        <div class="category-header" style="background: linear-gradient(135deg, {accent}, {accent}cc);">
            <h2>{escape_html(cat_name)}</h2>
            <div class="category-stats">{len(general_inds)} general &middot; {total_fine} fine-grained</div>
        </div>
        <div class="category-body" style="background: {bg};">
            <div class="level-section">
                <h3 class="level-title">General Indicators</h3>
                <div class="general-cards">{general_cards}
                </div>
            </div>
            <div class="arrow-divider">&#9660;</div>
            <div class="level-section">
                <h3 class="level-title">Fine-Grained Indicators</h3>
                {fine_section}
            </div>
        </div>
    </div>""")

    total_general = len(GENERAL_MISALIGNMENT_INDICATORS)
    total_fine = len(MISALIGNMENT_INDICATORS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Misalignment Indicator Taxonomy</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .page-header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .page-header h1 {{ font-size: 1.8em; margin-bottom: 8px; }}
        .page-header p {{ opacity: 0.9; font-size: 1em; }}
        .stats-bar {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
        }}
        .stat {{
            background: rgba(255,255,255,0.15);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .stat strong {{ font-size: 1.2em; }}
        .category-section {{
            margin-bottom: 24px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .category-header {{
            color: white;
            padding: 18px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .category-header h2 {{ font-size: 1.3em; }}
        .category-stats {{
            background: rgba(255,255,255,0.2);
            padding: 4px 14px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .category-body {{ padding: 20px 24px; }}
        .level-title {{
            font-size: 1.05em;
            color: #555;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid #ddd;
        }}
        .level-section {{ margin-bottom: 16px; }}
        .arrow-divider {{
            text-align: center;
            font-size: 1.5em;
            color: #aaa;
            margin: 8px 0;
        }}
        .general-cards {{ display: flex; flex-wrap: wrap; gap: 12px; }}
        .general-card {{
            background: white;
            padding: 14px 16px;
            border-radius: 8px;
            flex: 1 1 280px;
            max-width: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .subcategory {{ margin-bottom: 14px; }}
        .subcategory-header {{
            padding: 8px 14px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .subcategory-label {{ font-weight: 600; font-size: 0.95em; }}
        .subcategory-count {{ font-size: 0.8em; color: #888; }}
        .fine-cards {{ display: flex; flex-wrap: wrap; gap: 10px; padding-left: 12px; }}
        .fine-card {{
            background: white;
            padding: 12px 14px;
            border-radius: 8px;
            flex: 1 1 260px;
            max-width: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: 1px solid #eee;
        }}
        .card-name {{ font-weight: 600; font-size: 0.95em; margin-bottom: 4px; }}
        .card-def {{ font-size: 0.85em; color: #555; }}
        .card-examples {{
            margin-top: 8px;
            padding-left: 18px;
            font-size: 0.8em;
            color: #777;
        }}
        .card-examples li {{ margin-bottom: 3px; }}
    </style>
</head>
<body>
    <div class="page-header">
        <h1>Misalignment Indicator Taxonomy</h1>
        <p>General and fine-grained indicators for detecting misaligned reasoning</p>
        <div class="stats-bar">
            <div class="stat">General: <strong>{total_general}</strong></div>
            <div class="stat">Fine-Grained: <strong>{total_fine}</strong></div>
            <div class="stat">Categories: <strong>{len(MisalignmentCategory)}</strong></div>
        </div>
    </div>
    {"".join(sections)}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate indicator taxonomy HTML")
    parser.add_argument("-o", "--output", default="indicator_taxonomy.html", help="Output HTML file path")
    args = parser.parse_args()

    html = generate_html()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Indicator taxonomy HTML saved to: {output_path}")


if __name__ == "__main__":
    main()
