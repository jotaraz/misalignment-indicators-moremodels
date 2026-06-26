#!/usr/bin/env python3
"""
Generate a combined HTML viewer for indicator detection results.

Combines multiple HTML visualization files (individual runs, aggregated, filtered)
into a single file with clickable buttons to switch between views.

Usage:
    python generate_combined_indicator_html.py \
        --run1 results_run1.html \
        --run2 results_run2.html \
        --run3 results_run3.html \
        --unfiltered results_unfiltered.html \
        --filtered results_filtered.html \
        -o combined.html

    # Or use glob pattern
    python generate_combined_indicator_html.py \
        --input-dir indicator_results/ \
        --basename rollouts \
        -o combined.html
"""

import argparse
import re
from pathlib import Path


def extract_body_content(html_content: str) -> str:
    """Extract content between <body> tags."""
    # Find body content
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1)
    return html_content


def extract_styles(html_content: str) -> str:
    """Extract styles from HTML."""
    style_match = re.search(r'<style[^>]*>(.*?)</style>', html_content, re.DOTALL | re.IGNORECASE)
    if style_match:
        return style_match.group(1)
    return ""


def generate_combined_html(
    html_files: dict[str, str],
    output_file: str,
) -> None:
    """
    Generate combined HTML with tabs for different views.

    Args:
        html_files: Dict mapping view name to HTML file path
        output_file: Output file path
    """
    # Load all HTML files
    contents = {}
    styles_set = set()

    for view_name, filepath in html_files.items():
        if filepath and Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            contents[view_name] = extract_body_content(html_content)
            styles = extract_styles(html_content)
            if styles:
                styles_set.add(styles)
        else:
            print(f"Warning: File not found for {view_name}: {filepath}")

    if not contents:
        print("Error: No valid HTML files found")
        return

    # Combine styles (deduplicate)
    combined_styles = "\n".join(styles_set)

    # Generate tab buttons
    tab_buttons = []
    view_order = ['Run 1', 'Run 2', 'Run 3', 'Unfiltered (Aggregated)', 'Filtered']

    first_active = True
    for view_name in view_order:
        if view_name in contents:
            view_id = view_name.lower().replace(' ', '-').replace('(', '').replace(')', '')
            active_class = ' active' if first_active else ''
            # Add color indicators for different views
            if 'Run' in view_name:
                badge_class = 'badge-run'
            elif 'Unfiltered' in view_name:
                badge_class = 'badge-unfiltered'
            else:
                badge_class = 'badge-filtered'
            tab_buttons.append(
                f'<button class="view-tab{active_class}" data-view="{view_id}">'
                f'<span class="view-badge {badge_class}"></span>{view_name}</button>'
            )
            first_active = False

    tabs_html = '\n            '.join(tab_buttons)

    # Generate content sections
    content_sections = []
    first_active = True
    for view_name in view_order:
        if view_name in contents:
            view_id = view_name.lower().replace(' ', '-').replace('(', '').replace(')', '')
            hidden_class = '' if first_active else ' hidden'
            content_sections.append(
                f'<div class="view-content{hidden_class}" id="view-{view_id}">\n'
                f'        <div class="view-header"><h2>{view_name}</h2></div>\n'
                f'        {contents[view_name]}\n'
                f'    </div>'
            )
            first_active = False

    content_html = '\n    '.join(content_sections)

    # Generate the combined HTML
    combined_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Indicator Detection Results</title>
    <style>
        {combined_styles}

        /* Override and add combined view styles */
        html {{
            height: 100%;
            margin: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}

        .combined-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px 24px;
            border-radius: 10px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .combined-header h1 {{
            margin: 0;
            font-size: 1.5em;
        }}

        .combined-header p {{
            margin: 6px 0 0 0;
            opacity: 0.9;
            font-size: 0.9em;
        }}

        .view-tabs-container {{
            background: white;
            padding: 12px 20px;
            border-radius: 10px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .view-tabs-container h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1em;
        }}

        .view-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .view-tab {{
            padding: 8px 16px;
            border: 2px solid #ddd;
            background: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .view-tab:hover {{
            border-color: #667eea;
            background: #f0f0ff;
        }}

        .view-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }}

        .view-badge {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}

        .badge-run {{
            background: #4fc3f7;
        }}

        .badge-unfiltered {{
            background: #ffb74d;
        }}

        .badge-filtered {{
            background: #81c784;
        }}

        .view-tab.active .badge-run,
        .view-tab.active .badge-unfiltered,
        .view-tab.active .badge-filtered {{
            background: white;
        }}

        .view-content {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .view-content.hidden {{
            display: none;
        }}

        .view-header {{
            border-bottom: 2px solid #eee;
            padding: 12px 20px;
        }}

        .view-header h2 {{
            margin: 0;
            color: #333;
        }}

        /* Inner transcript tabs and content from visualize_transcripts.py */
        .view-content > .tabs-container {{
            margin: 0;
            border-radius: 0;
            box-shadow: none;
            border-bottom: 1px solid #eee;
        }}

        .view-content .transcript {{
            border-radius: 0;
            box-shadow: none;
        }}

        .view-content .transcript.hidden {{
            display: none;
        }}

        .view-content .transcript-messages {{
            height: 85vh;
            overflow-y: auto;
            padding: 4px 20px 20px 20px;
        }}

        /* Comparison info */
        .comparison-info {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px 15px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
            font-size: 0.9em;
        }}

        .comparison-info h4 {{
            margin: 0 0 6px 0;
            color: #1565c0;
        }}

        .comparison-info ul {{
            margin: 0;
            padding-left: 20px;
        }}

        .comparison-info li {{
            margin: 3px 0;
        }}
    </style>
</head>
<body>
    <div class="combined-header">
        <h1>Combined Indicator Detection Results</h1>
        <p>View and compare indicator detection across multiple runs, aggregated, and filtered results</p>
    </div>

    <div class="comparison-info">
        <h4>How to use this viewer:</h4>
        <ul>
            <li><strong>Run 1/2/3:</strong> Individual detection runs with different random seeds</li>
            <li><strong>Unfiltered (Aggregated):</strong> All unique indicators found across all runs (may include false positives)</li>
            <li><strong>Filtered:</strong> Validated indicators after a final judge pass (recommended for analysis)</li>
        </ul>
    </div>

    <div class="view-tabs-container">
        <h3>Select View:</h3>
        <div class="view-tabs">
            {tabs_html}
        </div>
    </div>

    {content_html}

    <script>
        // Tab switching functionality
        document.querySelectorAll('.view-tab').forEach(tab => {{
            tab.addEventListener('click', function() {{
                const viewId = this.dataset.view;

                // Update active tab
                document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');

                // Show selected view, hide others
                document.querySelectorAll('.view-content').forEach(content => {{
                    if (content.id === 'view-' + viewId) {{
                        content.classList.remove('hidden');
                    }} else {{
                        content.classList.add('hidden');
                    }}
                }});
            }});
        }});

        // Preserve inner tab state when switching outer views
        // (The inner transcript tabs from visualize_transcripts.py should still work)
    </script>
</body>
</html>
'''

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_html)

    print(f"Combined HTML saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined HTML viewer for indicator detection results'
    )

    # Individual file arguments
    parser.add_argument('--run1', help='Path to run 1 HTML file')
    parser.add_argument('--run2', help='Path to run 2 HTML file')
    parser.add_argument('--run3', help='Path to run 3 HTML file')
    parser.add_argument('--unfiltered', help='Path to unfiltered/aggregated HTML file')
    parser.add_argument('--filtered', help='Path to filtered HTML file')

    # Alternative: use directory and basename
    parser.add_argument('--input-dir', help='Input directory containing HTML files')
    parser.add_argument('--basename', help='Base name for auto-detecting files')

    parser.add_argument('-o', '--output', required=True, help='Output HTML file path')

    args = parser.parse_args()

    # Build file dict
    html_files = {}

    if args.input_dir and args.basename:
        # Auto-detect files
        input_dir = Path(args.input_dir)
        basename = args.basename

        candidates = {
            'Run 1': input_dir / f'{basename}_run1.html',
            'Run 2': input_dir / f'{basename}_run2.html',
            'Run 3': input_dir / f'{basename}_run3.html',
            'Unfiltered (Aggregated)': input_dir / f'{basename}_unfiltered.html',
            'Filtered': input_dir / f'{basename}_filtered.html',
        }

        for name, path in candidates.items():
            if path.exists():
                html_files[name] = str(path)
                print(f"Found: {name} -> {path}")
            else:
                print(f"Not found: {name} -> {path}")
    else:
        # Use individual file arguments
        if args.run1:
            html_files['Run 1'] = args.run1
        if args.run2:
            html_files['Run 2'] = args.run2
        if args.run3:
            html_files['Run 3'] = args.run3
        if args.unfiltered:
            html_files['Unfiltered (Aggregated)'] = args.unfiltered
        if args.filtered:
            html_files['Filtered'] = args.filtered

    if not html_files:
        print("Error: No HTML files specified")
        print("Use --run1, --run2, etc. or --input-dir with --basename")
        return

    generate_combined_html(html_files, args.output)


if __name__ == '__main__':
    main()
