"""
HTML visualization for per-token probe scores on bloom rollouts.

Generates a self-contained HTML page that highlights tokens by probe score,
with red for high scores (deceptive direction) and blue for low scores.
"""

import html
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


def score_to_color(score: float, min_val: float, max_val: float) -> str:
    """Map a score to a background color: blue (low) -> white (mid) -> red (high)."""
    if max_val == min_val:
        return "rgba(255,255,255,0)"

    # Normalize to [-1, 1] range centered at midpoint
    mid = (max_val + min_val) / 2
    half_range = (max_val - min_val) / 2
    if half_range == 0:
        t = 0.0
    else:
        t = (score - mid) / half_range
        t = max(-1.0, min(1.0, t))

    if t > 0:
        # White -> Red
        r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
        a = 0.3 + 0.7 * t
    else:
        # White -> Blue
        r, g, b = int(255 * (1 + t)), int(255 * (1 + t)), 255
        a = 0.3 + 0.7 * (-t)

    return f"rgba({r},{g},{b},{a:.2f})"


def tokens_to_html(
    str_tokens: list[str],
    scores: list[float],
    min_val: float,
    max_val: float,
) -> str:
    """Render tokens as inline spans with background colors based on scores.

    Tokens with NaN scores (non-assistant tokens) are shown in plain grey.
    """
    parts = []
    for tok, s in zip(str_tokens, scores):
        escaped = html.escape(tok).replace("\n", "<br>")
        if s != s:  # NaN check
            parts.append(
                f'<span class="tok" style="color:#999" title="(not scored)">{escaped}</span>'
            )
        else:
            color = score_to_color(s, min_val, max_val)
            title = f"{s:.4f}"
            parts.append(
                f'<span class="tok" style="background:{color}" title="{title}">{escaped}</span>'
            )
    return "".join(parts)


def build_rollout_html(
    rollout_info: dict[str, Any],
    str_tokens: list[str],
    token_scores: list[float],
    prompt_score: float,
    extent: tuple[float, float],
) -> str:
    """Build HTML for a single rollout's token visualization."""
    bp = rollout_info["behavior_presence"]
    label = rollout_info["label"]
    idx = rollout_info["rollout_index"]

    label_class = "deceptive" if label == "deceptive" else "honest"
    label_badge = f'<span class="badge {label_class}">{label} (bp={bp})</span>'

    tok_html = tokens_to_html(str_tokens, token_scores, extent[0], extent[1])

    return f"""
    <div class="rollout-card">
        <div class="rollout-header">
            <h3>Rollout {idx + 1} {label_badge}</h3>
            <span class="prompt-score">Prompt Score: {prompt_score:.4f}</span>
        </div>
        <div class="token-container">
            {tok_html}
        </div>
    </div>
    """


def build_full_html(
    title: str,
    rollout_htmls: list[str],
    summary_metrics: dict[str, Any],
) -> str:
    """Build the complete HTML page with all rollouts."""
    # Summary table
    metrics_rows = ""
    for k, v in summary_metrics.items():
        if isinstance(v, float):
            formatted = f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
        else:
            formatted = str(v)
        metrics_rows += f"<tr><td>{html.escape(str(k))}</td><td>{formatted}</td></tr>\n"

    content = "\n".join(rollout_htmls)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ text-align: center; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; }}
        .summary-table {{
            margin: 0 auto 30px auto;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-table th, .summary-table td {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .summary-table th {{ background: #f0f0f0; }}
        .rollout-card {{
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .rollout-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }}
        .rollout-header h3 {{ margin: 0; font-size: 16px; }}
        .prompt-score {{
            font-family: monospace;
            font-size: 14px;
            color: #555;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
        }}
        .badge.deceptive {{ background: #ffdddd; color: #c00; }}
        .badge.honest {{ background: #ddffdd; color: #070; }}
        .token-container {{
            padding: 16px;
            line-height: 2.0;
            word-wrap: break-word;
        }}
        .tok {{
            display: inline;
            padding: 1px 0;
            border-radius: 2px;
            font-family: monospace;
            font-size: 13px;
            cursor: default;
            white-space: pre-wrap;
        }}
        .legend {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .legend-bar {{
            display: inline-block;
            width: 200px;
            height: 18px;
            background: linear-gradient(to right, rgba(100,100,255,0.8), rgba(255,255,255,0.3), rgba(255,100,100,0.8));
            border-radius: 3px;
            vertical-align: middle;
            margin: 0 8px;
        }}
        #scroll-to-top {{
            display: none;
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: #555;
            color: white;
            border: none;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            z-index: 1000;
            transition: opacity 0.2s;
        }}
        #scroll-to-top:hover {{
            background: #333;
        }}
    </style>
</head>
<body>
    <h1>{html.escape(title)}</h1>
    <div class="legend">
        Low score <span class="legend-bar"></span> High score
        &nbsp;&nbsp;(hover over tokens to see exact scores)
    </div>
    <table class="summary-table">
        <tr><th>Metric</th><th>Value</th></tr>
        {metrics_rows}
    </table>
    {content}
    <button id="scroll-to-top" title="Scroll to top">&#8679;</button>
    <script>
        const scrollBtn = document.getElementById('scroll-to-top');
        window.addEventListener('scroll', function() {{
            scrollBtn.style.display = window.scrollY > 300 ? 'block' : 'none';
        }});
        scrollBtn.addEventListener('click', function() {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }});
    </script>
</body>
</html>"""
