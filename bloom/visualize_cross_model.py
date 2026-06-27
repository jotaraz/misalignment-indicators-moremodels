#!/usr/bin/env python3
"""Visualize cross-model bloom conversations with ground-truth misalignment labels.

Renders an HTML page per rollout dir: the conversation between the bloom evaluator
(Claude) / environment and the probed target model, with the Opus ground-truth
overlaid — misaligned target turns flagged (with the judge's description) and each
indicator's spans highlighted inline.

Self-contained (stdlib only). Run locally:

    python bloom/visualize_cross_model.py bloom/bloom-results/sycophancy_gemma_2_27b
    python bloom/visualize_cross_model.py bloom/bloom-results/sandbagging_mistral_small_24b -o out.html
    python bloom/visualize_cross_model.py bloom/bloom-results/*_gemma_2_27b   # several dirs -> one file each

Reads <dir>/rollout.json and (if present) <dir>/rollout_misalignment_turns.json.
"""
import argparse
import html
import json
import re
from pathlib import Path

CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
header{position:sticky;top:0;background:#23272e;color:#fff;padding:10px 18px;z-index:10}
header b{color:#7dd3fc}
.legend{font-size:12px;color:#cbd5e1;margin-top:4px}
.legend span{margin-right:14px}
.wrap{max-width:1000px;margin:0 auto;padding:16px}
.rollout{background:#fff;border:1px solid #d8dee4;border-radius:8px;margin:16px 0;overflow:hidden}
.rollout>summary{cursor:pointer;padding:10px 14px;font-weight:600;background:#eef1f4;list-style:none}
.rollout>summary::-webkit-details-marker{display:none}
.rsum{font-weight:400;color:#475569;font-size:13px;margin:2px 0 0}
.body{padding:10px 14px}
.btn{background:#334155;color:#fff;border:0;border-radius:5px;font-size:12px;padding:3px 9px;margin-right:6px;cursor:pointer}
/* per-turn collapsible cell */
.turn{margin:8px 0;border-radius:6px;border-left:4px solid #cbd5e1;overflow:hidden}
.turn>summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:8px;padding:6px 12px}
.turn>summary::-webkit-details-marker{display:none}
.turn .tog::before{content:"\\25BE";color:#94a3b8;font-size:11px}
.turn:not([open]) .tog::before{content:"\\25B8"}
.turn .hl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;font-weight:700;white-space:nowrap}
.turn .preview{color:#475569;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.turn[open] .preview{display:none}
.tbody{padding:0 12px 9px}
.sys{background:#f1f5f9;border-color:#94a3b8}
.user{background:#eff6ff;border-color:#3b82f6}
.tool{background:#fefce8;border-color:#ca8a04}
.tool .resp,.tool .tbody{font-family:ui-monospace,monospace;font-size:12px}
.target{background:#f0fdf4;border-color:#22c55e}
.target>summary .hl{color:#15803d}
.target.bad{background:#fef2f2;border-color:#ef4444}
.target.bad>summary .hl{color:#b91c1c}
.reasoning{color:#6b7280;font-style:italic;background:#fafafa;border:1px dashed #d1d5db;border-radius:5px;padding:6px 8px;margin:4px 0;white-space:pre-wrap}
.resp{white-space:pre-wrap}
.badge{display:inline-block;background:#ef4444;color:#fff;font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;margin-left:8px}
.gt{margin-top:6px;padding:6px 9px;background:#fff7ed;border:1px solid #fdba74;border-radius:5px;font-size:13px}
.gt .lbl{font-weight:700;color:#9a3412}
.chip{display:inline-block;background:#fde68a;color:#92400e;font-size:11px;font-weight:600;padding:1px 7px;border-radius:9px;margin:2px 4px 0 0}
mark{background:#fde047;padding:0 1px;border-radius:2px}
"""


def _content_parts(content):
    """Return list of (kind, text) where kind in {reasoning, text, other}."""
    if isinstance(content, str):
        return [("text", content)]
    out = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                out.append(("other", str(b))); continue
            t = b.get("type")
            if t == "reasoning":
                out.append(("reasoning", b.get("reasoning", "")))
            elif t == "text":
                out.append(("text", b.get("text", "")))
            else:
                out.append(("other", json.dumps(b)[:2000]))
    else:
        out.append(("other", str(content)))
    return out


def _highlight(escaped_text, spans):
    """spans: list of (raw_span_text, indicator_name, css_class). Wrap first match of each."""
    for raw, name, cls in spans:
        if not raw or len(raw) < 4:
            continue
        esc = html.escape(raw)
        # whitespace-tolerant + case-insensitive match against the escaped content
        words = esc.split()
        if not words:
            continue
        pat = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
        m = pat.search(escaped_text)
        if m:
            tip = html.escape(name)
            escaped_text = (escaped_text[:m.start()]
                            + f'<mark class="{cls}" title="{tip}">' + m.group(0) + "</mark>"
                            + escaped_text[m.end():])
    return escaped_text


def render_turn(role, content, view, *, is_target, target_idx, gt_for_turn, model):
    """Render one event as an HTML turn block."""
    parts = _content_parts(content)
    misaligned = gt_for_turn is not None and gt_for_turn.get("misaligned")
    # span list for highlighting (indicator spans + misalignment evidence)
    spans = []
    if gt_for_turn:
        for ind in gt_for_turn.get("indicators", []):
            for s in ind.get("spans", []):
                spans.append((s, ind["indicator_name"], "indicator"))

    if is_target:
        cls = "target bad" if misaligned else "target"
        label = f"TARGET — {model}" + (f"  ·  turn {target_idx}" if target_idx is not None else "")
    elif role == "user":
        cls, label = "user", "EVALUATOR / USER  (Claude)"
    elif role == "system":
        cls, label = "sys", "SYSTEM PROMPT"
    elif role == "tool":
        cls, label = "tool", "TOOL RESULT"
    else:  # assistant but not target = evaluator scaffolding
        cls, label = "sys", "EVALUATOR (scenario setup)"

    html_parts = []
    for kind, text in parts:
        esc = html.escape(text)
        esc = _highlight(esc, spans if kind in ("text", "reasoning") else [])
        if kind == "reasoning":
            html_parts.append(f'<div class="reasoning">💭 {esc}</div>')
        else:
            html_parts.append(f'<div class="resp">{esc}</div>')

    badge = ' <span class="badge">⚠ MISALIGNED</span>' if misaligned else ""
    gt_html = ""
    if misaligned:
        desc = html.escape(gt_for_turn.get("description", ""))
        inds = gt_for_turn.get("indicators", [])
        chips = "".join(f'<span class="chip">{html.escape(i["indicator_name"])}</span>' for i in inds)
        gt_html = (f'<div class="gt"><span class="lbl">GT judge:</span> {desc}'
                   + (f'<div>{chips}</div>' if chips else "") + "</div>")

    # one-line preview shown when the cell is collapsed
    full_text = "\n".join(t for _, t in parts).strip()
    first_line = next((ln.strip() for ln in full_text.splitlines() if ln.strip()), "")
    preview = html.escape(first_line[:160] + ("…" if len(first_line) > 160 else ""))

    # default-collapse only long system/scaffolding cells; everything else open
    open_attr = "" if (cls.startswith("sys") and len(full_text) > 400) else " open"
    return (f'<details class="turn {cls}"{open_attr}>'
            f'<summary><span class="tog"></span>'
            f'<span class="hl">{label}{badge}</span>'
            f'<span class="preview">{preview}</span></summary>'
            f'<div class="tbody">{"".join(html_parts)}{gt_html}</div></details>')


def _find_turn(norm_targets, probes, turn_index_fallback):
    """Locate the 0-based target turn for a GT entry by matching its quoted text.

    GT turn_index counting differs by modality (1-based for conversation, 0-based-ish
    for agentic/simenv), so we anchor on the verbatim evidence/spans instead and only
    fall back to turn_index when no quote matches.
    """
    for p in probes:
        if not p or len(p) < 8:
            continue
        np = re.sub(r"\s+", " ", p).lower()[:80]
        for i, t in enumerate(norm_targets):
            if np in t:
                return i
    return turn_index_fallback


def build_gt_index(gt, var, rep, target_texts):
    """Return {target_turn_index: {misaligned, description, indicators}} for one rollout.

    target_texts: per-target-turn concatenated text (for content-based turn matching).
    """
    if not gt:
        return {}, None
    norm = [re.sub(r"\s+", " ", t).lower() for t in target_texts]
    for er in gt.get("evaluated_rollouts", []):
        if er.get("variation_number") == var and er.get("repetition_number") == rep:
            idx = {}
            for mt in er.get("misaligned_turns", []):
                ti = _find_turn(norm, mt.get("evidence", []), mt.get("turn_index", 1) - 1)
                idx.setdefault(ti, {})["misaligned"] = True
                idx[ti]["description"] = mt.get("description", "")
            for it in er.get("indicators_turns", []):
                spans = [s for ind in it.get("present_indicators", []) for s in ind.get("spans", [])]
                ti = _find_turn(norm, spans, it.get("turn_index", 1) - 1)
                idx.setdefault(ti, {})["indicators"] = it.get("present_indicators", [])
            return idx, er.get("summary", "")
    return {}, None


def render_dir(d: Path) -> str:
    rollout = json.load(open(d / "rollout.json"))
    model = rollout.get("metadata", {}).get("target", "?")
    gt = None
    gtp = d / "rollout_misalignment_turns.json"
    if gtp.exists():
        gt = json.load(open(gtp))

    blocks = []
    for r in rollout.get("rollouts", []):
        var, rep = r.get("variation_number"), r.get("repetition_number")
        events = r["transcript"]["events"]
        # pass 1: per-target-turn text, used to anchor GT labels by quoted content
        target_texts = []
        for e in events:
            m = e.get("edit", {}).get("message") if isinstance(e.get("edit"), dict) else None
            if isinstance(m, dict) and m.get("role") == "assistant" and "target" in e.get("view", []):
                target_texts.append(" ".join(t for _, t in _content_parts(m.get("content"))))
        gt_idx, summary = build_gt_index(gt, var, rep, target_texts)
        turns_html = []
        target_i = -1
        n_bad = 0
        for e in events:
            m = e.get("edit", {}).get("message") if isinstance(e.get("edit"), dict) else None
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            view = e.get("view", [])
            is_target = role == "assistant" and "target" in view
            ti = None
            gt_for_turn = None
            if is_target:
                target_i += 1
                ti = target_i
                g = gt_idx.get(ti)
                if g:
                    gt_for_turn = g
                    if g.get("misaligned"):
                        n_bad += 1
            turns_html.append(render_turn(role, m.get("content"), view,
                                          is_target=is_target, target_idx=ti,
                                          gt_for_turn=gt_for_turn, model=model))
        vdesc = html.escape((r.get("variation_description") or "")[:160])
        head = (f'v{var}r{rep}'
                + (f' <span class="badge">{n_bad} misaligned turn(s)</span>' if n_bad else ''))
        sumtxt = html.escape((summary or "")[:400]) if summary else vdesc
        blocks.append(
            f'<details class="rollout"><summary>{head}<div class="rsum">{sumtxt}</div></summary>'
            f'<div class="body">{"".join(turns_html)}</div></details>')

    n_rollouts = len(rollout.get("rollouts", []))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(d.name)}</title>
<style>{CSS}</style></head><body>
<header><b>{html.escape(d.name)}</b> &nbsp; model={html.escape(model)} &nbsp; rollouts={n_rollouts}
&nbsp; <button class="btn" onclick="document.querySelectorAll('details.turn').forEach(d=>d.open=true)">▾ expand all turns</button>
<button class="btn" onclick="document.querySelectorAll('details.turn').forEach(d=>d.open=false)">▸ collapse all turns</button>
<div class="legend"><span>🟩 target</span><span>🟦 evaluator/user</span><span>⬜ system/setup</span>
<span>🟥 GT-misaligned turn</span><span><mark>indicator span</mark></span><span>💭 reasoning</span>
<span>(click any turn header to minimize it)</span></div>
</header><div class="wrap">{"".join(blocks)}</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="rollout dir(s) (each with rollout.json [+ GT])")
    ap.add_argument("-o", "--output", default=None,
                    help="output html (single dir only); else writes <dir>/visualization_gt.html")
    args = ap.parse_args()

    for ds in args.dirs:
        d = Path(ds)
        if not (d / "rollout.json").exists():
            print(f"skip {d}: no rollout.json"); continue
        out = Path(args.output) if (args.output and len(args.dirs) == 1) else d / "visualization_gt.html"
        out.write_text(render_dir(d), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
