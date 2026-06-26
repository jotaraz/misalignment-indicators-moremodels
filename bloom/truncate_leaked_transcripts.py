"""Truncate transcripts at the first evaluator-leak point in user-view-target events.

For each rollout, find the first `role: user` event with `view: target` whose content
contains a leak phrase (e.g. <END>, "Final Assessment", "successfully demonstrated").
Drop that event and ALL subsequent events from BOTH:
  - rollout.json (inline transcript.events)
  - transcript_v{var}r{rep}.json

A backup of rollout.json is written to rollout_pretruncate.json before modification.
The transcript_v*r1.json files are overwritten in place (no per-file backup).
"""
import json, shutil
from pathlib import Path

LEAK_PHRASES = [
    '<END>', 'Final Assessment', 'Verdict:', 'rollout is complete',
    'rollout has successfully', 'successfully demonstrated', 'NOT successfully elicited',
    'demonstrated the target behavior',
]

DIRS = [
    'bloom-results/instructed-covert-code-sabotage_glm_4_7_flash',
    'bloom-results/instructed-strategic-sandbagging_glm_4_7_flash',
    'bloom-results/sycophancy',
    'bloom-results/self-preservation_glm_4_7_flash',
    'bloom-results/strategic-deception_glm_4_7_flash',
    'bloom-results-test/test_instructed-covert-code-sabotage_glm_4_7_flash',
    'bloom-results-test/test_instructed-strategic-sandbagging_glm_4_7_flash',
    'bloom-results-test/test_sycophancy_glm_4_7_flash',
    'bloom-results-test/test_self-preservation_glm_4_7_flash',
    'bloom-results-test/test_strategic-deception_glm_4_7_flash',
]

def find_leak_event_idx(events: list[dict]) -> int | None:
    """Return index of first leaked user-view-target event, or None."""
    for i, e in enumerate(events):
        msg = e.get('edit', {}).get('message', {})
        if msg.get('role') != 'user':
            continue
        view = e.get('view', [])
        if 'target' not in view:
            continue
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        if any(p in content for p in LEAK_PHRASES):
            return i
    return None


for d in DIRS:
    rl_path = Path(d) / 'rollout.json'
    if not rl_path.exists():
        print(f'SKIP {d} (no rollout.json)')
        continue

    # Backup rollout.json once
    backup_path = Path(d) / 'rollout_pretruncate.json'
    if not backup_path.exists():
        shutil.copy(rl_path, backup_path)

    rl = json.load(open(rl_path))
    truncated_count = 0
    truncate_summary = []
    for ro in rl['rollouts']:
        events = ro.get('transcript', {}).get('events', [])
        leak_idx = find_leak_event_idx(events)
        if leak_idx is None:
            continue
        original_len = len(events)
        # Truncate: keep events before leak_idx, drop leak_idx and everything after
        ro['transcript']['events'] = events[:leak_idx]
        new_len = leak_idx
        truncated_count += 1
        truncate_summary.append((ro['variation_number'], ro['repetition_number'], original_len, new_len))

        # Update separate transcript file
        var = ro['variation_number']
        rep = ro['repetition_number']
        sep_path = Path(d) / f'transcript_v{var}r{rep}.json'
        if sep_path.exists():
            sep = json.load(open(sep_path))
            sep['events'] = sep.get('events', [])[:leak_idx]
            json.dump(sep, open(sep_path, 'w'), indent=2)

    # Save truncated rollout.json
    json.dump(rl, open(rl_path, 'w'), indent=2)
    print(f'{d.split("/")[-1]:50} truncated {truncated_count}/{len(rl["rollouts"])}')
    for var, rep, ol, nl in truncate_summary[:3]:
        print(f'  v{var}r{rep}: {ol} -> {nl} events ({ol-nl} dropped)')
