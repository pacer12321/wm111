"""Read nested event intervals; emit non-overlapping accounting, not kernel time.

Usage: python summarize_events.py <event_intervals.rankN.json> [...]
Never sum inclusive parent and child times. Stream intervals include host gaps
and synchronization waits. Cross-rank results must not be summed into latency.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def summarize(value):
    rows = value['rows']
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows)
    children = defaultdict(list)
    for r in rows:
        assert math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] >= 0
        if r['parent'] is not None:
            assert r['parent'] in by_id and r['parent'] < r['id']
            children[r['parent']].append(r)
    totals, calls = defaultdict(float), defaultdict(int)
    exclusive, warnings = defaultdict(float), []
    for r in rows:
        path = [r['name']]
        parent = r['parent']
        while parent is not None:
            p = by_id[parent]
            path.insert(0, p['name'])
            parent = p['parent']
        label = ' > '.join(path)
        totals[label] += r['elapsed_ms']
        calls[label] += 1
        remainder = r['elapsed_ms'] - sum(c['elapsed_ms'] for c in children[r['id']])
        if remainder < -0.05:
            warnings.append({'id': r['id'], 'negative_remainder_ms': remainder})
        # Keep signed numerical residuals: do not silently inflate totals by
        # clamping each negative measurement to zero.
        exclusive[label] += remainder
    roots = [r for r in rows if r['parent'] is None]
    assert len(roots) == 1 and roots[0]['name'] == 'dit_forward'
    whole = roots[0]['elapsed_ms']
    assert abs(sum(exclusive.values()) - whole) < max(0.01, whole * 1e-7)
    return {'case': value['case'], 'rank': value['rank'], 'forward': value['forward'],
            'metric': value['metric'], 'forward_elapsed_ms': whole,
            'not_kernel_busy_time': True, 'cross_rank_sum_is_not_latency': True,
            'inclusive_scopes_DO_NOT_SUM': [
                {'scope': key, 'calls': calls[key], 'inclusive_ms': total,
                 'percent_of_forward': 100 * total / whole if whole else 0}
                for key, total in sorted(totals.items(), key=lambda x: -x[1])],
            'nonoverlapping_scope_self_intervals': [
                {'scope': key, 'self_ms': total, 'percent_of_forward': 100 * total / whole if whole else 0}
                for key, total in sorted(exclusive.items(), key=lambda x: -x[1])],
            'warnings': warnings}


if __name__ == '__main__':
    print(json.dumps([summarize(json.loads(Path(name).read_text())) for name in sys.argv[1:]], indent=2))
