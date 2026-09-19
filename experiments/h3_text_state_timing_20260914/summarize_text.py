"""Per-rank text-state intervals. Never sum parallel rank times into latency."""
import json
import math
from pathlib import Path


def summarize_rank(records, request_seconds):
    assert [r['forward'] for r in records] == list(range(1, 50))
    rank = records[0]['rank']
    totals = {s: {'calls': 0, 'npu_interval_seconds': 0., 'cpu_enqueue_seconds': 0.} for s in ('T', 'S')}
    forward_seconds = 0.
    for record in records:
        assert record['rank'] == rank and record['world'] == 4
        assert [(r['side'], r['layer']) for r in record['text_calls']] == [(s, i) for i in range(50) for s in ('T','S')]
        forward_seconds += record['forward_npu_interval_ms'] / 1000
        for row in record['text_calls']:
            assert math.isfinite(row['npu_interval_ms']) and row['npu_interval_ms'] >= 0
            target = totals[row['side']]
            target['calls'] += 1
            target['npu_interval_seconds'] += row['npu_interval_ms'] / 1000
            target['cpu_enqueue_seconds'] += row['cpu_enqueue_seconds']
    total = sum(r['npu_interval_seconds'] for r in totals.values())
    return {'rank': rank, 'T': totals['T'], 'S': totals['S'],
            'T_plus_S_interval_seconds': total,
            'T_plus_S_percent_of_instrumented_request': total / request_seconds * 100,
            'T_plus_S_percent_of_forward_intervals': total / forward_seconds * 100,
            'second_copy_S_interval_seconds_NOT_guaranteed_savings': totals['S']['npu_interval_seconds'],
            'sum_forward_interval_seconds': forward_seconds}


def summarize_run(profile_dir, request_seconds):
    assert request_seconds > 0
    ranks = []
    for rank in range(4):
        rows = [json.loads(s) for s in (Path(profile_dir) / f'text_state.rank{rank}.jsonl').read_text().splitlines()]
        ranks.append(summarize_rank(rows, request_seconds))
    return {'measurement': 'original_text_state_T_and_S_block_only',
            'request_seconds_includes_timing_overhead': request_seconds,
            'physical_cards': [0,1,2,3], 'world_size': 4, 'ranks_NOT_additive': ranks,
            'limits': ['NPU intervals include host dispatch gaps, not just kernel busy time.',
                       'A deduplication can remove one text-state copy, not both.',
                       'Measured intervals are not guaranteed end-to-end savings.',
                       'Four-card results cannot directly explain the old eight-card 84.01-second C/D difference.',
                       'Forward-boundary synchronization and event collection add measurement overhead.']}
