"""Bounded official metadata search only; no model calls or media generation."""
import collections
import io
import itertools
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://storage.googleapis.com/dm-perception-test/zip_data/challenge_action_localisation_valid_annotations.zip'
EXPLORER = 'https://ptchallenge-workshop.github.io/data.json'


def fetch(url, limit=2_000_000):
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise RuntimeError('Metadata unexpectedly exceeds per-file cap')
    return payload


def main():
    z = zipfile.ZipFile(io.BytesIO(fetch(URL)))
    print('Annotation members:', z.namelist())
    actions = json.loads(z.read(next(name for name in z.namelist() if name.endswith('.json'))))
    explorer_path = ROOT / 'perception_explorer.json'
    explorer = json.loads(explorer_path.read_text()) if explorer_path.is_file() else json.loads(fetch(EXPLORER))
    (ROOT / 'perception_action_valid.json').write_text(json.dumps(actions, indent=2))
    if not explorer_path.exists():
        explorer_path.write_text(json.dumps(explorer, indent=2))
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    skip = {'Other', 'Holding something in a state', 'Moving object(s) around'}
    for video, record in actions.items():
        if video not in explorer:
            continue
        segments = record['action_localisation']
        counts = collections.Counter(s['label'] for s in segments)
        unique = [s for s in segments if counts[s['label']] == 1 and s['label'] not in skip]
        for a, b in itertools.combinations(unique, 2):
            a, b = sorted((a, b), key=lambda s: s['timestamps'][0])
            if a['timestamps'][1] > b['timestamps'][0]:
                continue
            labels = (a['label'], b['label'])
            pair_key = tuple(sorted(labels))
            pairs[pair_key][labels].append({'video': video, 'metadata': record['metadata'],
                'events': [a, b], 'question_records': explorer[video]})
    candidates = []
    for key, orders in pairs.items():
        if len(orders) != 2:
            continue
        grouped = []
        for order, records in orders.items():
            records.sort(key=lambda r: r['metadata']['num_frames'] / r['metadata']['frame_rate'])
            grouped.append({'order': order, 'videos': records[:2]})
        candidates.append({'events': key, 'directions': grouped})
    (ROOT / 'natural_pair_metadata_candidates.json').write_text(json.dumps(candidates, indent=2))
    print(json.dumps({'annotated_videos': len(actions), 'explorer_videos': len(explorer),
                      'intersection': len(set(actions) & set(explorer)), 'pair_event_types': len(candidates)}))
    for item in candidates[:12]:
        print(json.dumps({'events': item['events'], 'directions': [
            {'order': direction['order'], 'videos': [
                {'video': r['video'], 'duration': r['metadata']['num_frames']/r['metadata']['frame_rate'],
                 'times': [s['timestamps'] for s in r['events']]} for r in direction['videos']]}
            for direction in item['directions']]}))


if __name__ == '__main__':
    main()
