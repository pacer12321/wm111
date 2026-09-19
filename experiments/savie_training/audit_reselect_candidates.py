"""CPU-only candidate inventory for a new, stratified RANDOM training selection.

Does not train, encode, overwrite caches, or declare visual QC passed. Metadata
checks are necessary but not sufficient for instruction and temporal alignment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time


def probe(path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1',
            '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration',
            '-of', 'json', str(path)], capture_output=True, text=True,
            check=True, timeout=20)
        data = json.loads(result.stdout)
        s = data['streams'][0]
        n = s.get('nb_frames')
        return dict(path=str(path), width=int(s['width']), height=int(s['height']),
            fps=float(Fraction(s.get('avg_frame_rate') or '0/1')),
            frames=int(n) if n and n != 'N/A' else None,
            duration=float(s.get('duration') or data.get('format', {}).get('duration') or 0))
    except Exception as e:
        return dict(path=str(path), error=type(e).__name__ + ': ' + str(e)[:160])


def run(args):
    begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    paths = sorted({r[k] for r in rows for k in ('source_relpath', 'target_relpath')})
    resolved = {}
    for rel in paths:
        for root in args.video_root:
            path = root / rel
            if path.is_file():
                resolved[rel] = path
                break
    print(json.dumps(dict(stage='inventory', candidates=len(rows),
        unique_files=len(paths), present_files=len(resolved))), flush=True)
    probes = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with (args.output/'file_probes.jsonl').open('x') as f:
            for rel, value in zip(resolved, pool.map(probe, resolved.values())):
                probes[rel] = value
                f.write(json.dumps(dict(relative_path=rel, **value))+'\n')
                if len(probes) % 200 == 0:
                    f.flush()
                    print(json.dumps(dict(stage='probe', completed=len(probes),
                        total=len(resolved), seconds=round(time.monotonic()-begin, 1))), flush=True)
    categories = Counter(); present_categories = Counter(); eligible_categories = Counter()
    exclusions = Counter(); hist = Counter(); frames = Counter(); fps = Counter(); aspects = Counter()
    locations = Counter(); groups = set(); lengths = []
    with (args.output/'candidate_inventory.jsonl').open('x') as out:
        for row in rows:
            category = row.get('category', 'unknown'); categories[category] += 1
            s = probes.get(row['source_relpath']); t = probes.get(row['target_relpath'])
            reasons = []
            if s is None or t is None: reasons.append('missing_original_pair')
            elif 'error' in s or 'error' in t: reasons.append('ffprobe_failed')
            else:
                present_categories[category] += 1
                duration = min(s['duration'], t['duration'])
                if duration < 2: reasons.append('under_two_seconds')
                if abs(s['duration']-t['duration']) > max(.25, .05*s['duration']):
                    reasons.append('duration_mismatch')
                if abs(s['fps']-t['fps']) > 1: reasons.append('fps_mismatch')
                if min(s['width'],s['height'],t['width'],t['height']) < 256:
                    reasons.append('under_256_pixels')
            # This is a text-candidate label only, never a visual-QC certificate.
            if row.get('edit_type_heuristic') != 'content_edit_candidate':
                reasons.append('temporal_label_not_cleared')
            enriched = dict(row, source_probe=s, target_probe=t,
                metadata_exclusions=reasons, metadata_eligible=not reasons,
                visual_qc_passed=False, motion_bin='unmeasured', edit_extent_bin='unmeasured')
            if not reasons:
                eligible_categories[category] += 1
                lengths.append(duration); frames[str((s['frames'], t['frames']))] += 1
                fps[str((s['fps'],t['fps']))] += 1
                b = '<3s' if duration < 3 else '3-4s' if duration < 4 else '4-5s' if duration < 5 else '5-6s' if duration < 6 else '>=6s'
                hist[b] += 1
                ratio = s['width']/s['height']
                aspects['landscape' if ratio>1.1 else 'portrait' if ratio<.9 else 'square'] += 1
                loc = '/'.join(Path(row['source_relpath']).parts[1:-1]); locations[loc] += 1
                groups.add(row['source_relpath'])
                enriched.update(duration_bin=b, common_duration_seconds=duration,
                    native_frame_count_pair=[s['frames'],t['frames']],
                    source_group=row['source_relpath'])
            exclusions.update(reasons)
            out.write(json.dumps(enriched,ensure_ascii=False)+'\n')
    summary = dict(manifest=str(args.manifest),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        candidates=len(rows), candidate_categories=dict(categories),
        present_pair_categories=dict(present_categories), eligible_categories=dict(eligible_categories),
        unique_eligible_sources=len(groups), exclusions=dict(exclusions), duration_bins=dict(hist),
        native_frames_pairs=dict(frames), native_fps_pairs=dict(fps), aspect_bins=dict(aspects),
        source_scene_path_counts=dict(locations),
        duration_min=min(lengths,default=None),duration_max=max(lengths,default=None),
        seconds=round(time.monotonic()-begin,2),
        status='inventory_only_not_a_frozen_2k_or_quality_certificate',
        selection_rule='source-disjoint stratification followed by seeded random sampling; quotas need approval',
        required_remaining_checks=['instruction compliance','temporal alignment','motion',
            'edit extent','matched Qwen/VAE clip and geometry','actual H3 encoded length',
            'short/mid/long and local/global coverage','independent validation split'])
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--video-root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
