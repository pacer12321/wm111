"""Create one new private color sample by byte copy; never alter the old sample."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import socket

ROOT = Path('/cache/zhonghao/h3')
OLD = ROOT / 'data/explicit_reverse_couple_124'
NEW = ROOT / 'data/shirt_red_couple_124'
SOURCE_SHA = '4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2'
PROMPT = ("Change only the man's pale shirt to a solid red shirt. "
          "Preserve the shirt's fabric, shape, and details, and keep the woman's clothing unchanged. "
          "Preserve the original actions, their order and timing, the people, camera viewpoint and motion, "
          "framing, lighting, and background.")


def regular(path):
    if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
        raise RuntimeError(f'Expected canonical regular file: {path}')
    return path


def digest(path):
    return hashlib.sha256(regular(path).read_bytes()).hexdigest()


def prepare():
    if (socket.gethostname() != 'ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0'
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            != '8e904236-bb23-44b5-b7ba-af73ba5c1f77'):
        raise RuntimeError('Wrong or changed 31731 host')
    if ROOT.resolve(strict=True) != ROOT or NEW.exists() or NEW.is_symlink():
        raise RuntimeError('New namespace already exists or private root is noncanonical')
    source = regular(OLD / 'source.mp4')
    old_manifest = regular(OLD / 'manifest.json')
    source_hash, manifest_hash = digest(source), digest(old_manifest)
    old = json.loads(old_manifest.read_text())
    meta = old['source_metadata']
    if (source_hash != SOURCE_SHA or source.stat().st_size != 1527004
            or old['source_sha256'] != SOURCE_SHA
            or meta != {'width': 1280, 'height': 720, 'frames': 124, 'fps': 24,
                        'duration_seconds': 5.167, 'has_audio': False}
            or old['provenance']['start_frame'] != 104
            or old['provenance']['end_frame_exclusive'] != 228):
        raise RuntimeError('Original source/provenance mismatch')
    manifest = {
        'schema_version': 1, 'sample_id': NEW.name,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_video': str(NEW / 'source.mp4'), 'source_sha256': source_hash,
        'source_metadata': meta, 'edit_prompt': PROMPT,
        'requested_generation': old['requested_generation'], 'provenance': old['provenance'],
        'router_control': {'temporal_change_required': False,
                           'text_only_should_identify_temporal_edit': False,
                           'vlm_necessity_evidence': False},
        'preparation': {'operation': 'exact_byte_copy_no_reencoding',
                        'copied_from': str(source), 'original_manifest': str(old_manifest),
                        'original_manifest_sha256': manifest_hash},
        'notes': ['Source is the existing unreversed clip; only the requested edit changed.',
                  'router_control is a manual intended-task label, never a VLM prediction.',
                  'Every source+edit must separately pass real VLM inference before ABC.',
                  'No ground-truth edited target exists; generated outputs require quality review.']}
    NEW.mkdir(exist_ok=False)
    with source.open('rb') as src, (NEW / 'source.mp4').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    if digest(NEW / 'source.mp4') != source_hash or digest(old_manifest) != manifest_hash:
        raise RuntimeError('Copy or original integrity changed; retain partial namespace for inspection')
    with (NEW / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'prepared': True, 'source': str(NEW / 'source.mp4'),
                      'sha256': source_hash, 'bytes': 1527004,
                      'manifest_sha256': digest(NEW / 'manifest.json'),
                      'old_manifest_unchanged': True, 'new_inference_started': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        parser.error('No file creation without --prepare')
    prepare()
