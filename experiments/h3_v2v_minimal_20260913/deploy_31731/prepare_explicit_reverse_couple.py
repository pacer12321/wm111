#!/usr/bin/env python3
"""Prepare one disclosed 124-frame source crop; no generated target or model."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

SOURCE = Path('/cache/zhonghao/temporal_source_review_20260913/reorder04_couple/source.mp4')
SOURCE_SHA = 'a18b5f2dcabe5a97d30350de9e402039a7e6c0886442162b164887698d75d638'
ROOT = Path('/cache/zhonghao/h3/data/explicit_reverse_couple_124')
HOST = 'ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0'
FFMPEG = Path('/cache/zhonghao/h3/env/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2')
PROMPT = ('Reverse the temporal order of the entire reference clip: play all of its actions backward in time. '
          'Keep the same two people, their appearance, lighting, camera framing, and background.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    run = subprocess.run(['/usr/local/ffmpeg/bin/ffprobe', '-v', 'error', '-show_streams', '-show_format',
                          '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def main():
    if socket.gethostname() != HOST or ROOT.resolve() != ROOT or SOURCE.resolve() != SOURCE:
        raise RuntimeError('Wrong host or noncanonical sample path')
    if SOURCE.is_symlink() or digest(SOURCE) != SOURCE_SHA:
        raise RuntimeError('Original public source hash changed')
    original = probe(SOURCE)
    videos = [s for s in original['streams'] if s['codec_type'] == 'video']
    if (len(videos) != 1 or videos[0]['r_frame_rate'] != '24/1'
            or int(videos[0]['nb_frames']) != 228 or len(original['streams']) != 1):
        raise RuntimeError('Original source frame/audio metadata differs from reviewed clip')
    if not FFMPEG.is_file() or FFMPEG.is_symlink() or FFMPEG.resolve() != FFMPEG:
        raise RuntimeError('Verified private ffmpeg binary missing or redirected')
    # Only the exact empty directory created by the earlier failed attempt is reusable.
    if ROOT.exists():
        if ROOT.is_symlink() or not ROOT.is_dir() or any(ROOT.iterdir()):
            raise RuntimeError('Refusing to overwrite existing derived sample artifacts')
    else:
        ROOT.mkdir(parents=True, exist_ok=False)
    output = ROOT / 'source.mp4'
    command = [str(FFMPEG), '-v', 'error', '-nostdin', '-n', '-i', str(SOURCE),
               '-vf', 'trim=start_frame=104:end_frame=228,setpts=PTS-STARTPTS', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-r', '24', '-frames:v', '124', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    metadata = probe(output)
    video = [s for s in metadata['streams'] if s['codec_type'] == 'video'][0]
    if int(video['nb_frames']) != 124 or video['r_frame_rate'] != '24/1':
        raise RuntimeError('Derived source frame count/FPS verification failed')
    record = {
        'schema_version': 1, 'sample_id': 'explicit_reverse_couple_124',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_video': str(output), 'source_sha256': digest(output),
        'source_metadata': {'width': video['width'], 'height': video['height'], 'frames': 124,
                            'fps': 24, 'duration_seconds': float(metadata['format']['duration']), 'has_audio': False},
        'edit_prompt': PROMPT,
        'requested_generation': {'width': 1344, 'height': 768, 'fps': 24, 'duration_seconds': 5.0,
                                 'seed': 4101, 'num_inference_steps': 50, 'flow_shift': 12.0, 'audio_flow_shift': 3.0},
        'provenance': {'dataset': 'OmniEdit-Bench', 'official_source_id': 'Te_Temporal_reordering_04',
                       'source_url': 'https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/resolve/main/Temporal/Temporal%20Composition/Temporal_reordering/4_1280x720.mp4',
                       'original_path': str(SOURCE), 'original_sha256': SOURCE_SHA,
                       'start_frame': 104, 'end_frame_exclusive': 228, 'original_fps': 24,
                       'derived_source': True, 'instruction_is_self_authored': True,
                       'official_paired_target_available': False, 'gt_target_generated': False,
                       'license': 'CC-BY-NC-4.0 (dataset card; research use)'},
        'router_control': {'temporal_change_required': True, 'text_only_should_identify_temporal_edit': True,
                           'vlm_necessity_evidence': False},
        'processing_command': command, 'ffprobe': metadata,
        'notes': ['Source crop retains second kiss followed by chest-to-chest embrace; ordering manually reviewed.',
                  'The crop is not reversed: temporal reversal is the requested model edit, not input preprocessing.',
                  '124 source frames match the existing H3 5-second target frame-count convention; actual latent alignment must still be verified.',
                  'This is a derived source with a self-authored explicit instruction, not an official paired edit or a target GT.']}
    with (ROOT / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
