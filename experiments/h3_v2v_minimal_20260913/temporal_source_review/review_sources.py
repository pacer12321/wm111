"""Bounded CPU-only review of selected public OmniEdit source clips; no model calls."""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/cache/zhonghao/temporal_source_review_20260913')
FFMPEG = '/usr/local/ffmpeg/bin/ffmpeg'
FFPROBE = '/usr/local/ffmpeg/bin/ffprobe'
BASE = 'https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/resolve/main/'
CASES = {
    'reorder04_couple': 'Temporal/Temporal%20Composition/Temporal_reordering/4_1280x720.mp4',
    'causal03_cube': 'Reasoning/Causal_reasoning_editing/3_1280x720.mp4',
    'reorder01_smoke': 'Temporal/Temporal%20Composition/Temporal_reordering/1_1280x720.mp4',
    'reorder02_drink': 'Temporal/Temporal%20Composition/Temporal_reordering/2_1280x720.mp4',
    'reorder07_jumps': 'Temporal/Temporal%20Composition/Temporal_reordering/7_1280x720.mp4',
    'reorder08_dance': 'Temporal/Temporal%20Composition/Temporal_reordering/8_1280x720.mp4',
    'reorder10_dog': 'Temporal/Temporal%20Composition/Temporal_reordering/10_1280x720.mp4',
    'speed01_motorcycle': 'Temporal/Motion%20Attribute/Motion_speed_change/1_1280x720.mp4',
}


def review(case, relative_url):
    folder = ROOT / case
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'source.mp4'
    url = BASE + relative_url
    if not source.exists():
        # Refuse unexpectedly large responses in this bounded source review.
        with requests.get(url, stream=True, timeout=(20, 40)) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(1024 * 128):
                data.extend(chunk)
                if len(data) > 15_000_000:
                    raise ValueError('Unexpected source size exceeds review cap')
        source.write_bytes(data)
    probe = json.loads(subprocess.check_output([
        FFPROBE, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(source),
    ], timeout=30))
    duration = float(probe['format']['duration'])
    if duration > 30:
        raise ValueError('Unexpected duration exceeds bounded review scope')
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    stride = max(1, round((numerator / denominator) / 4))
    # Select actual input frame indices, retaining exact source PTS via showinfo.
    result = subprocess.run([
        FFMPEG, '-hide_banner', '-nostdin', '-n', '-threads', '2', '-i', str(source),
        '-vf', f'select=not(mod(n\\,{stride})),scale=360:-2,showinfo',
        '-vsync', 'vfr', '-threads', '2', str(folder / 'frame_%03d.png'),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    times = [float(t) for t in re.findall(r'\bn:\s*\d+.*?\bpts_time:([\d.]+)', result.stderr)]
    frames = sorted(folder.glob('frame_*.png'))
    if len(times) != len(frames):
        raise ValueError(f'Timestamp/frame mismatch: {len(times)} != {len(frames)}')
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 18)
    sheets = []
    for first in range(0, len(frames), 16):
        part = frames[first:first + 16]
        cell_width, cell_height = 360, 230
        sheet = Image.new('RGB', (cell_width * 4, 45 + cell_height * math.ceil(len(part) / 4)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f'{case} | source PTS | duration {duration:.3f}s | stride {stride}', fill='black', font=font)
        for offset, frame in enumerate(part):
            column, row = offset % 4, offset // 4
            x, y = column * cell_width, 45 + row * cell_height
            draw.text((x + 5, y + 2), f'{times[first + offset]:.3f} s', fill='black', font=font)
            with Image.open(frame) as item:
                sheet.paste(item, (x, y + 26))
        path = folder / f'contact_{first // 16 + 1:02d}.jpg'
        sheet.save(path, quality=92)
        sheets.append(str(path))
    record = {
        'case': case, 'source_url': url, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'duration_seconds': duration, 'video_frame_rate': stream['avg_frame_rate'],
        'streams': [{'index': s['index'], 'type': s['codec_type'], 'codec': s['codec_name']} for s in probe['streams']],
        'sample_stride_source_frames': stride, 'sample_source_pts_seconds': times,
        'contact_sheets': sheets, 'probe': probe,
    }
    (folder / 'metadata.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k != 'probe'}), flush=True)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = sys.argv[1:] or list(CASES)[:4]
    if any(name not in CASES for name in selected):
        raise ValueError('Unknown review case')
    for name in selected:
        review(name, CASES[name])
