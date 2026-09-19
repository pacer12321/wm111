"""Two bounded public source downloads and CPU-only contact sheets, never VLM/NPU."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.request

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'video_6578': (15_341_984, '4dac7f97753f8405e8d4944467029c04'),
    'video_1374': (26_863_094, '4a92c1581d8150cb4ad6066995ecf29f'),
}
FFMPEG = '/usr/local/ffmpeg/bin/ffmpeg'
FFPROBE = '/usr/local/ffmpeg/bin/ffprobe'
MAX_MEDIA_BYTES = 42_300_000


def review(video, size, expected_md5):
    folder = ROOT / video
    folder.mkdir(exist_ok=True)
    path = folder / 'source.mp4'
    url = f'https://storage.googleapis.com/dm-perception-test/visualisation_videos/{video}.mp4'
    if not path.exists():
        with urllib.request.urlopen(url, timeout=40) as response:
            if int(response.headers['Content-Length']) != size:
                raise RuntimeError('Unexpected public media size')
            data = response.read(size + 1)
        if len(data) != size or hashlib.md5(data).hexdigest() != expected_md5:
            raise RuntimeError('Downloaded public media differs from official object metadata')
        with path.open('xb') as stream:
            stream.write(data)
    if path.stat().st_size != size or hashlib.md5(path.read_bytes()).hexdigest() != expected_md5:
        raise RuntimeError('Existing source failed byte identity check')
    probe = json.loads(subprocess.check_output([
        FFPROBE, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path),
    ], timeout=30))
    track = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    numerator, denominator = map(int, track['avg_frame_rate'].split('/'))
    stride = max(1, round(numerator / denominator / 2))
    proc = subprocess.run([
        FFMPEG, '-hide_banner', '-nostdin', '-n', '-threads', '1', '-i', str(path),
        '-vf', f'select=not(mod(n\\,{stride})),scale=320:-2,showinfo',
        '-vsync', 'vfr', '-threads', '1', str(folder / 'frame_%03d.jpg'),
    ], capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:])
    timestamps = [float(t) for t in re.findall(r'\bn:\s*\d+.*?\bpts_time:([\d.]+)', proc.stderr)]
    frames = sorted(folder.glob('frame_*.jpg'))
    if not frames or len(timestamps) != len(frames):
        raise RuntimeError('Frame/timestamp identity failed')
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 17)
    contacts = []
    for offset in range(0, len(frames), 16):
        part = frames[offset:offset+16]
        sheet = Image.new('RGB', (1280, 42 + 212 * math.ceil(len(part)/4)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'Perception Test {video} | unedited source PTS | stride {stride}', fill='black', font=font)
        for index, source_frame in enumerate(part):
            x, y = index % 4 * 320, 42 + index // 4 * 212
            draw.text((x+5, y), f'{timestamps[offset+index]:.3f}s', fill='black', font=font)
            with Image.open(source_frame) as im:
                sheet.paste(im, (x, y+26))
        out = folder / f'contact_{offset//16+1:02d}.jpg'
        sheet.save(out, quality=85)
        contacts.append(out.name)
    record = {'official_dataset': 'Perception Test', 'video_id': video, 'source_url': url,
              'license': 'CC-BY-4.0, per official google-deepmind/perception_test README',
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': size,
              'download_md5_checked': expected_md5, 'source_reordered_or_reencoded': False,
              'generated_target': False, 'vlm_or_npu_called': False,
              'source_stride': stride, 'sampled_pts_seconds': timestamps,
              'contact_sheets': contacts, 'probe': probe}
    (folder / 'metadata.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'video': video, 'source': str(path), 'size_bytes': size,
                      'duration': probe['format']['duration'], 'contacts': contacts}), flush=True)


if __name__ == '__main__':
    if str(ROOT) != '/cache/zhonghao/temporal_source_review_20260913/search_round_20260913_2120':
        raise RuntimeError('Only the isolated personal CPU review directory is allowed')
    if sum(size for size, _ in EXPECTED.values()) > MAX_MEDIA_BYTES:
        raise RuntimeError('Media budget exceeded')
    os.nice(19)
    for video, (size, md5) in EXPECTED.items():
        review(video, size, md5)
