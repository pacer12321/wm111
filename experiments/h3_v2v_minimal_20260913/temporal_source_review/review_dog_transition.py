"""Dense CPU-only inspection of the existing dog source between 0.5 and 1.7s."""
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path('/cache/zhonghao/temporal_source_review_20260913/reorder10_dog')
result = subprocess.run([
    '/usr/local/ffmpeg/bin/ffmpeg', '-hide_banner', '-nostdin', '-n', '-threads', '2',
    '-i', str(root / 'source.mp4'), '-vf',
    'select=between(n\\,12\\,40),scale=480:-2,showinfo', '-vsync', 'vfr', '-threads', '2',
    str(root / 'dense_%03d.png'),
], capture_output=True, text=True, timeout=60, check=True)
times = [float(t) for t in re.findall(r'\bn:\s*\d+.*?\bpts_time:([\d.]+)', result.stderr)]
frames = sorted(root.glob('dense_*.png'))
assert len(times) == len(frames)
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 20)
for start in range(0, len(frames), 12):
    part = frames[start:start + 12]
    sheet = Image.new('RGB', (1920, 45 + 300 * math.ceil(len(part) / 4)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), 'reorder10_dog | every original frame | source PTS | 24 fps', fill='black', font=font)
    for i, path in enumerate(part):
        x, y = (i % 4) * 480, 45 + (i // 4) * 300
        draw.text((x + 5, y + 2), f'{times[start+i]:.3f}s', fill='black', font=font)
        with Image.open(path) as frame:
            sheet.paste(frame, (x, y + 27))
    sheet.save(root / f'dense_contact_{start//12+1:02d}.jpg', quality=94)
print(f'Inspected transition sampling created: {len(frames)} original frames, {times[0]:.3f}–{times[-1]:.3f}s')
