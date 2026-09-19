"""Create an explicitly labeled, silent synchronized source/D review derivative."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/experiments/h3_v2v_minimal_20260913/results_31731/a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd/source.mp4')
TARGET = HERE.parent / 'output/d_50step.mp4'
OUTPUT = HERE / 'source_vs_D_synchronized.mp4'
RECORD = HERE / 'side_by_side_record.json'
EXPECTED = {
    SOURCE: '4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2',
    TARGET: '5cb4b4e3f215e6f3cb0403a8a993e4977d40f17eaccbda876a2fba8d7bf4363b',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUTPUT.exists() or RECORD.exists():
        raise RuntimeError('Refusing to overwrite comparison artifacts')
    for path, expected in EXPECTED.items():
        if path.is_symlink() or digest(path) != expected:
            raise RuntimeError('Input hash mismatch')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    count = 0
    with av.open(str(SOURCE)) as left, av.open(str(TARGET)) as right, av.open(str(OUTPUT), 'w') as output:
        for container in (left, right):
            container.streams.video[0].codec_context.thread_count = 1
        track = output.add_stream('libx264', rate=24)
        track.width, track.height = 1344, 428
        track.pix_fmt = 'yuv420p'
        track.options = {'crf': '18', 'preset': 'fast'}
        track.codec_context.thread_count = 2
        for i, frames in enumerate(zip(left.decode(video=0), right.decode(video=0), strict=True)):
            if i >= 124 or any(frame.pts * frame.time_base != Fraction(i, 24) for frame in frames):
                raise RuntimeError('Frame count / synchronization mismatch')
            canvas = Image.new('RGB', (1344, 428), '#151515')
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 12), f'SOURCE (original) | {i / 24:.3f}s', fill='white', font=font)
            draw.text((682, 12), f'D (S-hybrid+same-frame) | shirt -> red | {i / 24:.3f}s', fill='white', font=font)
            for col, frame in enumerate(frames):
                fitted = ImageOps.contain(frame.to_image(), (672, 384), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (col * 672 + (672 - fitted.width) // 2, 44 + (384 - fitted.height) // 2))
            combined = av.VideoFrame.from_image(canvas)
            combined.pts, combined.time_base = i, Fraction(1, 24)
            for packet in track.encode(combined):
                output.mux(packet)
            count += 1
        if count != 124:
            raise RuntimeError('Incomplete source/output decode')
        for packet in track.encode():
            output.mux(packet)
    for path, expected in EXPECTED.items():
        if digest(path) != expected:
            raise RuntimeError('Input changed')
    verified = 0
    with av.open(str(OUTPUT)) as check:
        if check.streams.audio or check.streams.video[0].average_rate != 24:
            raise RuntimeError('Unexpected derivative streams')
        check.streams.video[0].codec_context.thread_count = 1
        for i, frame in enumerate(check.decode(video=0)):
            if frame.pts * frame.time_base != Fraction(i, 24) or (frame.width, frame.height) != (1344, 428):
                raise RuntimeError('Invalid derivative timing/dimensions')
            verified += 1
            if i == 60:
                with (HERE / 'source_vs_D_frame_060.png').open('xb') as stream:
                    frame.to_image().save(stream, format='PNG')
    if verified != 124:
        raise RuntimeError('Invalid derivative length')
    record = {'output': str(OUTPUT), 'sha256': digest(OUTPUT), 'bytes': OUTPUT.stat().st_size,
              'left_source_sha256': EXPECTED[SOURCE], 'right_D_sha256': EXPECTED[TARGET],
              'frames': verified, 'fps': 24, 'dimensions': [1344, 428],
              'synchronization': 'Same original frame index and verified i/24 PTS; no retiming or reversal',
              'resizing': 'Aspect-preserving fit into672x384 per half with black letterboxing; labels above',
              'silent': True, 'audio_note': 'Source has no audio; generated D audio deliberately omitted from visual comparison only',
              'is_original_model_output': False, 'new_model_call': False, 'inputs_unchanged': True}
    with RECORD.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
