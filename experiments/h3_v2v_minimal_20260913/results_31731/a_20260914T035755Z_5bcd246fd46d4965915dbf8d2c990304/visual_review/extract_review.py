"""Read-only input decoding for the new shirt-color A trial; local CPU QA.

No model invocation. Import only verified media helpers from an older review;
never execute that review's main or reuse its old request/output identity.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parent
OLD = ROOT / 'a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd'
HELPER = OLD / 'visual_review/extract_review.py'
HELPER_SHA = 'b641e849c6aa903c7fb5494d32884fca7b2e2ecea4279ac5fc8673192a4e6fe5'
INDICES = (0, 15, 30, 45, 60, 75, 90, 105, 123)
SPECS = {
    'Source': (OLD / 'source.mp4', '4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2', 1527004, (1280, 720), False),
    'A_color': (RUN / 'a_50step.mp4', '392fd304bab4ca92f8a883f6c85c8a27f7adbd61dd5f1b5a6f678e1dee06d9ba', 5031092, (1344, 768), True),
}


def contact(indices, frames):
    width, height, header, label = 560, 320, 44, 30
    sheet = Image.new('RGB', (width * 2, header + len(indices) * (height + label)), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    for col, title in enumerate(('Source | unchanged original', 'A dense | only man shirt -> red')):
        draw.text((col * width + 8, 10), title, fill='black', font=font)
    for row, i in enumerate(indices):
        y = header + row * (height + label)
        for col, kind in enumerate(SPECS):
            draw.text((col * width + 8, y + 3), f'frame {i} | PTS {i / 24:.3f}s', fill='black', font=font)
            fitted = ImageOps.contain(frames[kind][i], (width, height), Image.Resampling.LANCZOS)
            draw.rectangle((col * width, y + label, (col + 1) * width - 1, y + label + height - 1), fill='black')
            sheet.paste(fitted, (col * width + (width - fitted.width) // 2, y + label + (height - fitted.height) // 2))
    return sheet


def main():
    if RUN.name != 'a_20260914T035755Z_5bcd246fd46d4965915dbf8d2c990304':
        raise RuntimeError('Wrong current run')
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Media helper changed')
    spec = importlib.util.spec_from_file_location('verified_media_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if helper.INDICES != INDICES:
        raise RuntimeError('Unexpected helper sampling')
    names = [f'{kind}_frame_{i:03d}.png' for kind in SPECS for i in INDICES]
    contacts = ['source_A_color_contact.jpg'] + [f'source_A_color_part_{n}.jpg' for n in (1, 2, 3)]
    outputs = names + contacts + ['extraction_record.json']
    if any((HERE / name).exists() or (HERE / name).is_symlink() for name in outputs):
        raise RuntimeError('Refusing to overwrite review artifacts')
    for path, sha, size, _, _ in SPECS.values():
        if helper.digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError(f'Input identity mismatch: {path}')
    records, frames = {}, {}
    for kind, values in SPECS.items():
        records[kind], frames[kind] = helper.decode(kind, values)
    for path, sha, size, _, _ in SPECS.values():
        if helper.digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError('Input changed during decode')
    for kind in SPECS:
        for i in INDICES:
            helper.save_image(frames[kind][i], HERE / f'{kind}_frame_{i:03d}.png', 'PNG')
    helper.save_image(contact(INDICES, frames), HERE / contacts[0], 'JPEG', quality=94)
    for part in range(3):
        helper.save_image(contact(INDICES[part * 3:part * 3 + 3], frames), HERE / contacts[part + 1], 'JPEG', quality=94)
    record = dict(run=RUN.name, sample_id='shirt_red_couple_124', requested_steps=50,
                  videos=records, sampled_indices=list(INDICES),
                  method='CPU full 124-frame decode and all PTS checks; nine corresponding-index native PNGs per video; aspect-preserving contacts',
                  helper_sha256=HELPER_SHA, contacts=contacts, native_png_count=18,
                  all_input_hashes_verified_before_and_after=True,
                  input_files_modified=False, model_calls=False, remote_operations=False,
                  continuous_playback=False, audio_reviewed=False,
                  limits=['Full decode is media integrity, not visual quality verification.',
                          'Sampled images cannot rule out between-frame flicker or establish exact timing/camera preservation.',
                          'Different source/output dimensions; contacts preserve aspect ratio without claiming pixel alignment.'])
    with (HERE / 'extraction_record.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({'run': RUN.name, 'decoded_frames': {kind: rec['decoded_frames'] for kind, rec in records.items()}, 'contacts': contacts}))


if __name__ == '__main__':
    main()
