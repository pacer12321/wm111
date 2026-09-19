"""Local CPU-only five-frame comparison; never modifies original videos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
INDICES = (0, 30, 60, 90, 123)
SOURCES = {
    "source": ("source.mp4", "e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9"),
    "A8": ("a_50step.mp4", "1df518c92da95f9f926d1a82284199a6017660a46a502502c08b12219c52c9a8"),
}


def main():
    if HERE.name != "visual_review" or RUN.name != "a_20260913T133502Z_294205bef2e043dc84b41315e544b36d":
        raise RuntimeError("Unexpected review directory")
    expected_outputs = [HERE / f"{kind}_frame_{i:03d}.png" for kind in SOURCES for i in INDICES]
    expected_outputs += [HERE / "source_vs_A8_contact.jpg", HERE / "extraction_record.json"]
    if any(path.exists() for path in expected_outputs):
        raise RuntimeError("Refusing to overwrite an earlier review artifact")
    metadata, images = {}, {}
    for kind, (name, expected_sha) in SOURCES.items():
        path = RUN / name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != expected_sha:
            raise RuntimeError(f"Original video identity mismatch: {kind}")
        samples = []
        with av.open(str(path)) as container:
            track = container.streams.video[0]
            track.codec_context.thread_count = 1
            record = dict(path=str(path), sha256=sha, bytes=path.stat().st_size,
                          width=track.width, height=track.height,
                          average_rate=str(track.average_rate), stream_time_base=str(track.time_base),
                          declared_frames=track.frames,
                          video_duration_seconds=float(track.duration * track.time_base),
                          has_audio=bool(container.streams.audio), audio_reviewed=False)
            count = 0
            for index, frame in enumerate(container.decode(track)):
                count += 1
                if index not in INDICES:
                    continue
                image = frame.to_image()
                output = HERE / f"{kind}_frame_{index:03d}.png"
                image.save(output)
                images[kind, index] = image
                samples.append(dict(frame_index=index, pts=frame.pts, time_base=str(frame.time_base),
                                    seconds=float(frame.pts * frame.time_base), image=str(output)))
            if count != 124 or [row["frame_index"] for row in samples] != list(INDICES):
                raise RuntimeError(f"Unexpected decoded frame count: {kind}={count}")
            metadata[kind] = dict(record, decoded_frame_count=count, samples=samples)
    width, height, header, label = 672, 384, 38, 31
    sheet = Image.new("RGB", (2 * width, header + len(INDICES) * (height + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    draw.text((10, 7), "SOURCE | same frame index", fill="black", font=font)
    draw.text((width + 10, 7), "A8 dense Ref2VA | 50 steps", fill="black", font=font)
    for row, index in enumerate(INDICES):
        y = header + row * (height + label)
        for col, kind in enumerate(SOURCES):
            sample = next(item for item in metadata[kind]["samples"] if item["frame_index"] == index)
            draw.text((col * width + 9, y + 4), f"frame {index} | PTS {sample['seconds']:.3f} s", fill="black", font=font)
            thumb = images[kind, index].resize((width, height), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (col * width, y + label))
    contact = HERE / "source_vs_A8_contact.jpg"
    sheet.save(contact, quality=92)
    report = dict(method="CPU PyAV full decode, selected-frame PNGs, side-by-side PIL contact; no continuous playback",
                  indices=list(INDICES), videos=metadata, contact=str(contact),
                  source_files_modified=False, new_model_called=False, remote_operations=False,
                  limitations=["Five selected frames do not establish absence of flicker between samples.",
                               "Equal frame index is not a measured optical-flow/camera correspondence.",
                               "No audio quality, edit metric, DiT timing, or NFE measurement here."])
    with (HERE / "extraction_record.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
