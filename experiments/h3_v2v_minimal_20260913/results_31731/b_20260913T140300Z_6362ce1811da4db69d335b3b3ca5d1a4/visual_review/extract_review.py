"""Local CPU-only source/A8/B8 five-frame comparison; inputs are read-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN_B = HERE.parent
RUN_A = RUN_B.parent / "a_20260913T133502Z_294205bef2e043dc84b41315e544b36d"
INDICES = (0, 30, 60, 90, 123)
SOURCES = {
    "source": (RUN_A / "source.mp4", "e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9"),
    "A8": (RUN_A / "a_50step.mp4", "1df518c92da95f9f926d1a82284199a6017660a46a502502c08b12219c52c9a8"),
    "B8": (RUN_B / "b_50step.mp4", "ca8aa801cb1c96026b237f4f1dbb93d8e4846071338aae3796ae77c24537d2fa"),
}


def main():
    if HERE.name != "visual_review" or RUN_B.name != "b_20260913T140300Z_6362ce1811da4db69d335b3b3ca5d1a4":
        raise RuntimeError("Unexpected private review directory")
    expected_outputs = [HERE / f"{kind}_frame_{i:03d}.png" for kind in SOURCES for i in INDICES]
    expected_outputs += [HERE / "source_A8_B8_contact.jpg", HERE / "extraction_record.json"]
    if any(path.exists() for path in expected_outputs):
        raise RuntimeError("Refusing to overwrite earlier review artifacts")
    # Verify all complete inputs before any decoding, especially the new B download.
    for kind, (path, expected_sha) in SOURCES.items():
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError(f"Input absent, incomplete, or identity-mismatched: {kind}")
    metadata, images = {}, {}
    for kind, (path, expected_sha) in SOURCES.items():
        samples = []
        with av.open(str(path)) as container:
            track = container.streams.video[0]
            track.codec_context.thread_count = 1
            record = dict(path=str(path), sha256=expected_sha, bytes=path.stat().st_size,
                          width=track.width, height=track.height, average_rate=str(track.average_rate),
                          time_base=str(track.time_base), declared_frames=track.frames,
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
            if (count != 124 or track.average_rate != 24 or (track.width, track.height) != (1344, 768)
                    or [item["frame_index"] for item in samples] != list(INDICES)):
                raise RuntimeError(f"Unexpected input format or sample count: {kind}")
            metadata[kind] = dict(record, decoded_frame_count=count, samples=samples)
    width, height, header, label = 448, 256, 37, 29
    sheet = Image.new("RGB", (3 * width, header + len(INDICES) * (height + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    titles = {"source": "SOURCE", "A8": "A8 dense Ref2VA | 50 steps", "B8": "B8 VDN + global source | 50 steps"}
    for col, kind in enumerate(SOURCES):
        draw.text((col * width + 8, 8), titles[kind], fill="black", font=font)
    for row, index in enumerate(INDICES):
        y = header + row * (height + label)
        for col, kind in enumerate(SOURCES):
            sample = next(item for item in metadata[kind]["samples"] if item["frame_index"] == index)
            draw.text((col * width + 8, y + 3), f"frame {index} | PTS {sample['seconds']:.3f} s", fill="black", font=font)
            sheet.paste(images[kind, index].resize((width, height), Image.Resampling.LANCZOS), (col * width, y + label))
    contact = HERE / "source_A8_B8_contact.jpg"
    sheet.save(contact, quality=93)
    report = dict(method="CPU PyAV full decode; five corresponding-index PNGs per source; PIL contact; no continuous playback",
                  videos=metadata, indices=list(INDICES), contact=str(contact),
                  inputs_modified=False, remote_operations=False, new_model_called=False,
                  limits=["Selected frames do not establish absence of inter-frame flicker or exact temporal/camera correspondence.",
                          "No audio review, VLM/perceptual metric, latency/NFE, or memory measurement here."])
    with (HERE / "extraction_record.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"contact": str(contact), "all_full_sha_verified": True,
                      "decoded_frames": {kind: row["decoded_frame_count"] for kind, row in metadata.items()},
                      "sample_pts": {kind: [item["seconds"] for item in row["samples"]] for kind, row in metadata.items()}}))


if __name__ == "__main__":
    main()
