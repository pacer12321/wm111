"""Local CPU review of source order, reversed source stills, and formal A output.

Reversed source stills are an index-order diagnostic only, not generated GT or
a new video. Native PNGs retain each input's dimensions; contacts letterbox.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
RUN_NAME = "a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd"
RUN = Path("C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/experiments/h3_v2v_minimal_20260913/results_31731") / RUN_NAME
INDICES = (0, 15, 30, 45, 60, 75, 90, 105, 123)
SOURCE_SHA = "4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2"
A_SHA = "17b83ee2196302e9523e3e54aa239e3d6e7c8d5070c83d7c066e4f8cf06b38d7"
RESULT_SHA = "c15ab3b9319b9f1c3e40810848041a6419d8ed3222d0dbb3ba9e1d73e44e5d28"
PROMPT = "Reverse the temporal order of the entire reference clip: play all of its actions backward in time. Keep the same two people, their appearance, lighting, camera framing, and background."
SOURCES = {
    "source": (RUN / "source.mp4", SOURCE_SHA, 1527004, (1280, 720), False),
    "A8": (RUN / "a_50step.mp4", A_SHA, 4271598, (1344, 768), True),
}


def regular(path):
    if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError(f"Expected regular non-symlink input: {path}")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def png_name(kind, index):
    if kind == "source_reverse_reference":
        return f"source_reverse_reference_output_{index:03d}_source_{123-index:03d}.png"
    return f"{kind}_frame_{index:03d}.png"


def save_image(image, path, fmt, **options):
    with path.open("xb") as stream:
        image.save(stream, format=fmt, **options)


def decode(kind, spec):
    path, sha, size, dimensions, audio_expected = spec
    selected = set(INDICES)
    if kind == "source":
        selected.update(123 - i for i in INDICES)
    images, pts = {}, []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or bool(container.streams.audio) != audio_expected:
            raise RuntimeError(f"Unexpected video/audio streams: {kind}")
        track = container.streams.video[0]
        track.codec_context.thread_count = 1
        if (track.average_rate != 24 or track.frames != 124
                or (track.width, track.height) != dimensions
                or track.duration is None or track.time_base is None
                or track.duration * track.time_base != Fraction(124, 24)):
            raise RuntimeError(f"Unexpected stream specification: {kind}")
        record = dict(path=str(path), sha256=sha, bytes=size, dimensions=list(dimensions),
                      average_rate=str(track.average_rate), time_base=str(track.time_base),
                      declared_frames=track.frames, has_audio=audio_expected, audio_reviewed=False,
                      video_duration_seconds=float(track.duration * track.time_base))
        for i, frame in enumerate(container.decode(track)):
            if (i >= 124 or frame.pts is None or frame.time_base is None
                    or frame.pts * frame.time_base != Fraction(i, 24)
                    or (frame.width, frame.height) != dimensions):
                raise RuntimeError(f"Unexpected frame format or PTS: {kind} index {i}")
            pts.append(dict(frame_index=i, pts=frame.pts, time_base=str(frame.time_base),
                            seconds=float(frame.pts * frame.time_base)))
            if i in selected:
                images[i] = frame.to_image()
                if images[i].size != dimensions:
                    raise RuntimeError("Native PNG dimensions changed")
        if len(pts) != 124 or set(images) != selected:
            raise RuntimeError(f"Incomplete full decode: {kind}")
        record.update(decoded_frames=len(pts), all_decoded_pts=pts,
                      every_pts_equals_frame_index_divided_by_24=True)
    return record, images


def contact(indices, images):
    width, height, header, label = 448, 256, 61, 41
    sheet = Image.new("RGB", (3 * width, header + len(indices) * (height + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    titles = ("SOURCE forward", "SOURCE reversed-index reference", "A8 output | 50 requested steps")
    for col, title in enumerate(titles):
        draw.text((col * width + 7, 6), title, fill="black", font=font)
    draw.text((width + 7, 29), "Diagnostic stills only; NOT generated GT", fill="darkred", font=font)
    for row, index in enumerate(indices):
        y = header + row * (height + label)
        refs = (("source", index), ("source", 123 - index), ("A8", index))
        for col, (kind, source_index) in enumerate(refs):
            draw.text((col * width + 7, y + 3), f"row i={index} | source index={source_index}" if col == 1
                      else f"frame {source_index} | PTS {source_index/24:.3f}s", fill="black", font=font)
            if col == 1:
                draw.text((col * width + 7, y + 22), f"source PTS {source_index/24:.3f}s", fill="black", font=font)
            fitted = ImageOps.contain(images[kind][source_index], (width, height), Image.Resampling.LANCZOS)
            x_offset, y_offset = (width - fitted.width) // 2, (height - fitted.height) // 2
            draw.rectangle((col * width, y + label, (col + 1) * width - 1, y + label + height - 1), fill="black")
            sheet.paste(fitted, (col * width + x_offset, y + label + y_offset))
    return sheet


def main():
    if HERE != RUN / "visual_review" or HERE.is_symlink() or RUN.is_symlink():
        raise RuntimeError("Unexpected private review directory")
    kinds = ("source", "source_reverse_reference", "A8")
    contacts = ["source_forward_reversed_A8_contact.jpg"] + [f"source_forward_reversed_A8_contact_part_{n}.jpg" for n in (1, 2, 3)]
    outputs = [HERE / png_name(kind, i) for kind in kinds for i in INDICES]
    outputs += [HERE / name for name in contacts] + [HERE / "extraction_record.json"]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("Refusing to overwrite review artifacts")
    result_path, request_path = RUN / "result.json", RUN / "request.json"
    if digest(result_path) != RESULT_SHA:
        raise RuntimeError("Result identity mismatch")
    result_raw, request_raw = result_path.read_bytes(), regular(request_path).read_bytes()
    result, request = json.loads(result_raw), json.loads(request_raw)
    if (result.get("success") is not True or result.get("requested_steps") != 50
            or result.get("http_code") != "200" or result.get("curl_returncode") != 0
            or result.get("content_type") != "video/mp4" or result.get("output_sha256") != A_SHA
            or result.get("output_video") != f"/cache/zhonghao/h3/model_trial/01234567/explicit_reverse_couple_124/A/runs/{RUN_NAME}/output/a_50step.mp4"
            or result.get("ffprobe", {}).get("verified") is not True):
        raise RuntimeError("Unexpected formal result identity/specification")
    if (request.get("run_id") != "91c6c0f98ba84c7c828202d5934176cd"
            or request.get("sample_id") != "explicit_reverse_couple_124"
            or request.get("source_sha256") != SOURCE_SHA or request.get("requested_steps") != 50
            or request.get("fields", {}).get("prompt") != PROMPT):
        raise RuntimeError("Unexpected formal request identity/prompt")
    for kind, (path, sha, size, _, _) in SOURCES.items():
        if digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError(f"Input identity/size mismatch: {kind}")
    records, images = {}, {}
    for kind, spec in SOURCES.items():
        records[kind], images[kind] = decode(kind, spec)
    for kind, (path, sha, size, _, _) in SOURCES.items():
        if digest(path) != sha or path.stat().st_size != size:
            raise RuntimeError(f"Input changed during decode: {kind}")
    if result_path.read_bytes() != result_raw or request_path.read_bytes() != request_raw:
        raise RuntimeError("Request/result changed during decode")
    # Only after full validation do we create native PNGs and letterboxed contacts.
    for kind in kinds:
        for i in INDICES:
            input_kind, input_index = ("source", 123-i) if kind == "source_reverse_reference" else (kind, i)
            save_image(images[input_kind][input_index], HERE / png_name(kind, i), "PNG")
    save_image(contact(INDICES, images), HERE / contacts[0], "JPEG", quality=94)
    for part in range(3):
        save_image(contact(INDICES[part*3:part*3+3], images), HERE / contacts[part+1], "JPEG", quality=94)
    record = dict(run=RUN_NAME, videos=records, output_row_indices=list(INDICES),
                  reversed_source_indices=[123-i for i in INDICES],
                  prompt=PROMPT, result_sha256=RESULT_SHA, request_sha256=hashlib.sha256(request_raw).hexdigest(),
                  contacts=[str(HERE / name) for name in contacts], native_png_count=27,
                  source_reverse_reference_is_generated_ground_truth=False,
                  reverse_reference_is_new_video=False,
                  native_aspect_ratios_preserved=True, contact_resizing="Aspect-preserving contain with black letterbox",
                  all_input_hashes_checked_before_and_after_decode=True,
                  new_model_calls=False, remote_operations=False, input_files_modified=False,
                  continuous_playback=False, audio_reviewed=False,
                  limits=["Reverse source stills are a diagnostic index-order reference, not generated GT.",
                          "No claim of pixelwise or camera alignment between different-sized input/output videos.",
                          "Full decoding and matching PTS do not establish visual quality or editing success.",
                          "Only selected frames will be visually inspected; no continuous playback or audio review here."])
    with (HERE / "extraction_record.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"run": RUN_NAME, "all_sha_verified": True,
                      "decoded_frames": {kind: value["decoded_frames"] for kind, value in records.items()},
                      "contacts": record["contacts"]}))


if __name__ == "__main__":
    main()
