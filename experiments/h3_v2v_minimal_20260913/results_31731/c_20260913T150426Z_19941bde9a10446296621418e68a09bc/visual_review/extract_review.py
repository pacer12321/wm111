"""Local CPU-only source/A8/B8/C8 review; run only after C download approval.

All video inputs and the formal C result record are read-only. This script
validates every full decode before creating any review outputs, and refuses
to overwrite outputs from an earlier invocation. No remote/model operations.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import av
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN_C_NAME = "c_20260913T150426Z_19941bde9a10446296621418e68a09bc"
RESULTS = Path("C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/experiments/h3_v2v_minimal_20260913/results_31731")
RUN_C = RESULTS / RUN_C_NAME
RUN_A = RESULTS / "a_20260913T133502Z_294205bef2e043dc84b41315e544b36d"
RUN_B = RESULTS / "b_20260913T140300Z_6362ce1811da4db69d335b3b3ca5d1a4"
C_REMOTE_OUTPUT = f"/cache/zhonghao/h3/c_model_trial/01234567/lake_snow/C/runs/{RUN_C_NAME}/output/c_50step.mp4"
INDICES = (0, 30, 60, 90, 123)
KINDS = ("source", "A8", "B8", "C8")
PINNED_INPUTS = {
    "source": (RUN_A / "source.mp4", "e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9"),
    "A8": (RUN_A / "a_50step.mp4", "1df518c92da95f9f926d1a82284199a6017660a46a502502c08b12219c52c9a8"),
    "B8": (RUN_B / "b_50step.mp4", "ca8aa801cb1c96026b237f4f1dbb93d8e4846071338aae3796ae77c24537d2fa"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular_private_file(path):
    if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError(f"Expected a regular non-symlink input: {path}")
    return path


def load_c_result():
    path = regular_private_file(RUN_C / "result.json")
    raw = path.read_bytes()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("Formal C result must be a JSON object")
    if (result.get("success") is not True
            or type(result.get("requested_steps")) is not int
            or result["requested_steps"] != 50
            or result.get("http_code") != "200"
            or type(result.get("curl_returncode")) is not int
            or result["curl_returncode"] != 0
            or not isinstance(result.get("content_type"), str)
            or result["content_type"].split(";")[0].strip().lower() != "video/mp4"
            or result.get("output_video") != C_REMOTE_OUTPUT):
        raise RuntimeError("C result is not this run's successful formal 50-step output")
    sha = result.get("output_sha256")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise RuntimeError("C result is missing a complete lowercase SHA256")
    probe = result.get("ffprobe", {})
    checks = probe.get("checks", {}) if isinstance(probe, dict) else {}
    if (not isinstance(probe, dict) or probe.get("verified") is not True
            or checks.get("dimensions") != [1344, 768]
            or checks.get("fps") != 24 or checks.get("frame_count") != 124):
        raise RuntimeError("Formal C result lacks the expected verified video specification")
    return path, raw, result, sha


def decode_input(kind, path, expected_sha):
    images, samples, all_pts = {}, [], []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise RuntimeError(f"Expected exactly one video stream: {kind}")
        track = container.streams.video[0]
        track.codec_context.thread_count = 1
        if (track.average_rate != 24 or (track.width, track.height) != (1344, 768)
                or track.frames != 124 or track.duration is None or track.time_base is None
                or track.duration * track.time_base != Fraction(124, 24)):
            raise RuntimeError(f"Unexpected declared video specification: {kind}")
        record = dict(path=str(path), sha256=expected_sha, bytes=path.stat().st_size,
                      width=track.width, height=track.height, average_rate=str(track.average_rate),
                      time_base=str(track.time_base), declared_frames=track.frames,
                      video_duration_seconds=float(track.duration * track.time_base),
                      has_audio=bool(container.streams.audio), audio_reviewed=False)
        count = 0
        for index, frame in enumerate(container.decode(track)):
            count += 1
            if (index >= 124 or (frame.width, frame.height) != (1344, 768)
                    or frame.pts is None or frame.time_base is None):
                raise RuntimeError(f"Unexpected decoded frame: {kind} index {index}")
            timestamp = frame.pts * frame.time_base
            if timestamp != Fraction(index, 24):
                raise RuntimeError(f"Noncanonical decoded PTS: {kind} index {index}: {timestamp}")
            frame_record = dict(frame_index=index, pts=frame.pts, time_base=str(frame.time_base),
                                seconds=float(timestamp), exact_seconds=str(timestamp))
            all_pts.append(frame_record)
            if index in INDICES:
                image = frame.to_image()
                if image.size != (1344, 768):
                    raise RuntimeError(f"PNG extraction changed native size: {kind} index {index}")
                images[index] = image
                samples.append(dict(frame_record, image=str(HERE / f"{kind}_frame_{index:03d}.png")))
        if count != 124 or tuple(images) != INDICES:
            raise RuntimeError(f"Unexpected full decode or sample count: {kind}")
        record.update(decoded_frame_count=count, samples=samples, all_decoded_pts=all_pts,
                      full_pts_verified="Every decoded frame has PTS exactly index/24 seconds")
    return record, images


def save_image_exclusive(image, path, image_format, **options):
    with path.open("xb") as stream:
        image.save(stream, format=image_format, **options)


def main():
    if (HERE != RUN_C / "visual_review" or HERE.resolve() != HERE.absolute()
            or RUN_C.name != RUN_C_NAME or HERE.is_symlink() or RUN_C.is_symlink()):
        raise RuntimeError("Unexpected private C review directory")
    contact = HERE / "source_A8_B8_C8_contact.jpg"
    record_path = HERE / "extraction_record.json"
    outputs = [HERE / f"{kind}_frame_{index:03d}.png" for kind in KINDS for index in INDICES]
    outputs += [contact, record_path]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("Refusing to overwrite earlier review artifacts")
    result_path, result_raw, c_result, c_sha = load_c_result()
    sources = dict(PINNED_INPUTS, C8=(RUN_C / "c_50step.mp4", c_sha))
    # All four complete identities are verified before any decode or PNG output.
    for kind, (path, sha) in sources.items():
        if digest(regular_private_file(path)) != sha:
            raise RuntimeError(f"Input incomplete or identity-mismatched: {kind}")
    c_size = c_result["ffprobe"]["metadata"]["format"]["size"]
    if str((RUN_C / "c_50step.mp4").stat().st_size) != str(c_size):
        raise RuntimeError("C input byte count differs from its formal result")
    metadata, images = {}, {}
    for kind, (path, sha) in sources.items():
        metadata[kind], images[kind] = decode_input(kind, path, sha)
    # No artifact is written until all decodes and PTS pass and inputs remain stable.
    for kind, (path, sha) in sources.items():
        if digest(regular_private_file(path)) != sha:
            raise RuntimeError(f"Input changed during decoding: {kind}")
    if regular_private_file(result_path).read_bytes() != result_raw:
        raise RuntimeError("C result record changed during decoding")

    width, height, header, label = 448, 256, 37, 29
    sheet = Image.new("RGB", (4 * width, header + len(INDICES) * (height + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    titles = {"source": "SOURCE", "A8": "A8 dense Ref2VA | 50 steps",
              "B8": "B8 VDN + global source | 50 steps", "C8": "C8 same-frame source | 50 steps"}
    for col, kind in enumerate(KINDS):
        draw.text((col * width + 8, 8), titles[kind], fill="black", font=font)
    for row, index in enumerate(INDICES):
        y = header + row * (height + label)
        for col, kind in enumerate(KINDS):
            sample = next(item for item in metadata[kind]["samples"] if item["frame_index"] == index)
            draw.text((col * width + 8, y + 3), f"frame {index} | PTS {sample['seconds']:.3f} s", fill="black", font=font)
            sheet.paste(images[kind][index].resize((width, height), Image.Resampling.LANCZOS), (col * width, y + label))
    for kind in KINDS:
        for index in INDICES:
            save_image_exclusive(images[kind][index], HERE / f"{kind}_frame_{index:03d}.png", "PNG")
    save_image_exclusive(sheet, contact, "JPEG", quality=93)
    report = dict(method="CPU PyAV full decode and every-frame PTS verification; five native PNGs per video; PIL contact; no continuous playback",
                  videos=metadata, indices=list(INDICES), contact=str(contact),
                  c_run=RUN_C_NAME, c_result_path=str(result_path),
                  c_result_sha256=hashlib.sha256(result_raw).hexdigest(),
                  c_formal_result=c_result, all_input_sha_verified_before_and_after_decode=True,
                  native_png_count=20, contact_columns=list(KINDS),
                  inputs_modified=False, remote_operations=False, new_model_called=False,
                  quality_evidence=False,
                  limits=["Extraction only: observations require a separate visual review.",
                          "Selected frames do not establish absence of inter-frame flicker or exact temporal/camera correspondence.",
                          "No audio review, VLM/perceptual metric, latency/NFE, or memory measurement here."])
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"contact": str(contact), "all_full_sha_verified": True,
                      "c_result_sha256": report["c_result_sha256"],
                      "decoded_frames": {kind: row["decoded_frame_count"] for kind, row in metadata.items()},
                      "sample_pts": {kind: [item["seconds"] for item in row["samples"]] for kind, row in metadata.items()}}))


if __name__ == "__main__":
    main()
