"""One immutable, versioned source/target clip consumed by BOTH encoders.

This does not choose a dataset, fabricate long examples, or certify visual quality.
Native duration is retained up to H3's last complete VAE chunk; no temporal padding.
Spatial policy is explicit, never inferred separately by Qwen and VAE producers.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess

SCHEMA = "savie-shared-clip-v1"
FPS = 24


def audio_latent_count(frame_count, fps):
    # Exact formula in the active serving time_request.py (40 Hz, 2 channels).
    # Output audio loss is still disabled; this only defines structural rows.
    if frame_count <= 0 or fps <= 0:
        raise ValueError("positive frame count and fps required")
    return int(round(float(frame_count / fps) * 40.0))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_identity(row):
    return {k: row[k] for k in ("sample_id", "source_relpath", "target_relpath", "instruction")}


def clip_directory(root, sample_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError("sample_id must be a safe single path component")
    return Path(root) / sample_id


def aligned_h3_length(available_frames):
    # Verified checkpoint: clip=17, frame_overlap=5, token_overlap=2,
    # tokens_per_chunk=5, no isolated final frame. Trim, never pad.
    chunks = (int(available_frames) - 5) // 17
    if chunks < 1:
        raise ValueError("fewer than 22 frames; cannot encode without padding")
    return chunks * 17 + 5, chunks * 5 + 2


def probe_video(path):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration",
        "-of", "json", str(path)], text=True)
    data = json.loads(raw)
    s = data["streams"][0]
    fps = Fraction(s["avg_frame_rate"])
    frames = int(s["nb_read_frames"])
    duration = Fraction(s.get("duration") or data["format"]["duration"])
    if frames < 1 or fps <= 0 or duration <= 0:
        raise ValueError(f"invalid video {path}")
    return dict(width=int(s["width"]), height=int(s["height"]),
        frames=frames, fps=str(fps), duration=str(duration))


def make_spec(row, source_probe, target_probe, width, height, spatial_policy):
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("H3 output width/height must be positive multiples of 32")
    if spatial_policy not in ("center-crop", "resize"):
        raise ValueError("explicit supported spatial policy required")
    if (source_probe["width"], source_probe["height"]) != (
        target_probe["width"], target_probe["height"]
    ):
        raise ValueError("different source/target geometry needs explicit alignment review")
    durations = [Fraction(p["duration"]) for p in (source_probe, target_probe)]
    if abs(durations[0] - durations[1]) > Fraction(1, FPS):
        raise ValueError("source/target duration mismatch exceeds one output frame")
    available = int(min(durations) * FPS)
    frames, latent_frames = aligned_h3_length(available)
    return dict(schema=SCHEMA, identity=row_identity(row), fps=FPS,
        start_frame=0, frame_count=frames, latent_frames=latent_frames,
        width=width, height=height, spatial_policy=spatial_policy,
        source_probe=source_probe, target_probe=target_probe,
        available_frames=available, trimmed_tail_frames=available-frames,
        temporal_padding_frames=0, artificial_length_buckets=False,
        vae_contract=dict(clip_length=17, frame_overlap=5, token_overlap=2,
            tokens_chunk_size=5, isolated_last_frame=False))


def render_clip(src, dst, spec):
    w, h = spec["width"], spec["height"]
    spatial = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        if spec["spatial_policy"] == "center-crop" else f"scale={w}:{h}")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-threads", "2",
        "-i", str(src), "-map", "0:v:0", "-an", "-vf",
        f"setpts=PTS-STARTPTS,fps={FPS},{spatial},setsar=1",
        "-frames:v", str(spec["frame_count"]), "-c:v", "libx264", "-threads", "2",
        "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], check=True)
    p = probe_video(dst)
    if (p["frames"], p["width"], p["height"], Fraction(p["fps"])) != (
        spec["frame_count"], w, h, Fraction(FPS)
    ):
        raise ValueError(f"rendered clip violates contract: {p}")


def load_prepared_clip(root, row, *, verify_files=True):
    folder = clip_directory(root, row["sample_id"])
    spec = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
    claimed = spec.pop("clip_contract_sha256")
    if digest(spec) != claimed or (folder / "ready.done").read_text().strip() != claimed:
        raise ValueError("clip contract changed or preparation incomplete")
    spec["clip_contract_sha256"] = claimed
    if spec["schema"] != SCHEMA or spec["identity"] != row_identity(row):
        raise ValueError("clip belongs to a different sample/instruction or old schema")
    if verify_files:
        for role in ("source", "target"):
            if file_sha(folder / f"{role}.mp4") != spec[f"{role}_clip_sha256"]:
                raise ValueError(f"{role} prepared clip content changed")
    return folder, spec


def payload_metadata(spec):
    return dict(preprocessing_schema=SCHEMA,
        clip_contract_sha256=spec["clip_contract_sha256"],
        source_clip_sha256=spec["source_clip_sha256"],
        target_clip_sha256=spec["target_clip_sha256"],
        sample_id=spec["identity"]["sample_id"],
        instruction_sha256=digest(spec["identity"]["instruction"]),
        frame_count=spec["frame_count"], video_fps=spec["fps"],
        expected_latent_shape=[spec["latent_frames"], spec["height"]//16, spec["width"]//16])


def validate_payload(payload, expected):
    for key, value in expected.items():
        if key not in payload or payload[key] != value:
            raise ValueError(f"encoded payload mismatch: {key}")


def merge_encoded_pair(video, prompt):
    # Used by training before merging dictionaries, not after one identity has
    # overwritten the other. Legacy caches are deliberately rejected.
    keys = ("preprocessing_schema", "clip_contract_sha256", "source_clip_sha256",
        "target_clip_sha256", "sample_id", "instruction_sha256", "frame_count",
        "video_fps", "expected_latent_shape")
    if video.get("preprocessing_schema") != SCHEMA:
        raise ValueError("legacy video cache has no shared-clip contract; re-encode")
    validate_payload(prompt, {k: video[k] for k in keys})
    expected = tuple(video["expected_latent_shape"])
    for role in ("source", "target"):
        x = video[f"{role}_video_latents"]
        if tuple(x.shape) != (24,) + expected:
            raise ValueError(f"{role} latent shape disagrees with prepared clip")
    return dict(video, **{k: v for k, v in prompt.items() if k not in keys})


def prepare_pair(root, video_root, row, width, height, spatial_policy):
    folder = clip_directory(root, row["sample_id"])
    source, target = (Path(video_root) / row[f"{role}_relpath"] for role in ("source", "target"))
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError(f"missing pair {row['sample_id']}")
    spec = make_spec(row, probe_video(source), probe_video(target), width, height, spatial_policy)
    spec.update(source_original_sha256=file_sha(source), target_original_sha256=file_sha(target))
    if (folder / "ready.done").exists():
        _, old = load_prepared_clip(root, row)
        for key, value in spec.items():
            if old[key] != value:
                raise ValueError("existing clip has another policy/input; use a new output root")
        return old
    # Refuse silent overwrite of an interrupted/foreign prepared sample.
    folder.mkdir(parents=True, exist_ok=False)
    for role, original in (("source", source), ("target", target)):
        render_clip(original, folder / f"{role}.mp4", spec)
        spec[f"{role}_clip_sha256"] = file_sha(folder / f"{role}.mp4")
    spec["clip_contract_sha256"] = digest(spec)
    (folder / "clip.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "ready.done").write_text(spec["clip_contract_sha256"], encoding="ascii")
    return spec


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--spatial-policy", choices=["center-crop", "resize"], required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    failures = 0
    for row in rows[:args.limit]:
        try:
            spec = prepare_pair(args.output_root, args.video_root, row, args.width, args.height, args.spatial_policy)
            print(json.dumps(dict(sample_id=row["sample_id"], status="prepared", spec=spec)), flush=True)
        except Exception as exc:
            failures += 1
            print(json.dumps(dict(sample_id=row.get("sample_id"), status="rejected", error=str(exc))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
