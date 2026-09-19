#!/usr/bin/env python3
"""Streaming producer for aligned H3 source/target video latents."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.models.minimax_h3.reference_video import load_video_frames
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from shared_clip_contract import (clip_directory, load_prepared_clip,
    payload_metadata, validate_payload)


def unpatchify(rows: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    latent_t, latent_h, latent_w = shape
    channels = rows.shape[-1] // 4
    ph, pw = latent_h // 2, latent_w // 2
    grid = rows.reshape(1, latent_t, ph, pw, channels, 1, 2, 2)
    return torch.einsum("nthwcrpq->nctrhpwq", grid).reshape(
        channels, latent_t, latent_h, latent_w
    )


def save_direct(path: Path, payload: dict) -> None:
    # /temp on this cluster forbids rename; completion is committed by a separate
    # .done marker only after torch.save has closed the final-path file.
    torch.save(payload, path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    Path(str(path).removesuffix(".pt") + ".done").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prepared-clip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=120)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    vae = MiniMaxH3VideoVAE(str(args.model / "video_vae"), device=device)
    vae.eval().requires_grad_(False)

    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid worker partition")
    for index in range(args.worker_index, len(rows), args.num_workers):
        row = rows[index]
        output = args.output_dir / f"video_{index:06d}.pt"
        done = args.output_dir / f"video_{index:06d}.done"
        waiting = time.monotonic()
        marker = clip_directory(args.prepared_clip_root, row["sample_id"]) / "ready.done"
        while not marker.is_file():
            if time.monotonic() - waiting > args.wait_timeout:
                break
            time.sleep(args.poll_seconds)
        if not marker.is_file():
            print(json.dumps(dict(index=index, sample_id=row["sample_id"],
                status="not_ready_skipped")), flush=True)
            continue
        folder, spec = load_prepared_clip(args.prepared_clip_root, row)
        metadata = payload_metadata(spec)
        if done.is_file():
            validate_payload(torch.load(output, map_location="cpu", weights_only=True), metadata)
            continue
        started = time.perf_counter()
        with torch.inference_mode():
            source_rows, source_shape = vae.encode_video(load_video_frames(str(folder / "source.mp4")))
            target_rows, target_shape = vae.encode_video(load_video_frames(str(folder / "target.mp4")))
        if source_shape != target_shape:
            raise RuntimeError(f"sample {index}: latent shape mismatch {source_shape} != {target_shape}")
        if tuple(source_shape) != tuple(metadata["expected_latent_shape"]):
            raise RuntimeError(f"VAE silently changed clip: {source_shape} != {metadata['expected_latent_shape']}")
        save_direct(output, {
            **metadata,
            "source_video_latents": unpatchify(source_rows, source_shape).to(torch.bfloat16),
            "target_video_latents": unpatchify(target_rows, target_shape).to(torch.bfloat16),
            "sample_id": row["sample_id"],
            "instruction": row["instruction"],
            "latent_shape": source_shape,
        })
        print(json.dumps({
            "index": index,
            "sample_id": row["sample_id"],
            "seconds": time.perf_counter() - started,
            "latent_shape": source_shape,
        }), flush=True)


if __name__ == "__main__":
    main()
