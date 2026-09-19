#!/usr/bin/env python3
"""Measure one production-resolution H3 video-VAE encode."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time

import torch
from vllm_omni.diffusion.models.minimax_h3.reference_video import load_video_frames
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--base", required=True)
    parser.add_argument("--vae-subfolder", default="video_vae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="h3-precompute-") as tmp:
        prepared = f"{tmp}/prepared.mp4"
        start = time.perf_counter()
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
            "-map", "0:v:0", "-an", "-vf", "fps=24,scale=1344:768:flags=lanczos,setsar=1",
            "-frames:v", "124", "-c:v", "libx264", "-pix_fmt", "yuv420p", prepared,
        ], check=True)
        transcode_seconds = time.perf_counter() - start

        start = time.perf_counter()
        frames = load_video_frames(prepared)
        decode_seconds = time.perf_counter() - start

        start = time.perf_counter()
        vae = MiniMaxH3VideoVAE(
            f"{args.base}/{args.vae_subfolder}", device=torch.device(args.device)
        )
        vae.eval().requires_grad_(False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            if args.batch_size == 1:
                latents, latent_shape = vae.encode_video(frames)
                latent_shapes = [list(latent_shape)]
            else:
                encoded = vae.model.encode_videos(
                    [frames] * args.batch_size, use_fp16_latent=True
                )
                latents = encoded[0]
                latent_shapes = [list(item.shape) for item in encoded]
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - start
        print(json.dumps({
            "frames": list(frames.shape),
            "latents": list(latents.shape),
            "latent_shapes": latent_shapes,
            "batch_size": args.batch_size,
            "seconds_per_video": encode_seconds / args.batch_size,
            "transcode_seconds": transcode_seconds,
            "decode_seconds": decode_seconds,
            "vae_load_seconds": load_seconds,
            "vae_encode_seconds": encode_seconds,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }, indent=2))


if __name__ == "__main__":
    main()
