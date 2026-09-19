#!/usr/bin/env python3
"""Precompute the exact Ref2VA Qwen video-presentation hidden states."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.presentation import (
    minimax_h3_ref2va_video_presentation,
)
from vllm_omni.diffusion.models.minimax_h3.reference_video import (
    sample_reference_video_frames,
)
from shared_clip_contract import (clip_directory, load_prepared_clip,
    payload_metadata, validate_payload)


class SingleRankEncoderGroup:
    world_size = 1
    ranks = [0]
    rank_in_group = 0
    device_group = None


def save_direct(path: Path, payload: dict) -> None:
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--wait-timeout", type=float, default=120)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(args.model), subfolder="tokenizer", local_files_only=True
    )
    processor = Qwen3VLProcessor.from_pretrained(
        str(args.model), subfolder="processor", local_files_only=True
    )
    encoder = MiniMaxH3Qwen3VLEncoder(
        str(args.model / "text_encoder"), device=device, load_model=True,
        encoder_group=SingleRankEncoderGroup(),
    )
    encoder.load_to_device()
    encoder.eval().requires_grad_(False)

    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid worker partition")
    for index in range(args.worker_index, len(rows), args.num_workers):
        row = rows[index]
        output = args.output_dir / f"prompt_{index:06d}.pt"
        done = args.output_dir / f"prompt_{index:06d}.done"
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
        source = folder / "source.mp4"
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"qwen-{index:06d}-") as tmp:
            sampled = sample_reference_video_frames(str(source), workdir=tmp)
            videos = [np.stack(sampled["frames"])]
            vision = processor.video_processor(
                videos=videos, do_sample_frames=False, return_tensors="pt"
            )
        video_grid = vision["video_grid_thw"]
        merge = int(processor.image_processor.merge_size) ** 2
        blocks = int(video_grid[0, 0])
        per_block = int(video_grid[0, 1]) * int(video_grid[0, 2]) // merge
        timestamps = sampled["block_timestamps"]
        if len(timestamps) != blocks:
            raise RuntimeError(
                f"sample {index}: video blocks {blocks} != timestamps {len(timestamps)}"
            )
        ids, tags = minimax_h3_ref2va_video_presentation(
            tokenizer,
            prompt=row["instruction"],
            condition_labels=[("video", 1)],
            image_token_count=None,
            video_block_token_counts=[[per_block] * blocks],
            video_block_timestamps=[timestamps],
        )
        with torch.inference_mode():
            hidden = encoder.encode_ids(
                ids,
                pixel_values_videos=vision["pixel_values_videos"],
                video_grid_thw=video_grid,
            )
        if hidden.shape[-2] != tags.numel() or not bool(((tags == 0) | (tags == 1)).all()):
            raise RuntimeError("Qwen hidden states and visual/text modality tags disagree")
        save_direct(output, {
            **metadata,
            "prompt_embeds": hidden.to(torch.bfloat16),
            "text_token_tags": tags.to(torch.long),
            "sample_id": row["sample_id"],
        })
        print(json.dumps({
            "index": index,
            "sample_id": row["sample_id"],
            "tokens": int(tags.numel()),
            "seconds": time.perf_counter() - started,
        }), flush=True)


if __name__ == "__main__":
    main()
