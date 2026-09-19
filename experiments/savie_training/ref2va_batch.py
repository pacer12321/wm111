"""Ref2VA latent packing for the trained dual-stream sparse attention.

The packed document is ``[text | source-video | target-audio | target-video]``.
Source and target video rows are both presented to the H3 transformer, but the loss is
computed only on target rows.  Source rows use the official Ref2VA condition timestep
0.999; target video/audio timesteps are sampled from the DMD8 inference grid.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

from src.models.sequence_layout import dual_layout_from_video_spans
from src.training.t2va_batch import (AUDIO_SHIFT, VIDEO_SHIFT, few_step_timesteps,
                                     sample_few_step_timesteps)


PATCH_SIZE = (1, 2, 2)
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
COND_TIMESTEP = 0.999
_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


def _axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    return torch.from_numpy(
        np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    ).to(torch.float64)


def _time_grid(length: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(length)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _time_span(length: int) -> float:
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(length))


def build_ref2va_layout(text_len: int, latent_t: int, latent_h: int, latent_w: int,
                        audio_t: int, device):
    """Build the exact one-source Ref2VA structural tensors without padding rows."""
    if latent_h % 2 or latent_w % 2:
        raise ValueError("latent spatial dimensions must be divisible by the H3 2x2 patch")
    ph, pw = latent_h // 2, latent_w // 2
    frame_rows = ph * pw
    source_rows = target_rows = latent_t * frame_rows
    audio_rows = audio_t * 2

    text_sl = slice(0, text_len)
    source_sl = slice(text_sl.stop, text_sl.stop + source_rows)
    audio_sl = slice(source_sl.stop, source_sl.stop + audio_rows)
    target_sl = slice(audio_sl.stop, audio_sl.stop + target_rows)
    seq_len = target_sl.stop

    text_indices = torch.arange(text_sl.start, text_sl.stop, dtype=torch.long)
    source_indices = torch.arange(source_sl.start, source_sl.stop, dtype=torch.long)
    target_indices = torch.arange(target_sl.start, target_sl.stop, dtype=torch.long)
    video_indices = torch.cat([source_indices, target_indices])
    audio_indices = torch.arange(audio_sl.start, audio_sl.stop, dtype=torch.long)

    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    position_ids[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    sqrt_area = math.sqrt(latent_h * latent_w)
    hh, ww = torch.meshgrid(
        _axis(latent_h, 2, sqrt_area), _axis(latent_w, 2, sqrt_area), indexing="ij"
    )
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    source_origin = float(text_len)
    source_grid = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    source_grid[:, :, 0] = _time_grid(latent_t, source_origin)[:, None]
    source_grid[:, :, 1:] = frame[None]
    position_ids[source_sl] = source_grid.reshape(-1, 3)

    target_origin = source_origin + _time_span(latent_t)
    audio_grid = target_origin + torch.arange(audio_t, dtype=torch.float64)
    position_ids[audio_sl, 0] = audio_grid.repeat(2)
    position_ids[audio_sl, 2] = torch.cat([
        torch.full((audio_t,), float(ww.min()), dtype=torch.float64),
        torch.full((audio_t,), float(ww.max()), dtype=torch.float64),
    ])
    target_grid = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    target_grid[:, :, 0] = _time_grid(latent_t, target_origin)[:, None]
    target_grid[:, :, 1:] = frame[None]
    position_ids[target_sl] = target_grid.reshape(-1, 3)

    token_tags = torch.full((seq_len,), VIDEO_TAG, dtype=torch.long)
    token_tags[text_sl] = TEXT_TAG
    token_tags[audio_sl] = AUDIO_TAG
    video_spans = (
        {"start": source_sl.start, "latent_grid": (latent_t, ph, pw), "role": "reference"},
        {"start": target_sl.start, "latent_grid": (latent_t, ph, pw), "role": "target"},
    )
    layout = dual_layout_from_video_spans(seq_len, video_spans, text_indices)
    return {
        "position_ids": position_ids.to(device),
        "token_tags": token_tags.to(device),
        "video_indices": video_indices.to(device),
        "audio_indices": audio_indices.to(device),
        "text_indices": text_indices.to(device),
        "layout": layout,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }


def _optimal_two_means_threshold(values: torch.Tensor) -> float:
    """Exact 1-D two-means SSE split used by the validated latent selector."""
    ordered = torch.sort(values.detach().float().reshape(-1).cpu()).values
    n = int(ordered.numel())
    if n < 2:
        raise ValueError("latent selector requires at least two token scores")
    prefix = ordered.cumsum(0).double()
    prefix_sq = ordered.square().cumsum(0).double()
    counts_left = torch.arange(1, n, dtype=torch.float64)
    counts_right = n - counts_left
    left_sum = prefix[:-1]
    left_sq = prefix_sq[:-1]
    right_sum = prefix[-1] - left_sum
    right_sq = prefix_sq[-1] - left_sq
    sse = (left_sq - left_sum.square() / counts_left
           + right_sq - right_sum.square() / counts_right)
    split = int(torch.argmin(sse).item()) + 1
    return (float(ordered[split - 1]) + float(ordered[split])) / 2.0


def unpatchify_video_rows(rows: torch.Tensor, shape: tuple[int, int, int, int, int]):
    """Inverse of H3's (1,2,2) video patchification for one packed video stream."""
    batch, channels, latent_t, latent_h, latent_w = shape
    if batch != 1 or latent_h % 2 or latent_w % 2:
        raise ValueError(f"unsupported H3 latent shape for unpatchify: {shape}")
    ph, pw = latent_h // 2, latent_w // 2
    expected = (latent_t * ph * pw, channels * 4)
    if tuple(rows.shape) != expected:
        raise ValueError(f"video rows {tuple(rows.shape)} != expected {expected}")
    grid = rows.reshape(batch, latent_t, ph, pw, channels, 1, 2, 2)
    return torch.einsum("nthwcrpq->nctrhpwq", grid).reshape(
        batch, channels, latent_t, latent_h, latent_w
    )


def latent_selector_mask(source: torch.Tensor, first_x0: torch.Tensor):
    """The exact selector used by the fastest validated DMD8+latent-skip run.

    It compares the *first DMD8 forward's x0 prediction* with the clean normalised
    source latent.  It never reads the clean training target.  Scores are computed per
    latent position after channel normalisation, pooled onto the DiT 2x2 token grid,
    split with exact two-means, then spatially dilated by radius one.
    """
    if source.shape != first_x0.shape or source.ndim != 5 or source.shape[0] != 1:
        raise ValueError(
            f"selector expects equal [1,C,T,H,W] source/x0, got "
            f"{tuple(source.shape)} and {tuple(first_x0.shape)}"
        )
    source_unit = F.normalize(source.float(), dim=1, eps=1e-6)
    x0_unit = F.normalize(first_x0.float(), dim=1, eps=1e-6)
    latent_score = (x0_unit - source_unit).square().sum(dim=1, keepdim=True)
    token_score = F.avg_pool3d(latent_score, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    threshold = _optimal_two_means_threshold(token_score)
    active = token_score >= threshold
    active_dilated = F.max_pool3d(
        active.float(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    ).bool()
    return active_dilated.reshape(-1), token_score.reshape(-1), threshold


def pack_ref2va_batch(sample, device, noise_generator, timestep_generator,
                      num_steps: int = 8, *, step_index: int | None = None,
                      noise_bundle: dict | None = None):
    source = sample["source_video_latents"].to(device, torch.float32)[None]
    target = sample["target_video_latents"].to(device, torch.float32)[None]
    if source.shape != target.shape:
        raise ValueError(f"source/target latent shapes differ: {source.shape} vs {target.shape}")
    audio = sample["target_audio_latents"].to(device, torch.float32)
    prompt = sample["prompt_embeds"].to(device, torch.bfloat16)[None]
    text_tags = sample["text_token_tags"]
    # Qwen context contains text AND visual tokens. Preserve the encoder's
    # AdaLN modality tags exactly, matching serving; location is not modality.
    if prompt.ndim != 3 or prompt.shape[0] != 1:
        raise ValueError("prompt_embeds must have shape [tokens, hidden]")
    if not isinstance(text_tags, torch.Tensor) or text_tags.ndim != 1:
        raise ValueError("text_token_tags must be a 1-D integer tensor")
    if text_tags.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("text_token_tags must have integer dtype")
    if text_tags.numel() != prompt.shape[1] or not text_tags.numel():
        raise ValueError("text_token_tags must match the nonempty prompt token count")
    if not bool(((text_tags == VIDEO_TAG) | (text_tags == TEXT_TAG)).all()):
        raise ValueError("Qwen video-presentation tags must be VIDEO_TAG=0 or TEXT_TAG=1")
    _, channels, latent_t, latent_h, latent_w = source.shape
    if channels != 24:
        raise ValueError(f"H3 visual VAE requires 24 channels, got {channels}")
    if audio.ndim != 3 or audio.shape[:2] != (2, 32):
        raise ValueError(f"expected target audio latents (2,32,A), got {audio.shape}")

    structural = build_ref2va_layout(
        int(text_tags.numel()), latent_t, latent_h, latent_w, int(audio.shape[-1]), device
    )
    structural["token_tags"].index_copy_(
        0, structural["text_indices"], text_tags.to(device=device, dtype=torch.long)
    )
    if step_index is None:
        t_v, t_a, step_index = sample_few_step_timesteps(
            timestep_generator, num_steps, VIDEO_SHIFT, AUDIO_SHIFT
        )
    else:
        video_grid, audio_grid = few_step_timesteps(num_steps, VIDEO_SHIFT, AUDIO_SHIFT)
        if not 0 <= step_index < num_steps:
            raise ValueError(f"step_index {step_index} outside [0,{num_steps})")
        t_v, t_a = float(video_grid[step_index]), float(audio_grid[step_index])
    if noise_bundle is None:
        target_noise = torch.randn(target.shape, generator=noise_generator,
                                   device=device, dtype=torch.float32)
        source_noise = torch.randn(source.shape, generator=noise_generator,
                                   device=device, dtype=torch.float32)
    else:
        target_noise = noise_bundle["target_noise"]
        source_noise = noise_bundle["source_noise"]
    audio_rows = audio.permute(0, 2, 1).reshape(-1, 32)
    audio_noise = (
        torch.randn(audio_rows.shape, generator=noise_generator,
                    device=device, dtype=torch.float32)
        if noise_bundle is None else noise_bundle["audio_noise"]
    )

    source_noisy = COND_TIMESTEP * source + (1.0 - COND_TIMESTEP) * source_noise
    target_noisy = t_v * target + (1.0 - t_v) * target_noise
    source_rows = patchify_video_latents(source_noisy, PATCH_SIZE)
    target_rows = patchify_video_latents(target_noisy, PATCH_SIZE)
    video_rows = torch.cat([source_rows, target_rows], dim=0)
    noisy_audio_rows = t_a * audio_rows + (1.0 - t_a) * audio_noise
    audio_policy = sample.get("audio_input_policy", "legacy-noisy")
    if audio_policy == "fixed-silent":
        # This option may only be enabled together with identical serving logic.
        # No target-audio objective is introduced.
        noisy_audio_rows = torch.zeros_like(audio_rows)
    elif audio_policy != "legacy-noisy":
        raise ValueError(f"unknown audio input policy: {audio_policy}")

    row_timesteps = torch.full(
        (structural["layout"].seq_len,), t_v, dtype=torch.float32, device=device
    )
    source_start = structural["layout"].source_start
    source_end = structural["layout"].source_end
    row_timesteps[source_start:source_end] = COND_TIMESTEP
    row_timesteps[structural["audio_indices"]] = t_a
    timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)

    inputs = dict(
        hidden_states=video_rows[None],
        audio_hidden_states=noisy_audio_rows[None],
        encoder_hidden_states=prompt,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=structural["token_tags"],
        position_ids=structural["position_ids"],
        video_indices=structural["video_indices"],
        audio_indices=structural["audio_indices"],
        text_indices=structural["text_indices"],
        return_dict=False,
    )
    return {
        "inputs": inputs,
        "layout": structural["layout"],
        "source_rows": structural["source_rows"],
        "target_video_rows": patchify_video_latents(target - target_noise, PATCH_SIZE),
        "target_audio_rows": audio_rows - audio_noise,
        "t_v": t_v,
        "t_a": t_a,
        "step_index": step_index,
        "source_clean_latents": source,
        "source_clean_rows": patchify_video_latents(source, PATCH_SIZE),
        "target_noisy_rows": target_rows,
        "noise_bundle": {
            "target_noise": target_noise,
            "source_noise": source_noise,
            "audio_noise": audio_noise,
        },
    }
