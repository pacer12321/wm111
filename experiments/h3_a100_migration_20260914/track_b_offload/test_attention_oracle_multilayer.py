"""Small CPU contract test for the multilayer Oracle analyzer."""

from __future__ import annotations

import math

import torch

from analyze_attention_oracle_multilayer import analyze_one, b_vdn_allowed_keys


def main() -> None:
    torch.manual_seed(7)
    frames, height, width = 7, 1, 2
    per = height * width
    source_positions = torch.arange(frames * per)
    target_positions = torch.arange(frames * per, 2 * frames * per)
    used = int(2 * frames * per)
    coords = torch.tensor(
        [[frame, 0, x] for frame in range(frames) for x in range(width)],
        dtype=torch.float32,
    )
    heads, head_dim = 2, 4
    payload = {
        "meta": {
            "schema": "h3_dense_local_post_rope_qkv_v2",
            "layer": 24,
            "step": 0,
            "num_heads": heads,
            "head_dim": head_dim,
            "softmax_scale": head_dim**-0.5,
            "source_positions": source_positions,
            "source_coords": coords,
            "target_positions": target_positions,
            "target_coords": coords,
            "layout": {
                "used_len": used,
                "video_start": int(target_positions[0]),
                "num_frames": frames,
                "tokens_per_frame": per,
                "frame_height": height,
                "frame_width": width,
                "text_start": used,
                "text_len": 0,
            },
        },
        "query_indices": target_positions,
        "q": torch.randn(frames * per, heads, head_dim, dtype=torch.bfloat16),
        "k": torch.randn(used, heads, head_dim, dtype=torch.bfloat16),
        "v": torch.randn(used, heads, head_dim, dtype=torch.bfloat16),
    }
    layout = payload["meta"]["layout"]
    endpoint = b_vdn_allowed_keys(
        target_frame=0, frames=frames, per=per, layout=layout, device=torch.device("cpu")
    )
    interior = b_vdn_allowed_keys(
        target_frame=2, frames=frames, per=per, layout=layout, device=torch.device("cpu")
    )
    assert bool(endpoint.all())
    assert bool(interior[: frames * per].all())  # every source key remains visible in B
    for teacher_mode in ("dense", "b_vdn"):
        result = analyze_one(payload, [1, 2], torch.device("cpu"), teacher_mode)
        assert result["teacher_mode"] == teacher_mode
        assert result["structural_anchors"] == [0, 5, 6]
        assert result["sampled_queries_per_target_frame"] == per
        for size in (1, 2):
            row = result["oracle"][str(size)]
            assert row["min_unique_visible_source_frames"] >= 3
            assert row["max_unique_visible_source_frames"] <= 3 + 1 + size
            assert 0.0 <= row["retained_dense_source_attention_mass_fraction"] <= 1.0
            assert math.isfinite(row["attention_core_relative_l2"])
            assert math.isfinite(row["per_head_relative_l2"]["max"])
    print("oracle analyzer synthetic test: PASS")


if __name__ == "__main__":
    main()
