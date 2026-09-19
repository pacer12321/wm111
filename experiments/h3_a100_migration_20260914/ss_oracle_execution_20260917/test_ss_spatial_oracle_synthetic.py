from __future__ import annotations

import torch

from analyze_ss_spatial_oracle import analyze_layer


frames = 2
height = width = 7
source_count = frames * height * width
used_len = source_count + 10
source_positions = torch.arange(source_count)
source_coords = torch.tensor(
    [
        [frame, y, x]
        for frame in range(frames)
        for y in range(height)
        for x in range(width)
    ]
)
payload = {
    "layout": {
        "num_frames": frames,
        "frame_height": height,
        "frame_width": width,
        "used_len": used_len,
    },
    "source_positions": source_positions,
    "source_coords": source_coords,
    "q": torch.randn(2, 2, 4),
    "q_indices": torch.tensor([24, 73]),
    "k": torch.randn(used_len, 2, 4),
    "v": torch.randn(used_len, 2, 4),
    "softmax_scale": 0.5,
    "layer": 1,
    "step": 4,
    "spotedit_refresh": False,
    "active_target_ratio": 0.3,
}
result = analyze_layer(payload, torch.device("cpu"), window=5, random_seed=7)
assert result["source_keys_spatial_local"] == frames * 25
assert result["all_key_reduction_fraction"] > 0
assert result["spatial_local_relative_output_error"]["mean"] >= 0
print("SS_ORACLE_SYNTHETIC_OK", result["spatial_local_relative_output_error"]["mean"])
