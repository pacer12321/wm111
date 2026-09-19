from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _infer_grid(count: int, frames: int | None, height: int | None, width: int | None):
    if frames and height and width:
        if frames * height * width != count:
            raise ValueError("The requested grid does not match the mask length")
        return frames, height, width
    if not frames:
        return None
    area = count // frames
    if area * frames != count:
        return None
    side = int(math.isqrt(area))
    if side * side == area:
        return frames, side, side
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--text-rows", type=int, default=6159)
    parser.add_argument("--source-rows", type=int, default=37296)
    parser.add_argument("--global-rows", type=int, default=81216)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()

    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    mask = payload["active_target_mask"].view(-1).to(torch.bool)
    target_rows = int(mask.numel())
    used_rows = args.text_rows + args.source_rows + target_rows
    if args.global_rows % args.world_size:
        raise ValueError("global rows must be divisible by world size")
    local_rows = args.global_rows // args.world_size

    global_active = torch.ones(args.global_rows, dtype=torch.bool)
    target_start = args.text_rows + args.source_rows
    global_active[target_start : target_start + target_rows] = mask

    shard_stats = []
    for rank in range(args.world_size):
        start = rank * local_rows
        end = start + local_rows
        shard = global_active[start:end]
        target_lo = max(start, target_start)
        target_hi = min(end, target_start + target_rows)
        active_target = (
            int(mask[target_lo - target_start : target_hi - target_start].sum())
            if target_hi > target_lo
            else 0
        )
        shard_stats.append(
            {
                "rank": rank,
                "global_range": [start, end],
                "rows": local_rows,
                "active_rows": int(shard.sum()),
                "stable_rows": int((~shard).sum()),
                "target_rows": max(0, target_hi - target_lo),
                "active_target_rows": active_target,
                "non_target_or_padding_active_rows": int(shard.sum()) - active_target,
            }
        )

    strided_stats = []
    all_indices = torch.arange(args.global_rows)
    for rank in range(args.world_size):
        indices = all_indices[rank::args.world_size]
        shard = global_active.index_select(0, indices)
        target_selector = (indices >= target_start) & (indices < target_start + target_rows)
        target_indices = indices[target_selector] - target_start
        active_target = int(mask.index_select(0, target_indices).sum())
        strided_stats.append(
            {
                "rank": rank,
                "rows": int(indices.numel()),
                "active_rows": int(shard.sum()),
                "stable_rows": int((~shard).sum()),
                "target_rows": int(target_selector.sum()),
                "active_target_rows": active_target,
                "non_target_or_padding_active_rows": int(shard.sum()) - active_target,
            }
        )

    result: dict[str, Any] = {
        "payload_metadata": _jsonable(payload),
        "layout": {
            "text_rows": args.text_rows,
            "source_rows": args.source_rows,
            "target_rows": target_rows,
            "used_rows": used_rows,
            "padding_rows": args.global_rows - used_rows,
            "global_rows": args.global_rows,
            "local_rows": local_rows,
            "packed_order": ["text", "source", "target", "padding"],
        },
        "target_mask": {
            "active": int(mask.sum()),
            "stable": int((~mask).sum()),
            "active_ratio": float(mask.float().mean()),
        },
        "contiguous_shards": shard_stats,
        "strided_even_odd_shards": strided_stats,
    }

    payload_grid = payload.get("token_grid")
    requested_grid = (args.frames, args.height, args.width)
    if not any(requested_grid) and isinstance(payload_grid, (list, tuple)) and len(payload_grid) == 3:
        requested_grid = tuple(int(item) for item in payload_grid)
    grid = _infer_grid(target_rows, *requested_grid)
    if grid:
        frames, height, width = grid
        cube = mask.view(frames, height, width)
        per_frame = cube.sum(dim=(1, 2))
        per_row = cube.sum(dim=(0, 2))
        per_col = cube.sum(dim=(0, 1))
        result["grid"] = {
            "shape": [frames, height, width],
            "active_per_frame": per_frame.tolist(),
            "active_ratio_per_frame": (per_frame.float() / (height * width)).tolist(),
            "active_per_spatial_row": per_row.tolist(),
            "active_per_spatial_col": per_col.tolist(),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
