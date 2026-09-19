import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
import re


ROOT = Path("/cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250")


def header(path: Path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


for label, path in (
    ("default", ROOT / "adapters/default/adapter_model.safetensors"),
    ("turbo", ROOT / "adapters/turbo/adapter_model.safetensors"),
):
    h = header(path)
    h.pop("__metadata__", None)
    print(f"=== {label} count={len(h)} ===")
    families = Counter()
    patterns = defaultdict(list)
    for key, info in sorted(h.items()):
        family = key
        for marker in ("transformer_blocks.", "token_refiner.", "norm_out."):
            if marker in key:
                family = marker.rstrip(".")
                break
        families[family] += 1
        pattern = re.sub(r"transformer_blocks\.\d+", "transformer_blocks.N", key)
        pattern = re.sub(r"refiner_blocks\.\d+", "refiner_blocks.N", pattern)
        pattern = pattern.replace("lora_A.default", "lora_X.default").replace("lora_B.default", "lora_X.default")
        pattern = pattern.replace("lora_A.turbo", "lora_X.turbo").replace("lora_B.turbo", "lora_X.turbo")
        patterns[pattern].append((key, info["shape"], info["dtype"]))
    print("families", dict(families))
    for pattern, entries in sorted(patterns.items()):
        shape_counts = Counter((tuple(shape), dtype) for _, shape, dtype in entries)
        print(pattern, "count=", len(entries), "shapes=", dict(shape_counts), "example=", entries[0][0])
