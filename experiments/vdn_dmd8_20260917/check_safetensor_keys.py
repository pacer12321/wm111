import os
import sys

from safetensors import safe_open

root = sys.argv[1]
for rel in (
    "linear_branch/model.safetensors",
    "adapters/default/adapter_model.safetensors",
    "adapters/turbo/adapter_model.safetensors",
):
    path = os.path.join(root, rel)
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        print(rel, "metadata=", handle.metadata(), "count=", len(keys))
        print(*keys[:8], sep="\n  ")
