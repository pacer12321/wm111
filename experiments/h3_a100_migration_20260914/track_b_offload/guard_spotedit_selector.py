"""Bounded foreign-GPU guard for the SpotEdit-style selector capture run."""
from pathlib import Path

import guard_bd_gpus as guard

guard.CONTROL = Path("/cache/zhonghao/h3/spotedit_selector_gpu_guard_20260915_v1")
guard.RUN = Path("/cache/zhonghao/h3/spotedit_selector_capture_20260915_v1")

if __name__ == "__main__":
    guard.main()
