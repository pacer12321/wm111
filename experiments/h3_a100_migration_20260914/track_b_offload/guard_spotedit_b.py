"""Bounded foreign-GPU guard for the isolated one-forward B experiment."""
from pathlib import Path

import guard_bd_gpus as guard

guard.CONTROL = Path("/cache/zhonghao/h3/spotedit_b_gpu_guard_20260915")
guard.RUN = Path("/cache/zhonghao/h3/spotedit_b_first_x0_20260915_v3")

if __name__ == "__main__":
    guard.main()
