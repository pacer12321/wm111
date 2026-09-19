"""Run one bounded, identity-verified cleanup sweep on the allocated A100 host."""
from pathlib import Path

import guard_bd_gpus as guard


def main() -> None:
    output = Path("/cache/zhonghao/h3/spotedit_selector_cleanup_20260915_v1")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.jsonl").open("a") as log:
        guard.sweep(log)


if __name__ == "__main__":
    main()
