"""Summarize cgroup memory traces without importing model dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    for raw in sys.argv[1:]:
        path = Path(raw)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            stat = row.get("stat", {})
            file_cache = max(0, int(stat.get("cache", 0)) - int(stat.get("shmem", 0)))
            row["derived_non_file"] = int(row.get("usage_in_bytes", 0)) - file_cache
        peak_usage = max(rows, key=lambda row: row.get("usage_in_bytes", 0))
        peak_non_file = max(rows, key=lambda row: row["derived_non_file"])
        print(json.dumps({
            "path": str(path),
            "samples": len(rows),
            "peak_usage": {
                "utc": peak_usage.get("utc"),
                "bytes": peak_usage.get("usage_in_bytes"),
                "stage": peak_usage.get("queue_stage"),
                "non_file": peak_usage["derived_non_file"],
            },
            "peak_non_file": {
                "utc": peak_non_file.get("utc"),
                "bytes": peak_non_file["derived_non_file"],
                "usage": peak_non_file.get("usage_in_bytes"),
                "stage": peak_non_file.get("queue_stage"),
            },
        }, indent=2))


if __name__ == "__main__":
    main()
