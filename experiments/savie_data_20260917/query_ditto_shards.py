#!/usr/bin/env python3
"""Query Ditto-1M archive shard sizes through a Hugging Face-compatible API."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


VIDEO_DIRS = (
    "global_freeform1",
    "global_freeform2",
    "global_freeform3",
    "global_style1",
    "global_style2",
    "local",
    "source",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--repo", default="QingyanBai/Ditto-1M")
    parser.add_argument("--output")
    args = parser.parse_args()

    groups = []
    for directory in VIDEO_DIRS:
        repo = urllib.parse.quote(args.repo, safe="/")
        url = (
            f"{args.endpoint}/api/datasets/{repo}/tree/main/videos/{directory}"
            "?recursive=true&expand=true"
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 SAViE-Ditto-audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            entries = json.load(response)
        files = [
            {"path": entry["path"], "size": int(entry.get("size", 0))}
            for entry in entries
            if entry.get("type") == "file"
        ]
        groups.append(
            {
                "directory": directory,
                "files": files,
                "file_count": len(files),
                "total_bytes": sum(item["size"] for item in files),
            }
        )

    result = {
        "repo": args.repo,
        "endpoint": args.endpoint,
        "total_bytes": sum(group["total_bytes"] for group in groups),
        "groups": groups,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
