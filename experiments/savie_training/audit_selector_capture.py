#!/usr/bin/env python3
"""Audit a saved latent-selector capture without rerunning inference."""

from __future__ import annotations

import argparse
import json

import torch


def summarize(value):
    if torch.is_tensor(value):
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        if value.numel():
            numeric = value.float()
            result.update({
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "nonzero": int(torch.count_nonzero(value)),
            })
        return result
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    print(json.dumps(summarize(payload), indent=2, default=str))


if __name__ == "__main__":
    main()
