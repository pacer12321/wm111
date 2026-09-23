"""Training must run on the same MiniMax-H3 partition inference serves: Ref2VA.

Inference (vllm-omni) loads the native `MiniMax-H3/Ref2VA/transformer` shards and merges
DMD8 on top. Training loads a diffusers-layout base through `build_model(base_source=...)`,
and until 2026-09-23 that base was OpenVDN's `h3-base`, the diffusers export of the FL2VA
partition. SAViE LoRA learned against FL2VA weights was then merged onto Ref2VA weights.

`build_ref2va_base.py` converts the native Ref2VA transformer into the `h3-base` layout and
writes RECEIPT_NAME beside it. `verify_ref2va_base` is the cheap startup check (index plus
safetensors headers, no payload reads) that the trainer and the readiness gate run: a
directory without a matching Ref2VA receipt -- `h3-base` included -- is refused.
"""

import hashlib
import json
import struct
from pathlib import Path

RECEIPT_NAME = "ref2va_base_receipt.json"
SCHEMA = "savie-ref2va-base-v1"
PARTITION = "ref2va"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
REQUIRED_CHECKS = (
    "source_key_plan",
    "template_layout",
    "converted_layout",
    "independent_numeric_spot_check",
    "differs_from_template",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def fingerprint_transformer(transformer_dir):
    """Index digest plus, per shard, file size and safetensors-header digest.

    A rewrite of any tensor's name, shape, dtype or extent changes a header digest; the
    payload itself is covered once, at build time, by the receipt's numeric checks.
    """
    transformer_dir = Path(transformer_dir)
    index_path = transformer_dir / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = transformer_dir / name
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            raw = handle.read(size)
        if len(raw) != size:
            raise ValueError(f"{path}: truncated safetensors header")
        shards[name] = {"size_bytes": path.stat().st_size,
                        "header_sha256": hashlib.sha256(raw).hexdigest()}
    return {
        "index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(transformer_dir / "config.json"),
        "tensor_count": len(index["weight_map"]),
        "shards": shards,
    }


def verify_ref2va_base(base):
    """Return the receipt of a converted Ref2VA base, or raise with the reason it is not one."""
    base = Path(base).resolve()
    receipt_path = base / RECEIPT_NAME
    if not receipt_path.is_file():
        raise ValueError(
            f"{base} has no {RECEIPT_NAME}: training must use the Ref2VA base that inference "
            "serves, not OpenVDN's FL2VA h3-base. Build it with build_ref2va_base.py and pass "
            "that directory as --base."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError(f"{receipt_path}: schema {receipt.get('schema')!r} != {SCHEMA!r}")
    if receipt.get("partition") != PARTITION or Path(receipt.get("source", "")).name != "Ref2VA":
        raise ValueError(f"{receipt_path}: not converted from a MiniMax-H3 Ref2VA checkpoint")
    checks = receipt.get("checks", {})
    failed = [name for name in REQUIRED_CHECKS if checks.get(name, {}).get("passed") is not True]
    if failed:
        raise ValueError(f"{receipt_path}: build checks not passed: {failed}")
    current = fingerprint_transformer(base / "transformer")
    if current != receipt.get("fingerprint"):
        raise ValueError(f"{base}/transformer changed after conversion; rebuild the Ref2VA base")
    return receipt
