# SPDX-License-Identifier: Apache-2.0
"""Strict, opt-in Stage-B loading for the isolated Ref2VA experiment.

The shared deployment and checkpoint files are never modified.  The original H3
loader first converts its per-head [q,k,v] checkpoint rows into [all Q,all K,all V].
This module merges diffusers-named LoRAs into those *already converted* weights.

Scale provenance, checked 2026-09-13 at OpenVDN commit
2f740c9291431d89d4f2330743b093fac4390d09:
  configs/training/stage_b_c1_vdn_anchor.yaml: rank=64, alpha=64.
  src/inference/utils/assemble.py: merge_lora_state(model, stage_b_loras).
  src/inference/utils/lora.py: scale=1, fp32 B@A, cast delta then add to W.
This is specifically the released Stage-B artifact, not an external/turbo LoRA.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

OFFICIAL_COMMIT = "2f740c9291431d89d4f2330743b093fac4390d09"
STAGE_B_RANK = 64
STAGE_B_ALPHA = 64
STAGE_B_SCALE = STAGE_B_ALPHA / STAGE_B_RANK
NUM_BLOCKS = 50
NUM_REFINERS = 2
BASE_TENSOR_COUNT = 535
BRANCH_TENSOR_COUNT = 800
LORA_PAIR_COUNT = 208
BRANCH_SUFFIXES = (
    "linear_attention.alpha.A_log",
    "linear_attention.alpha.down.weight",
    "linear_attention.alpha.dt_bias",
    "linear_attention.alpha.up.weight",
    "linear_attention.beta_proj.weight",
    "linear_attention.norm.weight",
    "linear_attention.output_gate.down.weight",
    "linear_attention.output_gate.up.bias",
    "linear_attention.output_gate.up.weight",
    "linear_attention.short_conv.k_sp.weight",
    "linear_attention.short_conv.k_tm.weight",
    "linear_attention.short_conv.v_sp.weight",
    "linear_attention.short_conv.v_tm.weight",
    "softmax_gate.up.bias",
    "softmax_gate.up.weight",
    "to_out_linear.weight",
)


@contextmanager
def openvdn_cpu_loading_scope():
    """Limit only B's CPU loading intraop pool; restore it even after failure.

    Interop threads are observed, never changed.  This scope must finish before
    the offloader is enabled or any request is served.  It is process-local;
    it does not alter another worker or the already-running experiment.
    """
    original_threads = torch.get_num_threads()
    record = {
        "intraop_before": original_threads,
        "interop_before": torch.get_num_interop_threads(),
        "requested_loading_intraop": 4,
        "phase_seconds": {},
        "status": "starting",
    }
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        record["intraop_loading"] = torch.get_num_threads()
        record["interop_loading"] = torch.get_num_interop_threads()
        if record["intraop_loading"] != 4:
            raise RuntimeError("PyTorch did not apply B's four-thread CPU loading scope")
        record["status"] = "loading"
        logger.info("OPENVDN_B_CPU_LOADING_BEGIN %s", json.dumps(record, sort_keys=True))
        yield record
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["exception_type"] = type(error).__name__
        raise
    finally:
        # Never call set_num_interop_threads: unlike intraop it is not a
        # reversible per-scope setting once parallel work has started.
        torch.set_num_threads(original_threads)
        record["intraop_after"] = torch.get_num_threads()
        record["interop_after"] = torch.get_num_interop_threads()
        record["scope_seconds"] = time.monotonic() - started
        logger.info("OPENVDN_B_CPU_LOADING_END %s", json.dumps(record, sort_keys=True))


def configure_openvdn_from_env(od_config: Any, *, partition: str):
    """Return the untouched A config, or an explicitly enabled isolated B copy."""
    enabled = os.environ.get("ZHONGHAO_H3_OPENVDN", "0")
    if enabled not in ("0", "1"):
        raise ValueError("ZHONGHAO_H3_OPENVDN must be exactly 0 or 1")
    if enabled == "0":
        return od_config, None
    if partition.lower() != "ref2va":
        raise ValueError("This Stage-B transfer experiment requires the Ref2VA base")
    if getattr(od_config, "quantization_config", None) is not None:
        raise ValueError("The validated Stage-B merge requires unquantized BF16/F32 weights")
    if int(od_config.parallel_config.tensor_parallel_size) != 1:
        raise ValueError("Stage-B weight merge supports DiT TP=1 (Ulysses SP is separate)")
    if getattr(od_config.parallel_config, "use_hsdp", False):
        raise ValueError("Stage-B loading has not been validated with HSDP")
    if getattr(od_config, "enable_distributed_layerwise_offload", False):
        raise ValueError("DLO/mmap can bypass this loader; use ordinary layerwise offload")
    if not getattr(od_config, "enable_layerwise_offload", False):
        raise ValueError("This 64-GiB Ascend experiment requires ordinary layerwise offload")
    checkpoint_env = os.environ.get("ZHONGHAO_H3_OPENVDN_CHECKPOINT")
    if not checkpoint_env:
        raise ValueError("Set ZHONGHAO_H3_OPENVDN_CHECKPOINT to the full Stage-B directory")
    checkpoint = Path(checkpoint_env).expanduser().resolve(strict=True)
    if not checkpoint.is_dir() or checkpoint.name != "stage-b-step-2000":
        raise ValueError("Only the complete stage-b-step-2000 artifact is supported")
    for rel in ("linear_branch/model.safetensors", "adapters/default/adapter_model.safetensors"):
        if not (checkpoint / rel).is_file():
            raise FileNotFoundError(checkpoint / rel)

    # Do not mutate shared config objects, including the pipeline's original one.
    from vllm_omni.diffusion.data import TransformerConfig

    config = copy.copy(od_config)
    original = od_config.tf_model_config
    mapping = original.to_dict() if hasattr(original, "to_dict") else dict(original)
    mapping = copy.deepcopy(mapping)
    if int(mapping.get("num_layers", 50)) != NUM_BLOCKS:
        raise ValueError("Truncated five-block configurations are not a full B model")
    mapping["openvdn_enabled"] = True
    config.tf_model_config = TransformerConfig.from_dict(mapping)
    logger.info("OPENVDN_B_CONFIG checkpoint=%s base_partition=%s TP=1", checkpoint, partition)
    return config, checkpoint


def branch_name_map() -> dict[str, str]:
    return {
        f"transformer_blocks.{index}.attn.{suffix}": f"blocks.{index}.attn.{suffix}"
        for index in range(NUM_BLOCKS)
        for suffix in BRANCH_SUFFIXES
    }


def _assert_same_keys(actual, expected, *, label: str) -> None:
    actual, expected = set(actual), set(expected)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{label}: missing={missing}, unexpected={unexpected}")


def _targets(model) -> dict[str, torch.Tensor]:
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    return targets


def _check_cpu_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name}: merge must happen on CPU before offload hooks, got {tensor.device}")
    if tensor.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"{name}: unsupported dtype {tensor.dtype}")


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    """Bound temporary validation memory even for a full, multi-GB MLP tensor."""
    _check_cpu_tensor(tensor, name)
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: expected contiguous checkpoint/parameter storage")
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), 1 << 20):
        if not bool(torch.isfinite(flat[start : start + (1 << 20)]).all()):
            raise ValueError(f"{name}: non-finite values in elements [{start}:{start + (1 << 20)}]")


def strict_base_weights(
    model, weights: Iterable[tuple[str, torch.Tensor]], *, prefix: str = "transformer."
):
    """Validate every original base key and tensor before the existing H3 loader."""
    targets = _targets(model)
    branch_targets = set(branch_name_map().values())
    if len(model.blocks) != NUM_BLOCKS or len(model.token_refiner.blocks) != NUM_REFINERS:
        raise ValueError("Expected all 50 DiT blocks and both token-refiner blocks")
    if not bool(model.arch.openvdn_enabled):
        raise ValueError("Complete Stage-B loading requires openvdn_enabled=True at construction")
    _assert_same_keys(branch_targets & targets.keys(), branch_targets, label="branch module construction")
    expected = targets.keys() - branch_targets
    if len(expected) != BASE_TENSOR_COUNT:
        raise ValueError(f"Expected {BASE_TENSOR_COUNT} base parameters/buffers, got {len(expected)}")
    seen = set()
    started = time.monotonic()
    for full_name, tensor in weights:
        if not full_name.startswith(prefix):
            raise ValueError(f"Unexpected non-transformer checkpoint key {full_name!r}")
        name = full_name[len(prefix) :]
        if name not in expected or name in seen:
            raise ValueError(f"Unexpected or duplicate base key {name!r}")
        target = targets[name]
        _check_cpu_tensor(target, name + " (target)")
        if tuple(tensor.shape) != tuple(target.shape) or tensor.dtype != target.dtype:
            raise ValueError(
                f"{name}: checkpoint {tuple(tensor.shape)}/{tensor.dtype} != "
                f"model {tuple(target.shape)}/{target.dtype}"
            )
        _assert_finite(tensor, name)
        seen.add(name)
        yield name, tensor
        if len(seen) % 50 == 0:
            logger.info("OPENVDN_B_BASE_VALIDATED tensors=%d/%d", len(seen), len(expected))
    _assert_same_keys(seen, expected, label="complete Ref2VA base")
    logger.info("OPENVDN_B_BASE_VALIDATED tensors=%d seconds=%.3f", len(seen), time.monotonic() - started)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key {key!r}")
        result[key] = value
    return result


def inspect_header(path: Path):
    """Read metadata only, validate extent; digest is a HEADER hash, not file hash."""
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path}: truncated length field")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if not 2 <= header_size <= 16 * 1024 * 1024:
            raise ValueError(f"{path}: unreasonable header length {header_size}")
        raw = stream.read(header_size)
    if len(raw) != header_size:
        raise ValueError(f"{path}: truncated header")
    header = json.loads(raw, object_pairs_hook=_unique_json_object)
    metadata = header.pop("__metadata__", {})
    end = 0
    for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        lo, hi = info["data_offsets"]
        elements = 1
        for dim in info["shape"]:
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(f"{path}:{name}: invalid dimension {dim!r}")
            elements *= dim
        if info["dtype"] != "BF16" or lo != end or hi - lo != elements * 2:
            raise ValueError(f"{path}:{name}: unexpected dtype or invalid/noncontiguous extent")
        end = hi
    if path.stat().st_size != 8 + header_size + end:
        raise ValueError(f"{path}: truncated file or unexpected trailing bytes")
    return header, {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(header),
        "metadata": metadata,
    }


def lora_targets(model):
    """208 explicit pairs: Q/K/V/O of all 50 DiT and both refiner blocks."""
    for family, blocks, source_template in (
        ("blocks", model.blocks, "transformer_blocks.{}.attn.orig"),
        ("token_refiner.blocks", model.token_refiner.blocks, "token_refiner.refiner_blocks.{}.attn"),
    ):
        for index, block in enumerate(blocks):
            attn = block.attn
            rows = attn.total_num_heads * attn.head_dim
            if attn.num_heads != attn.total_num_heads:
                raise ValueError("LoRA merge requires full TP=1 parameter shapes")
            qkv = attn.qkv_proj.weight
            if qkv.shape[0] != 3 * rows:
                raise ValueError(f"{family}.{index}: unexpected fused QKV shape {tuple(qkv.shape)}")
            prefix = source_template.format(index)
            for offset, projection in enumerate(("to_q", "to_k", "to_v")):
                # NEVER re-run the base checkpoint's head-group reorder here.
                target = qkv[offset * rows : (offset + 1) * rows]
                yield prefix + "." + projection, target, f"{family}.{index}.attn.qkv_proj.weight[{offset}]"
            yield prefix + ".to_out.0", attn.out_proj.weight, f"{family}.{index}.attn.out_proj.weight"


def merge_lora_pair_(target, a, b, *, name: str, scale: float = STAGE_B_SCALE, chunk_rows: int = 256):
    """Official fp32-delta/BF16-add semantics, row-chunked for bounded CPU RAM."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    _check_cpu_tensor(target, name)
    _assert_finite(a, name + ".lora_A")
    _assert_finite(b, name + ".lora_B")
    if a.ndim != 2 or b.ndim != 2 or target.ndim != 2:
        raise ValueError(f"{name}: only matrix LoRA is supported")
    if a.shape[0] != b.shape[1] or tuple(target.shape) != (b.shape[0], a.shape[1]):
        raise ValueError(f"{name}: invalid LoRA shapes {tuple(a.shape)}, {tuple(b.shape)}, {tuple(target.shape)}")
    with torch.inference_mode():
        a_fp32 = a.float()
        for start in range(0, target.shape[0], chunk_rows):
            stop = min(start + chunk_rows, target.shape[0])
            delta = (b[start:stop].float() @ a_fp32) * scale
            _assert_finite(delta, name + ".delta")
            target[start:stop].add_(delta.to(target.dtype))
            _assert_finite(target[start:stop], name + ".merged")


def load_complete_openvdn_checkpoint(model, checkpoint: Path, *, phase_timings=None):
    """Load all 800 branch tensors, then merge all 208 default adapter pairs."""
    if getattr(model, "_openvdn_checkpoint_loaded", False):
        raise RuntimeError("Refusing to merge the same Stage-B checkpoint twice")
    started = time.monotonic()
    if phase_timings is None:
        phase_timings = {}
    mapping = branch_name_map()
    params = _targets(model)
    _assert_same_keys(set(mapping.values()) & params.keys(), mapping.values(), label="all branch parameters")
    branch_path = checkpoint / "linear_branch/model.safetensors"
    lora_path = checkpoint / "adapters/default/adapter_model.safetensors"
    branch_header, branch_record = inspect_header(branch_path)
    lora_header, lora_record = inspect_header(lora_path)
    _assert_same_keys(branch_header, mapping, label="full branch checkpoint")
    for source, target_name in mapping.items():
        target = params[target_name]
        _check_cpu_tensor(target, target_name)
        if target.dtype != torch.bfloat16 or tuple(branch_header[source]["shape"]) != tuple(target.shape):
            raise ValueError(f"{source}: branch checkpoint/model shape or dtype mismatch")

    pairs = list(lora_targets(model))
    if len(pairs) != LORA_PAIR_COUNT:
        raise ValueError(f"Expected {LORA_PAIR_COUNT} LoRA pairs, got {len(pairs)}")
    expected_lora = {}
    for source, target, _ in pairs:
        expected_lora[source + ".lora_A.default.weight"] = (STAGE_B_RANK, target.shape[1])
        expected_lora[source + ".lora_B.default.weight"] = (target.shape[0], STAGE_B_RANK)
    _assert_same_keys(lora_header, expected_lora, label="full default LoRA checkpoint")
    for key, shape in expected_lora.items():
        if tuple(lora_header[key]["shape"]) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {lora_header[key]['shape']}")
    meta_alpha = lora_record["metadata"].get("alpha")
    if meta_alpha is not None and float(meta_alpha) != STAGE_B_ALPHA:
        raise ValueError("LoRA metadata contradicts the verified Stage-B alpha=64 recipe")

    # Each tensor is consumed immediately; never materialize a second full DiT.
    loaded = set()
    branch_started = time.monotonic()
    with safe_open(branch_path, framework="pt", device="cpu") as stream:
        for source, target_name in mapping.items():
            tensor = stream.get_tensor(source)
            _assert_finite(tensor, source)
            current = model.load_weights(((target_name, tensor),))
            _assert_same_keys(current, (target_name,), label="branch tensor loader result")
            loaded.update(current)
            if len(loaded) % 80 == 0:
                logger.info("OPENVDN_B_BRANCH_LOADED tensors=%d/%d", len(loaded), BRANCH_TENSOR_COUNT)
    _assert_same_keys(loaded, mapping.values(), label="loaded branch coverage")
    phase_timings["branch"] = time.monotonic() - branch_started
    logger.info("OPENVDN_B_PHASE_COMPLETE phase=branch seconds=%.3f", phase_timings["branch"])

    lora_started = time.monotonic()
    with safe_open(lora_path, framework="pt", device="cpu") as stream:
        for index, (source, target, target_name) in enumerate(pairs, start=1):
            a = stream.get_tensor(source + ".lora_A.default.weight")
            b = stream.get_tensor(source + ".lora_B.default.weight")
            merge_lora_pair_(target, a, b, name=target_name)
            if index % 20 == 0 or index == len(pairs):
                logger.info("OPENVDN_B_LORA_MERGED pairs=%d/%d", index, len(pairs))
    phase_timings["lora"] = time.monotonic() - lora_started
    logger.info("OPENVDN_B_PHASE_COMPLETE phase=lora seconds=%.3f", phase_timings["lora"])

    record = {
        "checkpoint": str(checkpoint),
        "official_scale_source_commit": OFFICIAL_COMMIT,
        "lora_rank": STAGE_B_RANK,
        "lora_alpha": STAGE_B_ALPHA,
        "lora_scale": STAGE_B_SCALE,
        "base_tensor_count": BASE_TENSOR_COUNT,
        "branch_tensor_count": len(loaded),
        "lora_pairs_merged": len(pairs),
        "merge_dtype": "FP32 delta, cast to parameter dtype, then add",
        "base_partition": "ref2va",
        "qkv_merge_layout": "post-base-loader contiguous Q/K/V thirds",
        "branch": branch_record,
        "lora": lora_record,
        "load_and_merge_seconds": time.monotonic() - started,
        "phase_seconds": dict(phase_timings),
    }
    model._openvdn_checkpoint_loaded = True
    logger.info("OPENVDN_B_LOAD_RECORD %s", json.dumps(record, sort_keys=True))
    return loaded, record
