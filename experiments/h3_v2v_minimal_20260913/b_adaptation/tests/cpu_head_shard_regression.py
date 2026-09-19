"""Small CPU-only numerical checks of the OpenVDN USP head-shard adaptation.

No H3 model is constructed, weights are not loaded, and no NPU tensor or device
API is used. The upstream module may import torch_npu as an optional dependency;
its attention dispatcher stays on CPU because every input explicitly uses CPU.

The Gloo test performs actual four-rank all_to_all_single/all_reduce operations.
Its transport adapter is a test implementation of sequence/head exchange, not
the vLLM/HCCL wrapper: it validates the decomposition but not HCCL integration.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


BASE = Path(__file__).resolve().parents[1]
WORLD_SIZE = 4
HEADS = 8
HIDDEN = 24


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules(base: Path = BASE):
    return (
        load_module("openvdn_cpu_upstream", base / "upstream/openvdn_npu.py"),
        load_module("openvdn_cpu_candidate", base / "patched/openvdn_npu.py"),
    )


@dataclasses.dataclass(frozen=True)
class Case:
    seed: int
    frames: int
    prefix: int
    text: int
    suffix: int
    extra_pad: int
    dim: int = 8
    text_before_video: bool = False

    def describe(self):
        return dataclasses.asdict(self)


CASES = (
    Case(11, 7, 9, 3, 2, 5, 4),
    Case(29, 17, 11, 5, 3, 7, 8),
    Case(41, 17, 5, 0, 3, 4, 4),
    Case(53, 7, 13, 1, 1, 1, 8, True),
    Case(67, 17, 3, 1, 0, 0, 8),
)


def initialize_small(module, generator):
    """Deterministic, finite weights; do not leave empty A_log/dt_bias uninitialized."""
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            sample = torch.randn(parameter.shape, generator=generator, device="cpu")
            if name.endswith("norm.weight"):
                sample = 1.0 + sample * 0.03
            elif name.endswith("A_log"):
                sample = -1.0 + sample * 0.03
            else:
                sample = sample * 0.05
            parameter.copy_(sample.to(parameter.dtype))


def make_case(case, dtype, upstream, candidate):
    per_frame = 4
    text_start = 1 if case.text_before_video else case.prefix + case.frames * per_frame
    if case.text_before_video and text_start + case.text > case.prefix:
        raise ValueError("text-before-video must fit within prefix")
    used = case.prefix + case.frames * per_frame + case.suffix
    if not case.text_before_video:
        used += case.text
    packed = ((used + case.extra_pad + WORLD_SIZE - 1) // WORLD_SIZE) * WORLD_SIZE
    # Candidate adds validation; the original code consumes the same fields.
    layout = candidate.OpenVDNLayout(
        used_len=used,
        video_start=case.prefix,
        num_frames=case.frames,
        tokens_per_frame=per_frame,
        frame_height=2,
        frame_width=2,
        text_start=text_start,
        text_len=case.text,
    )
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    x = (torch.randn(packed, HIDDEN, generator=generator, device="cpu") * 0.25).to(dtype)
    qkv = tuple(
        (torch.randn(packed, HEADS, case.dim, generator=generator, device="cpu") * 0.25).to(dtype)
        for _ in range(3)
    )
    original = upstream.BidirectionalLinearBranch(HIDDEN, HEADS, case.dim).to(dtype=dtype)
    initialize_small(original, generator)
    sharded = candidate.BidirectionalLinearBranch(HIDDEN, HEADS, case.dim).to(dtype=dtype)
    sharded.load_state_dict(original.state_dict(), strict=True)
    softmax_gate = upstream.OutputGate(HIDDEN, HEADS, dtype=dtype)
    initialize_small(softmax_gate, generator)
    for tensor in (x, *qkv, *original.parameters(), *sharded.parameters()):
        assert tensor.device.type == "cpu"
    return x, qkv, layout, original.eval(), sharded.eval(), softmax_gate.eval()


def compare(name, actual, expected, dtype, exact=False):
    assert actual.device.type == expected.device.type == "cpu"
    assert torch.isfinite(actual).all(), f"{name}: candidate contains nonfinite values"
    assert torch.isfinite(expected).all(), f"{name}: golden contains nonfinite values"
    if exact:
        atol = rtol = 0.0
    elif dtype == torch.float32:
        atol, rtol = 2e-6, 2e-5
    else:
        # BF16 has 7 fraction bits. This permits approximately one ULP away
        # from zero plus a small absolute floor, not multi-percent drift.
        atol, rtol = 1.0 / 512, 1.0 / 128
    delta = (actual.float() - expected.float()).abs()
    stats = {
        "check": name,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "rms_error": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}")
    return stats


def manual_softmax(qkv, layout, scale):
    """Independent Boolean-mask oracle, including global source rows and anchors."""
    query, key, value = qkv
    used = layout.used_len
    mask = torch.zeros(used, used, dtype=torch.bool, device="cpu")
    global_rows = list(range(layout.video_start)) + list(range(layout.video_end, used))
    anchor_rows = (
        list(range(layout.video_start, layout.video_start + layout.tokens_per_frame))
        + list(range(layout.video_end - layout.tokens_per_frame, layout.video_end))
    )
    mask[global_rows + anchor_rows, :] = True
    for frame in range(1, layout.num_frames - 1):
        lo = max(0, (frame // 5 - 1) * 5)
        hi = min(layout.num_frames - 1, (frame // 5 + 2) * 5 - 1)
        queries = slice(layout.video_start + frame * 4, layout.video_start + (frame + 1) * 4)
        keys = list(range(layout.video_start + lo * 4, layout.video_start + (hi + 1) * 4))
        mask[queries, global_rows + anchor_rows + keys] = True
    result = torch.zeros_like(query)
    result[:used] = F.scaled_dot_product_attention(
        query[:used].transpose(0, 1).unsqueeze(0),
        key[:used].transpose(0, 1).unsqueeze(0),
        value[:used].transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        scale=scale,
    ).squeeze(0).transpose(0, 1)
    return result


def golden_outputs(x, qkv, layout, original, softmax_gate, upstream):
    softmax = upstream.openvdn_softmax_attention(*qkv, layout, qkv[0].shape[-1] ** -0.5)
    linear = torch.zeros_like(qkv[0])
    linear[layout.video_start:layout.video_end] = original(x, qkv, layout).reshape(
        layout.num_frames * layout.tokens_per_frame, HEADS, -1
    )
    return softmax, linear, softmax * softmax_gate(x) + linear


def frame_mean_reference(x, layout):
    begin = layout.video_start + layout.tokens_per_frame
    end = layout.video_end - layout.tokens_per_frame
    return x[begin:end].reshape(layout.num_frames - 2, layout.tokens_per_frame, -1).mean(1, dtype=torch.float32)


def check_frame_reduction(x, layout, candidate):
    length = x.shape[0] // WORLD_SIZE
    sums, counts = [], []
    for rank in range(WORLD_SIZE):
        result = candidate.local_frame_sums_counts(x[rank * length:(rank + 1) * length], layout, rank * length)
        sums.append(result[0])
        counts.append(result[1])
    total_sum, total_count = torch.stack(sums).sum(0), torch.stack(counts).sum(0)
    expected_count = torch.full_like(total_count, layout.tokens_per_frame)
    count_stats = compare("frame-counts", total_count, expected_count, torch.float32, exact=True)
    mean = total_sum / total_count.unsqueeze(-1)
    mean_stats = compare("frame-mean", mean, frame_mean_reference(x, layout), torch.float32)
    return mean, [count_stats, mean_stats]


def check_single(case, dtype, upstream, candidate):
    x, qkv, layout, original, sharded, gate = make_case(case, dtype, upstream, candidate)
    softmax, linear, hybrid = golden_outputs(x, qkv, layout, original, gate, upstream)
    stats = [compare("upstream-softmax-independent-mask", softmax, manual_softmax(qkv, layout, case.dim ** -0.5), dtype)]
    frame_mean, frame_stats = check_frame_reduction(x, layout, candidate)
    stats.extend(frame_stats)
    beta_logits = sharded.beta_proj(x)
    shard_softmax, shard_linear = [], []
    for rank in range(WORLD_SIZE):
        h0, h1 = rank * (HEADS // WORLD_SIZE), (rank + 1) * (HEADS // WORLD_SIZE)
        local_qkv = tuple(t[:, h0:h1].contiguous() for t in qkv)
        shard_softmax.append(candidate.openvdn_softmax_attention(*local_qkv, layout, case.dim ** -0.5))
        shard_linear.append(sharded.forward_head_shard(local_qkv, beta_logits[:, h0:h1].contiguous(), frame_mean, layout, head_start=h0))
    actual_softmax = torch.cat(shard_softmax, dim=1)
    normalized_linear = torch.cat(shard_linear, dim=1)
    actual_linear = normalized_linear * sharded.output_gate(x)
    actual_hybrid = actual_softmax * gate(x) + actual_linear
    stats.extend((
        compare("four-head-shards-softmax", actual_softmax, softmax, dtype),
        compare("four-head-shards-linear", actual_linear, linear, dtype),
        compare("four-head-shards-hybrid", actual_hybrid, hybrid, dtype),
    ))
    zero = torch.ones(x.shape[0], dtype=torch.bool, device="cpu")
    zero[layout.video_start + 4:layout.video_end - 4] = False
    stats.append(compare("linear-source-text-anchor-padding-zero", actual_linear[zero], torch.zeros_like(actual_linear[zero]), dtype, exact=True))
    stats.append(compare("softmax-padding-zero", actual_softmax[layout.used_len:], torch.zeros_like(actual_softmax[layout.used_len:]), dtype, exact=True))
    return {"case": case.describe(), "dtype": str(dtype), "checks": stats}


def sequence_to_head(local):
    """[local S, all H, D] -> [all S, local H, D] via real Gloo A2A."""
    rows, heads, dim = local.shape
    per_head = heads // WORLD_SIZE
    outgoing = local.reshape(rows, WORLD_SIZE, per_head, dim).permute(1, 0, 2, 3).contiguous()
    incoming = torch.empty_like(outgoing)
    dist.all_to_all_single(incoming, outgoing)
    return incoming.reshape(WORLD_SIZE * rows, per_head, dim)


def head_to_sequence(global_head):
    """Inverse A2A. Rank order is sequence order on send, head order on receive."""
    rows, heads, dim = global_head.shape
    local_rows = rows // WORLD_SIZE
    outgoing = global_head.reshape(WORLD_SIZE, local_rows, heads, dim).contiguous()
    incoming = torch.empty_like(outgoing)
    dist.all_to_all_single(incoming, outgoing)
    return incoming.permute(1, 0, 2, 3).reshape(local_rows, WORLD_SIZE * heads, dim)


def gloo_worker(rank, init_method, base, cases, dtype_names, result_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=120))
    try:
        upstream, candidate = modules(Path(base))
        records = []
        with torch.inference_mode():
            for dtype_name in dtype_names:
                dtype = getattr(torch, dtype_name)
                for case in cases:
                    x, qkv, layout, original, sharded, gate = make_case(case, dtype, upstream, candidate)
                    rows = x.shape[0] // WORLD_SIZE
                    lo, hi = rank * rows, (rank + 1) * rows
                    x_local = x[lo:hi].contiguous()
                    head_qkv = tuple(sequence_to_head(t[lo:hi].contiguous()) for t in qkv)
                    h0, h1 = rank * 2, (rank + 1) * 2
                    checks = [compare(f"a2a-qkv-{i}", part, full[:, h0:h1], dtype, exact=True) for i, (part, full) in enumerate(zip(head_qkv, qkv))]
                    beta_head = sequence_to_head(sharded.beta_proj(x_local).unsqueeze(-1)).squeeze(-1)
                    checks.append(compare("a2a-beta", beta_head, sharded.beta_proj(x)[:, h0:h1], dtype))
                    sums, counts = candidate.local_frame_sums_counts(x_local, layout, lo)
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    checks.append(compare("allreduce-counts", counts, torch.full_like(counts, 4), torch.float32, exact=True))
                    means = sums / counts.unsqueeze(-1)
                    checks.append(compare("allreduce-frame-mean", means, frame_mean_reference(x, layout), torch.float32))
                    softmax_head = candidate.openvdn_softmax_attention(*head_qkv, layout, case.dim ** -0.5)
                    linear_head = sharded.forward_head_shard(head_qkv, beta_head, means, layout, head_start=h0)
                    local_softmax = head_to_sequence(softmax_head)
                    local_linear = head_to_sequence(linear_head) * sharded.output_gate(x_local)
                    local_hybrid = local_softmax * gate(x_local) + local_linear
                    softmax, linear, hybrid = golden_outputs(x, qkv, layout, original, gate, upstream)
                    checks.extend((
                        compare("gloo-softmax", local_softmax, softmax[lo:hi], dtype),
                        compare("gloo-linear", local_linear, linear[lo:hi], dtype),
                        compare("gloo-hybrid", local_hybrid, hybrid[lo:hi], dtype),
                    ))
                    records.append({"case": case.describe(), "dtype": dtype_name, "checks": checks})
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(records, gathered, dst=0)
        if rank == 0:
            Path(result_path).write_text(json.dumps({"world_size": WORLD_SIZE, "rank_results": gathered}, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def run_gloo(base, cases, dtype_names, output):
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch Gloo unavailable; cannot claim collective validation")
    # Loopback port is held for selection then released before child rendezvous.
    # Concurrent bind failure is explicit; no external network interface is used.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(gloo_worker, args=(f"tcp://127.0.0.1:{port}", str(base), cases, dtype_names, str(output)), nprocs=WORLD_SIZE, join=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "gloo", "all"), default="all")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file; refuses overwrite")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "both"), default="both")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"result already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype_names = ("float32", "bfloat16") if args.dtype == "both" else (args.dtype,)
    torch.set_num_threads(1)
    began = time.perf_counter()
    report = {
        "status": "running",
        "torch_version": torch.__version__,
        "device": "cpu",
        "world_size": WORLD_SIZE,
        "source_sha256": {
            str(path.relative_to(args.base)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.base / "upstream/openvdn_npu.py", args.base / "patched/openvdn_npu.py")
        },
        "npu_calls": "none; optional module import only",
        "limitations": [
            "Synthetic tiny tensors; not trained weights, video quality, NPU performance, or large-shape memory validation.",
            "Gloo transport is the test adapter, not the actual vLLM/HCCL wrapper or end-to-end transformer block.",
            "LoRA loading, full-model integration, and distributed output projection are outside this test.",
        ],
        "single_results": [],
    }
    if args.mode in ("single", "all"):
        upstream, candidate = modules(args.base)
        with torch.inference_mode():
            for dtype_name in dtype_names:
                for case in CASES:
                    record = check_single(case, getattr(torch, dtype_name), upstream, candidate)
                    report["single_results"].append(record)
                    print(json.dumps({"pass": "single", "dtype": dtype_name, "case": case.describe(), "max_abs": max(item["max_abs"] for item in record["checks"])}), flush=True)
    if args.mode in ("gloo", "all"):
        gloo_output = args.output.with_name(args.output.stem + ".gloo.json")
        if gloo_output.exists():
            raise FileExistsError(f"Gloo result already exists: {gloo_output}")
        run_gloo(args.base, CASES, dtype_names, gloo_output)
        report["gloo_results_path"] = str(gloo_output)
    report["status"] = "passed"
    report["wall_seconds"] = time.perf_counter() - began
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "mode": args.mode, "output": str(args.output), "seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
