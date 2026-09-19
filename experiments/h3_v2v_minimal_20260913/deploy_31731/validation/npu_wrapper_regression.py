"""31731-only opt-in tiny BF16 regression using real Ulysses/HCCL.

Run ONLY via run_validation.py, which holds a private group lock and all eight
physical-card leases. Only the fixed single-node group 01234567 is allowed. No H3
weights are loaded: small deterministic projections exercise real candidate
forward, strategy selection, SeqAllToAll4D and NPU kernels against a pinned,
unsharded original OpenVDN reference. This is NOT a speed/quality benchmark.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace


import profiles

WORLD = profiles.WORLD
HEADS = 56
DIM = 128
HIDDEN = 64


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_small(module, generator, torch):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(parameter.shape, generator=generator, device="cpu", dtype=torch.float32)
            if name.endswith("A_log"):
                values = -1.0 + 0.02 * values
            elif name.endswith("norm.weight") or name in ("q_norm.weight", "k_norm.weight"):
                values = 1.0 + 0.02 * values
            else:
                values = 0.03 * values
            parameter.copy_(values.to(dtype=parameter.dtype, device=parameter.device))


def compare(name, actual, expected, torch, *, exact=False):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise AssertionError(f"{name}: nonfinite candidate or golden")
    # BF16 epsilon=1/128. Report errors as well as a pass/fail; do not silently
    # enlarge tolerances after seeing a failure. Tiny GEMMs can use different
    # kernels for full and sharded sequence lengths.
    atol, rtol = (0.0, 0.0) if exact else (1.0 / 512, 1.0 / 128)
    delta = (a - b).abs()
    stats = {
        "check": name,
        "shape": list(actual.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "rms_error": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "golden_rms": float(b.square().mean().sqrt()) if b.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda message: f"{name}: {message}")
    return stats


def build_holder(candidate, branch, Attention, factory, NoParallelAttention, torch, generator):
    """Use real candidate methods/real Attention selector without a huge model."""
    nn = torch.nn

    class TupleLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    attention = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.skip_sequence_parallel = False
    attention.scatter_idx = 2
    attention.gather_idx = 1
    attention.use_sync = False
    attention.use_ring = False
    attention._kv_cache_dtype = None
    attention.parallel_strategy = factory(scatter_idx=2, gather_idx=1, use_sync=False, causal=False)
    attention._no_parallel_strategy = NoParallelAttention()

    holder = candidate.MiniMaxH3Attention.__new__(candidate.MiniMaxH3Attention)
    nn.Module.__init__(holder)
    holder.attention = attention
    holder.total_num_heads = holder.num_heads = holder.num_kv_heads = HEADS
    holder.head_dim = DIM
    holder.softmax_scale = DIM ** -0.5
    holder.openvdn_enabled = holder.openvdn_linear_enabled = True
    holder.openvdn_groups_per_call = 4
    holder.openvdn_interior_group_size = 0
    holder.openvdn_group_impl = "varlen"
    holder.qkv_proj = TupleLinear(HIDDEN, 3 * HEADS * DIM, bias=False, dtype=torch.bfloat16)
    holder.q_norm = candidate._norm(DIM, eps=1e-5)
    holder.k_norm = candidate._norm(DIM, eps=1e-5)
    holder.out_proj = TupleLinear(HEADS * DIM, HIDDEN, bias=False, dtype=torch.bfloat16)
    holder.softmax_gate = branch.OutputGate(HIDDEN, HEADS, dtype=torch.bfloat16)
    holder.linear_attention = branch.BidirectionalLinearBranch(HIDDEN, HEADS, DIM)
    holder.to_out_linear = nn.Linear(HEADS * DIM, HIDDEN, bias=False, dtype=torch.bfloat16)
    initialize_small(holder, generator, torch)
    return holder


def make_inputs(case, candidate, branch, torch, generator, device):
    # Case 1 has text/source spanning a rank boundary. Case 2 puts the target
    # before a long source/text suffix so multiple ranks hold no target rows;
    # both have target frames crossing a shard boundary and trailing padding.
    if case == "source_prefix":
        frames, height, width, start, text_start, text_len, suffix = 17, 2, 2, 73, 1, 43, 9
    else:
        frames, height, width, start, text_start, text_len, suffix = 7, 2, 4, 3, 76, 33, 73
    per_frame = height * width
    used = start + frames * per_frame + suffix
    packed = ((used + 11 + WORLD - 1) // WORLD) * WORLD
    layout = branch.OpenVDNLayout(
        used_len=used, video_start=start, num_frames=frames,
        tokens_per_frame=per_frame, frame_height=height, frame_width=width,
        text_start=text_start, text_len=text_len,
    )
    layout.validate(packed)
    x = (torch.randn((packed, HIDDEN), generator=generator, device="cpu") * 0.3).to(
        device=device, dtype=torch.bfloat16,
    )
    positions = torch.zeros((1, packed, 3), dtype=torch.float32)
    positions[0, :, 0] = torch.arange(packed, dtype=torch.float32) * 0.13
    hh, ww = torch.meshgrid(torch.arange(height) * 0.7, torch.arange(width) * 0.9, indexing="ij")
    for frame in range(frames):
        lo = start + frame * per_frame
        positions[0, lo : lo + per_frame, 0] = 91 + frame * 1.7
        positions[0, lo : lo + per_frame, 1] = hh.reshape(-1)
        positions[0, lo : lo + per_frame, 2] = ww.reshape(-1)
    rope_module = candidate.MiniMaxH3Rope(inv_freq_len=16)
    rope_module.inv_freq.copy_(10000.0 ** -(torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    rope = rope_module.to(device)(positions.to(device))
    cu = torch.tensor([0, used, packed], device=device, dtype=torch.int32)
    return x, rope, cu, layout


def prepare_qkv(holder, x, rope, candidate):
    packed_qkv, _ = holder.qkv_proj(x)
    raw = tuple(t.reshape(x.shape[0], HEADS, DIM) for t in packed_qkv.chunk(3, dim=-1))
    query = candidate._apply_rope(holder.q_norm(raw[0]), rope)
    key = candidate._apply_rope(holder.k_norm(raw[1]), rope)
    return raw, (query, key, raw[2])


def worker(rank, args_dict, manifest):
    # mp.spawn imports a fresh interpreter: the physical allocation is passed
    # explicitly, not communicated by mutating a module global in the parent.
    selected = profiles.require_cards(args_dict["group"], args_dict["physical_cards"])
    host = profiles.require_host(args_dict["host"])
    if os.environ.get("H3_MINIMAL_RUN_ID") != args_dict["run_id"]:
        raise RuntimeError("Worker run identity differs from its supervisor")
    if profiles.source_manifest(selected) != manifest:
        raise RuntimeError("Sources changed before worker NPU initialization")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args_dict["master_port"])
    import torch
    import torch_npu  # noqa: F401 -- register NPU/HCCL before vLLM platform selection
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.npu.set_device(rank)  # logical rank maps to the explicit physical_cards[rank]
    if torch.npu.device_count() != WORLD:
        raise RuntimeError("Expected exactly eight visible NPUs")
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.attention.parallel.factory import build_parallel_attention_strategy
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

    output = Path(args_dict["output"])
    rank_output = output.with_name(f"{output.stem}.rank{rank}.json")
    report = {
        "status": "running", "rank": rank, "physical_card": args_dict["physical_cards"][rank],
        "group": selected.group, "host": host, "run_id": args_dict["run_id"],
        "source_sha256": manifest, "tests": [],
    }
    initialized = False
    began = time.perf_counter()
    try:
        if profiles.source_manifest(selected) != manifest:
            raise RuntimeError("Test source files changed after launch")
        dist.init_process_group(
            backend="hccl", init_method=f"tcp://127.0.0.1:{args_dict['master_port']}",
            rank=rank, world_size=WORLD,
            timeout=datetime.timedelta(seconds=args_dict["collective_timeout"]),
        )
        init_distributed_environment(
            world_size=WORLD, rank=rank, local_rank=rank, backend="hccl",
            distributed_init_method=f"tcp://127.0.0.1:{args_dict['master_port']}",
        )
        initialize_model_parallel(
            sequence_parallel_size=WORLD, ulysses_degree=WORLD, ring_degree=1,
            tensor_parallel_size=1, backend="hccl",
        )
        initialized = True
        sp = get_sp_group()
        assert sp.ulysses_world_size == WORLD and sp.ulysses_rank == rank
        assert sp.ring_world_size == 1 and get_tensor_model_parallel_world_size() == 1
        # Evidence of one actual eight-rank collective, not two four-rank groups.
        rank_tensor = torch.tensor([rank], dtype=torch.int32, device=f"npu:{rank}")
        gathered = [torch.empty_like(rank_tensor) for _ in range(WORLD)]
        dist.all_gather(gathered, rank_tensor, group=sp.ulysses_group)
        peers = [int(value.cpu().item()) for value in gathered]
        if peers != list(range(WORLD)):
            raise RuntimeError("Ulysses communicator does not contain all eight logical ranks")
        report["parallelism_proof"] = {
            "ulysses_world_size": sp.ulysses_world_size,
            "ulysses_rank": sp.ulysses_rank, "ring_world_size": sp.ring_world_size,
            "tensor_parallel_world_size": get_tensor_model_parallel_world_size(),
            "collective_global_ranks": peers, "heads_per_ulysses_rank": HEADS // WORLD,
        }

        # Process-local module substitution only. No installed/shared source is
        # edited, and importing the model definition does not load a checkpoint.
        upstream = load_module("h3_npu_regression_upstream", Path(manifest["upstream/openvdn_npu.py"]["path"]))
        branch = load_module(
            "vllm_omni.diffusion.models.minimax_h3.openvdn_npu", Path(manifest["candidate/openvdn_npu.py"]["path"]),
        )
        candidate = load_module("h3_npu_regression_candidate", Path(manifest["candidate/minimax_h3_transformer.py"]["path"]))
        cfg = SimpleNamespace(parallel_config=SimpleNamespace(
            sequence_parallel_size=WORLD, ulysses_degree=WORLD, ring_degree=1,
            allgather_degree=1, ulysses_mode="strict",
        ))
        report["real_strategy_source"] = str(Path(sys.modules[UlyssesParallelAttention.__module__].__file__).resolve())
        report["real_strategy_sha256"] = digest(report["real_strategy_source"])
        expected_strategy = manifest["strategy/ulysses.py"]
        if (report["real_strategy_source"] != expected_strategy["path"]
                or report["real_strategy_sha256"] != expected_strategy["sha256"]):
            raise RuntimeError("Actually imported Ulysses strategy is not the manifested candidate vendor")
        report["torch_version"] = torch.__version__
        report["torch_npu_version"] = torch_npu.__version__
        report["device_name"] = torch.npu.get_device_name(rank)
        with set_forward_context(omni_diffusion_config=cfg), torch.inference_mode():
            generator = torch.Generator(device="cpu").manual_seed(8731)
            holder = build_holder(candidate, branch, Attention, build_parallel_attention_strategy,
                                  NoParallelAttention, torch, generator).eval().to(f"npu:{rank}")
            original = upstream.BidirectionalLinearBranch(HIDDEN, HEADS, DIM)
            original.load_state_dict(holder.linear_attention.state_dict(), strict=True)
            original = original.eval().to(f"npu:{rank}")
            strategy = holder.attention._get_active_parallel_strategy()
            if not isinstance(strategy, UlyssesParallelAttention) or strategy.name != "ulysses":
                raise RuntimeError(f"Actual Attention selected {type(strategy)!r}, not Ulysses")
            for case in ("source_prefix", "source_suffix"):
                x, rope, cu, layout = make_inputs(case, candidate, branch, torch, generator, f"npu:{rank}")
                rows = x.shape[0] // WORLD
                lo, hi = rank * rows, (rank + 1) * rows
                local_x, local_rope = x[lo:hi].contiguous(), rope[lo:hi].contiguous()
                raw, qkv = prepare_qkv(holder, x, rope, candidate)
                local_raw, local_qkv = prepare_qkv(holder, local_x, local_rope, candidate)

                # Original, unsharded B serves as the tiny golden reference.
                golden_soft = upstream.openvdn_softmax_attention(*qkv, layout, DIM ** -0.5)
                golden_linear = torch.zeros((x.shape[0], HEADS * DIM), dtype=x.dtype, device=x.device)
                golden_linear[layout.video_start : layout.video_end] = original(x, raw, layout)
                golden_out = holder.out_proj((golden_soft * holder.softmax_gate(x)).flatten(1))[0]
                golden_out = golden_out + holder.to_out_linear(golden_linear)

                # Direct real wrapper verifies the head-sharded readouts and
                # beta[D=1] all-to-all, before the final projections.
                got_soft, got_linear = holder._run_openvdn_ulysses(
                    local_x, *local_qkv, local_raw, layout, cu, strategy,
                )
                checks = [
                    compare("wrapper-softmax", got_soft, golden_soft[lo:hi], torch),
                    compare("wrapper-linear-gated", got_linear, golden_linear[lo:hi], torch),
                ]
                actual = holder(
                    local_x, rope_freqs=local_rope, cu_seqlens=cu,
                    max_seqlen=layout.used_len, openvdn_layout=layout,
                )
                checks.append(compare("actual-attention-forward-projected", actual, golden_out[lo:hi], torch))
                global_rows = torch.arange(lo, hi, device=x.device)
                zero_linear = (global_rows < layout.video_start + layout.tokens_per_frame) | (
                    global_rows >= layout.video_end - layout.tokens_per_frame
                )
                checks.append(compare("source-text-anchor-padding-linear-zero", got_linear[zero_linear],
                                      torch.zeros_like(got_linear[zero_linear]), torch, exact=True))
                padding = global_rows >= layout.used_len
                checks.append(compare("padding-final-output-zero", actual[padding],
                                      torch.zeros_like(actual[padding]), torch, exact=True))
                torch.npu.synchronize()
                report["tests"].append({
                    "case": case, "packed_rows": x.shape[0], "local_rows": [lo, hi],
                    "layout": vars(layout), "checks": checks,
                })
                print(json.dumps({"rank": rank, "case": case, "status": "passed",
                                  "max_abs": max(item["max_abs"] for item in checks)}), flush=True)
            dist.barrier(group=sp.ulysses_group)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        write_new(rank_output, report)
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
        elif dist.is_initialized():
            dist.destroy_process_group()


def parse_and_validate(argv=None):
    """All CPU-only gates run before torch or the NPU runtime is imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-npu", action="store_true")
    parser.add_argument("--group", choices=tuple(profiles.GROUPS), required=True)
    parser.add_argument("--physical-cards", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host-proof", type=Path, required=True)
    parser.add_argument("--manifest-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--collective-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if not args.allow_npu:
        parser.error("Only the lease-holding supervisor may opt in to this NPU test")
    if not re.fullmatch(r"[0-7](?:,[0-7]){7}", args.physical_cards):
        parser.error("An explicit ordered eight-physical-card list is required")
    args.physical_cards = [int(card) for card in args.physical_cards.split(",")]
    selected = profiles.require_cards(args.group, args.physical_cards)
    if args.master_port != selected.master_port or args.collective_timeout != 180:
        parser.error("The group has a fixed rendezvous port and 180-second collective timeout")
    if (not re.fullmatch(r"[0-9a-f]{32}", args.run_id)
            or os.environ.get("H3_MINIMAL_RUN_ID") != args.run_id
            or os.environ.get("H3_VALIDATION_GROUP") != selected.group
            or os.environ.get("H3_VALIDATION_SUPERVISOR_PID") != str(os.getppid())):
        raise RuntimeError("Not a child of the expected lease-holding supervisor")
    root = profiles.canonical_private(args.output.parent)
    if (root.parent != selected.output_root / "runs"
            or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_" + args.run_id, root.name)
            or args.output != root / "npu_wrapper.json"
            or args.host_proof != root / "host_identity.json"
            or args.manifest_proof != root / "source_manifest.json"
            or os.environ.get("H3_ROOT") != str(root)
            or os.environ.get("H3_OUTPUT") != str(root)):
        raise RuntimeError("Output/proof paths do not match this group's existing supervisor run")
    for proof in (args.host_proof, args.manifest_proof):
        profiles.canonical_private(proof)
        if not proof.is_file() or proof.is_symlink():
            raise RuntimeError("Supervisor proof is missing/nonregular")
    args.host = profiles.require_host(json.loads(args.host_proof.read_text()))
    if Path(__file__).resolve().parent != selected.code_root:
        raise RuntimeError("Wrapper must be deployed at the fixed private validation code root")
    for path in [args.output, *(args.output.with_name(f"{args.output.stem}.rank{rank}.json") for rank in range(WORLD))]:
        profiles.canonical_private(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    manifest = profiles.source_manifest(selected)
    if json.loads(args.manifest_proof.read_text()) != manifest:
        raise RuntimeError("Supervisor source manifest differs from the files about to execute")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", args.master_port))
    return args, selected, manifest


def main(argv=None):
    args, selected, manifest = parse_and_validate(argv)
    report = {
        "status": "running", "physical_cards": args.physical_cards, "world_size": WORLD,
        "group": selected.group, "host": args.host, "run_id": args.run_id,
        "dtype": "bfloat16", "heads": HEADS, "head_dim": DIM, "hidden": HIDDEN,
        "source_sha256": manifest, "test_script_sha256": digest(__file__),
        "limitations": [
            "Tiny synthetic weights/tensors, not H3/VDN checkpoint loading or video quality.",
            "QKV/output modules are small tuple-returning nn.Linear stubs, not vLLM checkpoint loaders.",
            "Real Attention strategy/factory/Ulysses/SeqAllToAll4D/HCCL and candidate forward execute.",
            "Attention backend construction is bypassed; OpenVDN dispatches its real NPU Softmax kernel.",
            "Full-model offload, service, large-shape memory and speed remain untested.",
        ],
    }
    began = time.perf_counter()
    import torch.multiprocessing as mp
    try:
        data = vars(args).copy()
        # Pass the ordered physical list and real host/boot proof into EVERY
        # spawned worker explicitly. No fork-only global allocation trick.
        data.update(output=str(args.output), physical_cards=list(selected.cards), host=dict(args.host))
        mp.spawn(worker, args=(data, manifest), nprocs=WORLD, join=True)
        profiles.require_host(args.host)
        if profiles.source_manifest(selected) != manifest:
            raise RuntimeError("Source files changed during validation")
        rank_results = [json.loads(args.output.with_name(f"{args.output.stem}.rank{rank}.json").read_text()) for rank in range(WORLD)]
        if any(item["status"] != "passed" for item in rank_results):
            raise RuntimeError("At least one rank did not pass")
        report.update(status="passed", rank_results=rank_results)
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        write_new(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
