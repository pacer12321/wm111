"""Actual C attention/SP8 regression. Entry restricted to its one-run supervisor.

No checkpoints, server, video generation or acceleration measurement. Full
56 heads are split over one real eight-rank HCCL/Ulysses group, 7 heads/rank.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
import traceback
import types
from types import SimpleNamespace

import c_profiles as p
from c_cases import CASES, REJECTIONS, WORLD, HEADS, DIM, HIDDEN, fixture, invalid_fixtures, oracle_masks, audit_masks, geometry

CHECKS = {"c-wrapper-softmax", "unchanged-b-linear", "c-actual-forward", "wrong-source-anchor-zero",
          "linear-nontarget-anchor-padding-zero", "padding-output-zero", "wrong-source-dense-oracle"}


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def tensors(data, torch):
    return dict(img_pos=torch.tensor(data["img_pos"], dtype=torch.int64),
                update_mask=torch.tensor(data["update_mask"], dtype=torch.bool),
                text_pos=torch.tensor(data["text_pos"], dtype=torch.int64),
                audio_pos=torch.tensor(data["audio_pos"], dtype=torch.int64),
                img_position_ids=torch.tensor(data["coords"], dtype=torch.float64).unsqueeze(0),
                cu_seqlens=torch.tensor(data["cu_seqlens"], dtype=torch.int32), metadata=data["metadata"])


def metadata_checks(adapter, torch):
    """Executed with CPU tensors before NPU set_device or any collective."""
    proof = {}
    for case in CASES:
        data = fixture(case)
        layout = adapter.infer_strict_source_layout(**tensors(data, torch))
        evidence = audit_masks(layout, len(data["coords"]))
        rejected = []
        for name, invalid in invalid_fixtures(case).items():
            try:
                adapter.infer_strict_source_layout(**tensors(invalid, torch))
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f"C silently accepted bad metadata: {case}/{name}")
        proof[case] = dict(rejected=rejected, layout=vars(layout), oracle=evidence)
    return proof


def dense_cpu(q, k, v, layout, torch):
    """Independent float32 CPU masked matrix attention, NOT C's span kernel."""
    _, mask = oracle_masks(layout, q.shape[0])
    qf, kf, vf = (x.detach().float().cpu().permute(1, 0, 2) for x in (q, k, v))
    allowed = torch.tensor(mask[:layout.used_len], dtype=torch.bool)
    scores = torch.matmul(qf[:, :layout.used_len], kf.transpose(-1, -2)) * DIM ** -0.5
    probability = torch.softmax(scores.masked_fill(~allowed.unsqueeze(0), float("-inf")), dim=-1)
    out = torch.zeros_like(qf)
    out[:, :layout.used_len] = torch.matmul(probability, vf)
    return out.permute(1, 0, 2).to(device=q.device, dtype=q.dtype)


def verify_parent_leases(root, run_id, host, base):
    """Require inherited live mutex FDs and the exact still-live parent process."""
    import fcntl
    path = root / "owner_proof.json"
    p.canonical(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing owner proof")
    owner = json.loads(path.read_text())
    pid = os.getppid()
    current = base.proc_identity(pid)
    expected_paths = [str(p.Profile().output_root / "run.lock")] + [str(p.Profile().lease_root / f"device{c}.lock") for c in p.CARDS]
    if (owner.get("supervisor_pid") != pid or os.environ.get("H3_C_SUPERVISOR_PID") != str(pid)
            or owner.get("run_id") != run_id or owner.get("host") != host or owner.get("case") != p.CASE
            or owner.get("cards") != list(p.CARDS) or owner.get("resources_checked") is not True
            or not current or not owner.get("proc_identity")
            or any(current.get(k) != owner["proc_identity"].get(k) for k in ("pid", "start_ticks", "pgrp", "session"))
            or owner.get("lease_paths") != expected_paths):
        raise RuntimeError("Not the live, resource-checked C supervisor child")
    fds = owner.get("lease_fds", [])
    if len(fds) != 9 or len(set(fds)) != 9 or any(type(fd) is not int or fd < 3 for fd in fds):
        raise RuntimeError("Incomplete nine inherited lease FDs")
    for fd, name in zip(fds, expected_paths):
        path = p.canonical(Path(name))
        a, b = os.fstat(fd), path.stat()
        if (not stat.S_ISREG(a.st_mode) or a.st_uid != os.getuid() or a.st_nlink != 1
                or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino)):
            raise RuntimeError("Lease FD does not identify its private mutex")
        check = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            try:
                fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError("Expected personal mutex is not actually locked")
        finally:
            os.close(check)


def parse_and_validate(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-npu", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.allow_npu:
        parser.error("Explicit --allow-npu required; never invoke outside the supervisor")
    if (not re.fullmatch(r"[0-9a-f]{32}", args.run_id)
            or os.environ.get("H3_MINIMAL_RUN_ID") != args.run_id
            or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"):
        raise RuntimeError("C run/physical allocation mismatch")
    root = p.canonical(args.output.parent)
    if (root.parent != p.Profile().output_root / "runs"
            or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_" + args.run_id, root.name)
            or args.output != root / "c_tiny.json" or os.environ.get("H3_ROOT") != str(root)
            or os.environ.get("H3_OUTPUT") != str(root) or Path(__file__).resolve().parent != p.CODE_ROOT):
        raise RuntimeError("Noncanonical C run/output/deployment")
    for name in ("host_identity.json", "source_manifest.json"):
        proof = p.canonical(root / name)
        if proof.is_symlink() or not proof.is_file():
            raise RuntimeError("Missing/nonregular proof")
    args.host = p.require_host(json.loads((root / "host_identity.json").read_text()))
    manifest = p.source_manifest()
    if manifest != json.loads((root / "source_manifest.json").read_text()):
        raise RuntimeError("C source manifest changed since preflight")
    verify_parent_leases(root, args.run_id, args.host, p.load_helper("supervision_base.py"))
    for path in (args.output, *(root / f"c_tiny.rank{r}.json" for r in range(WORLD))):
        p.canonical(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", p.PORT))
    return args, manifest


def worker(rank, args, manifest):
    p.require_host(args["host"])
    if (p.source_manifest() != manifest or os.environ.get("H3_MINIMAL_RUN_ID") != args["run_id"]
            or args["physical_cards"] != list(p.CARDS)
            or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"):
        raise RuntimeError("Worker identity/source/card mismatch before runtime initialization")
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(p.PORT))
    import torch
    import torch_npu
    import torch.distributed as dist
    torch.set_num_threads(1)
    report = dict(status="running", rank=rank, physical_card=args["physical_cards"][rank],
                  host=args["host"], run_id=args["run_id"], case=p.CASE, source_sha256=manifest, tests=[])
    initialized = False
    began = time.perf_counter()
    try:
        # Load only adapter/layout into a private package for CPU fail-closed
        # checks. This does not instantiate the transformer or initialize NPU.
        pkg = types.ModuleType("_c_precollective")
        pkg.__path__ = [str(p.VENDOR / "vllm_omni/diffusion/models/minimax_h3")]
        sys.modules[pkg.__name__] = pkg
        for name in ("openvdn_npu", "strict_source_layout", "strict_source_attention"):
            early = p.load_file(pkg.__name__ + "." + name, Path(manifest[f"candidate/{name}.py"]["path"]))
        report["precollective_metadata"] = metadata_checks(early, torch)
        if dist.is_initialized() or torch.npu.is_initialized():
            raise RuntimeError("Metadata rejection must finish before any NPU/collective initialization")
        torch.npu.set_device(rank)
        if torch.npu.device_count() != WORLD:
            raise RuntimeError("Expected exactly eight visible NPUs")
        from vllm_omni.diffusion.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel, get_sp_group,
            init_distributed_environment, initialize_model_parallel)
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.attention.layer import Attention
        from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
        from vllm_omni.diffusion.attention.parallel.factory import build_parallel_attention_strategy
        from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
        dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{p.PORT}", rank=rank, world_size=WORLD,
                                timeout=datetime.timedelta(seconds=180))
        init_distributed_environment(world_size=WORLD, rank=rank, local_rank=rank, backend="hccl",
                                     distributed_init_method=f"tcp://127.0.0.1:{p.PORT}")
        initialize_model_parallel(sequence_parallel_size=WORLD, ulysses_degree=WORLD, ring_degree=1,
                                  tensor_parallel_size=1, backend="hccl")
        initialized = True
        sp = get_sp_group()
        if sp.ulysses_world_size != WORLD or sp.ulysses_rank != rank or sp.ring_world_size != 1 or get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("Not one full USP8 / ring1 / TP1 group")
        rt = torch.tensor([rank], dtype=torch.int32, device=f"npu:{rank}")
        all_ranks = [torch.empty_like(rt) for _ in range(WORLD)]
        dist.all_gather(all_ranks, rt, group=sp.ulysses_group)
        peers = [int(v.cpu().item()) for v in all_ranks]
        if peers != list(range(WORLD)):
            raise RuntimeError("Incomplete eight-rank collective")
        report["parallelism_proof"] = dict(ulysses_world_size=WORLD, ulysses_rank=rank, ring_world_size=1,
                    tensor_parallel_world_size=1, collective_global_ranks=peers, heads_per_ulysses_rank=7)
        strategy_path = Path(sys.modules[UlyssesParallelAttention.__module__].__file__).resolve()
        if dict(path=str(strategy_path), sha256=p.digest(strategy_path)) != manifest["strategy/ulysses.py"]:
            raise RuntimeError("Wrong actual Ulysses implementation")
        report["strategy_proof"] = manifest["strategy/ulysses.py"]
        helper = p.load_helper("npu_wrapper_regression.py")
        upstream = p.load_file("_c_golden_b_linear", Path(manifest["upstream/openvdn_npu.py"]["path"]))
        prefix = "vllm_omni.diffusion.models.minimax_h3"
        # Import each actual file under its real package so C relative imports
        # bind to the same StrictSourceLayout class, not the early CPU copy.
        importlib.import_module(prefix)
        branch = p.load_file(prefix + ".openvdn_npu", Path(manifest["candidate/openvdn_npu.py"]["path"]))
        p.load_file(prefix + ".strict_source_layout", Path(manifest["candidate/strict_source_layout.py"]["path"]))
        adapter = p.load_file(prefix + ".strict_source_attention", Path(manifest["candidate/strict_source_attention.py"]["path"]))
        candidate = p.load_file(prefix + ".minimax_h3_transformer", Path(manifest["candidate/minimax_h3_transformer.py"]["path"]))
        cfg = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=WORLD, ulysses_degree=WORLD,
                              ring_degree=1, allgather_degree=1, ulysses_mode="strict"))
        with set_forward_context(omni_diffusion_config=cfg), torch.inference_mode():
            generator = torch.Generator(device="cpu").manual_seed(8731)
            holder = helper.build_holder(candidate, branch, Attention, build_parallel_attention_strategy,
                                         NoParallelAttention, torch, generator).eval().to(f"npu:{rank}")
            frozen_weights = {k: v.detach().cpu().clone() for k, v in holder.state_dict().items()}
            linear = upstream.BidirectionalLinearBranch(HIDDEN, HEADS, DIM)
            linear.load_state_dict(holder.linear_attention.state_dict(), strict=True)
            linear = linear.eval().to(f"npu:{rank}")
            strategy = holder.attention._get_active_parallel_strategy()
            if not isinstance(strategy, UlyssesParallelAttention) or strategy.name != "ulysses":
                raise RuntimeError("Actual attention did not select Ulysses")
            for case in CASES:
                data = fixture(case)
                layout = adapter.infer_strict_source_layout(**tensors(data, torch))
                n = len(data["coords"])
                x = (torch.randn((n, HIDDEN), generator=generator) * 0.3).to(device=f"npu:{rank}", dtype=torch.bfloat16)
                positions = torch.tensor(data["coords"], dtype=torch.float32).unsqueeze(0).to(x.device)
                rope_module = candidate.MiniMaxH3Rope(inv_freq_len=16)
                rope_module.inv_freq.copy_(10000.0 ** -(torch.arange(0, 32, 2, dtype=torch.float32) / 32))
                rope = rope_module.to(x.device)(positions)
                cu = torch.tensor(data["cu_seqlens"], device=x.device, dtype=torch.int32)
                rows, lo = n // WORLD, rank * (n // WORLD)
                hi = lo + rows
                local_x, local_rope = x[lo:hi].contiguous(), rope[lo:hi].contiguous()
                raw, qkv = helper.prepare_qkv(holder, x, rope, candidate)
                local_raw, local_qkv = helper.prepare_qkv(holder, local_x, local_rope, candidate)
                frozen_qkv = [v.clone() for v in (*raw, *qkv)]
                gold_soft = dense_cpu(*qkv, layout, torch)
                gold_linear = torch.zeros((n, HEADS * DIM), device=x.device, dtype=x.dtype)
                gold_linear[layout.video_start:layout.video_end] = linear(x, raw, layout)
                gold_out = holder.out_proj((gold_soft * holder.softmax_gate(x)).flatten(1))[0] + holder.to_out_linear(gold_linear)
                got_soft, got_linear = holder._run_openvdn_ulysses(local_x, *local_qkv, local_raw, layout, cu, strategy)
                actual = holder(local_x, rope_freqs=local_rope, cu_seqlens=cu, max_seqlen=layout.used_len, openvdn_layout=layout)
                checks = [helper.compare("c-wrapper-softmax", got_soft, gold_soft[lo:hi], torch),
                          helper.compare("unchanged-b-linear", got_linear, gold_linear[lo:hi], torch),
                          helper.compare("c-actual-forward", actual, gold_out[lo:hi], torch)]
                r = torch.arange(lo, hi, device=x.device)
                linear_zero = (r < layout.video_start + layout.tokens_per_frame) | (r >= layout.video_end - layout.tokens_per_frame)
                checks.append(helper.compare("linear-nontarget-anchor-padding-zero", got_linear[linear_zero],
                                             torch.zeros_like(got_linear[linear_zero]), torch, exact=True))
                checks.append(helper.compare("padding-output-zero", actual[r >= layout.used_len],
                                             torch.zeros_like(actual[r >= layout.used_len]), torch, exact=True))
                # Non-corresponding source values must never leak even into
                # first/last target anchors. Real SP pre/post exchange executes.
                zq, zk, zv = (torch.zeros_like(qkv[0]) for _ in range(3))
                zv[layout.source_start + layout.source_tokens_per_frame] = 8192
                qh, kh, vh, _, context = strategy.pre_attention(zq[lo:hi].unsqueeze(0), zk[lo:hi].unsqueeze(0),
                                                               zv[lo:hi].unsqueeze(0), None)
                leak_head = holder._openvdn_softmax(qh[0], kh[0], vh[0], layout)
                leak = strategy.post_attention(leak_head.unsqueeze(0), context).squeeze(0)
                leak_gold = dense_cpu(zq, zk, zv, layout, torch)
                checks.append(helper.compare("wrong-source-dense-oracle", leak, leak_gold[lo:hi], torch))
                first_or_last = ((r >= layout.video_start) & (r < layout.video_start + layout.tokens_per_frame)) | (
                    (r >= layout.video_end - layout.tokens_per_frame) & (r < layout.video_end))
                checks.append(helper.compare("wrong-source-anchor-zero", leak[first_or_last],
                                             torch.zeros_like(leak[first_or_last]), torch, exact=True))
                # Positive controls ensure zero outputs cannot trivially pass:
                # same-frame target, source, text, both audio blocks see V>0.
                positive = [layout.video_start + layout.tokens_per_frame, layout.source_start,
                            0, data["audio_pos"][0], data["audio_pos"][-1]]
                for row in positive:
                    if lo <= row < hi and not bool((leak[row - lo] > 0).all()):
                        raise AssertionError("C wrongly hid a preserved text/audio/source/same-frame edge")
                if not all(torch.equal(a, b) for a, b in zip((*raw, *qkv), frozen_qkv)):
                    raise AssertionError("C mutated frozen QKV")
                torch.npu.synchronize()
                report["tests"].append(dict(case=case, packed_rows=n, local_rows=[lo, hi], layout=vars(layout),
                    checks=checks, mask_proof=audit_masks(layout, n), frozen_qkv_unchanged=True,
                    positive_controls=True, source_is_visual_only=True))
                print(json.dumps(dict(rank=rank, case=case, status="passed")), flush=True)
            if any(not torch.equal(v.detach().cpu(), frozen_weights[k]) for k, v in holder.state_dict().items()):
                raise AssertionError("C changed weights/QKV/RoPE/gates/linear state")
            report["frozen_weights_unchanged"] = True
            dist.barrier(group=sp.ulysses_group)
        report.update(status="passed", torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        write_new(Path(args["output"]).with_name(f"c_tiny.rank{rank}.json"), report)
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
        elif dist.is_initialized():
            dist.destroy_process_group()


def validate_report(report, host, run_id, manifest):
    expected = dict(status="passed", case=p.CASE, host=host, run_id=run_id, world_size=WORLD,
                    physical_cards=list(p.CARDS), dtype="bfloat16", heads=HEADS, head_dim=DIM,
                    source_sha256=manifest, test_script_sha256=manifest["validation/c_npu_regression.py"]["sha256"])
    if any(report.get(k) != v for k, v in expected.items()):
        raise RuntimeError("C summary identity/source/full56head mismatch")
    ranks = report.get("rank_results", [])
    if len(ranks) != WORLD or {r.get("rank") for r in ranks} != set(range(WORLD)):
        raise RuntimeError("Expected exactly eight distinct C rank records")
    for item in ranks:
        rank = item["rank"]
        if (type(rank) is not int or item.get("status") != "passed" or item.get("case") != p.CASE
                or item.get("physical_card") != p.CARDS[rank] or item.get("host") != host
                or item.get("run_id") != run_id or item.get("source_sha256") != manifest
                or item.get("frozen_weights_unchanged") is not True
                or item.get("strategy_proof") != manifest["strategy/ulysses.py"]
                or item.get("parallelism_proof") != dict(ulysses_world_size=WORLD, ulysses_rank=rank,
                    ring_world_size=1, tensor_parallel_world_size=1, collective_global_ranks=list(range(WORLD)), heads_per_ulysses_rank=7)):
            raise RuntimeError("C rank lacks identity/frozen-math/full SP8 proof")
        if set(item.get("precollective_metadata", {})) != set(CASES):
            raise RuntimeError("Missing precollective metadata cases")
        tests = item.get("tests", [])
        if len(tests) != len(CASES) or {t.get("case") for t in tests} != set(CASES):
            raise RuntimeError("Missing legitimate C prefix layout tests")
        for test in tests:
            packed = len(fixture(test["case"])["coords"])
            expected_geometry = geometry(test["case"])
            if (test.get("packed_rows") != packed or test.get("local_rows") != [rank * (packed // WORLD), (rank + 1) * (packed // WORLD)]
                    or any(test.get("layout", {}).get(k) != v for k, v in vars(expected_geometry).items()
                           if k not in ("source_end", "video_end"))
                    or test.get("mask_proof") != audit_masks(expected_geometry, packed)
                    or item["precollective_metadata"][test["case"]].get("oracle") != audit_masks(expected_geometry, packed)):
                raise RuntimeError("C layout/SP8 row-boundary/oracle proof mismatch")
            if (set(item["precollective_metadata"][test["case"]].get("rejected", [])) != set(REJECTIONS)
                    or any(test.get(k) is not True for k in ("frozen_qkv_unchanged", "positive_controls", "source_is_visual_only"))):
                raise RuntimeError("Missing fail-closed or preservation checks")
            checks = test.get("checks", [])
            if len(checks) != len(CHECKS) or {c.get("check") for c in checks} != CHECKS:
                raise RuntimeError("Missing C numeric check")
            for check in checks:
                exact = check["check"].endswith("-zero")
                if any(type(check.get(k)) not in (int, float) or not math.isfinite(check[k]) or check[k] < 0
                       for k in ("atol", "rtol", "max_abs", "rms_error", "golden_rms")):
                    raise RuntimeError("Invalid C metric")
                if (check["atol"], check["rtol"]) != ((0.0, 0.0) if exact else (1 / 512, 1 / 128)):
                    raise RuntimeError("C tolerance was changed")
                if exact and (check["max_abs"] != 0 or check["rms_error"] != 0):
                    raise RuntimeError("Exact C zero check failed")
    return report


def verify_results(path, host, run_id, manifest):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing/nonregular C result")
    report = validate_report(json.loads(path.read_text()), host, run_id, manifest)
    for item in report["rank_results"]:
        rank_path = path.with_name(f"c_tiny.rank{item['rank']}.json")
        if rank_path.is_symlink() or json.loads(rank_path.read_text()) != item:
            raise RuntimeError("Embedded C rank evidence differs from rank file")
    return report


def main(argv=None):
    args, manifest = parse_and_validate(argv)
    report = dict(status="running", case=p.CASE, host=args.host, run_id=args.run_id, world_size=WORLD,
                  physical_cards=list(p.CARDS), dtype="bfloat16", heads=HEADS, head_dim=DIM,
                  source_sha256=manifest, test_script_sha256=p.digest(__file__),
                  limitations=["Synthetic weights only; no checkpoint/service/full DiT/pipeline/quality/speed claim.",
                               "Small tuple-returning projection stubs; real C attention/Ulysses/HCCL/NPU kernels.",
                               "Source suffix is unsupported and must reject, not a successful C layout.",
                               "Metadata is replicated, not a protocol to recover divergent worker inputs."])
    began = time.perf_counter()
    import torch.multiprocessing as mp
    try:
        worker_args = dict(output=str(args.output), run_id=args.run_id, host=args.host, physical_cards=list(p.CARDS))
        mp.spawn(worker, args=(worker_args, manifest), nprocs=WORLD, join=True)
        p.require_host(args.host)
        if p.source_manifest() != manifest:
            raise RuntimeError("C source changed during validation")
        report.update(status="passed", rank_results=[json.loads(args.output.with_name(f"c_tiny.rank{r}.json").read_text()) for r in range(WORLD)])
        validate_report(report, args.host, args.run_id, manifest)
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        write_new(args.output, report)


if __name__ == "__main__":
    main()
