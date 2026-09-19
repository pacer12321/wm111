"""Small two-rank CUDA/NCCL native prepared-ring FORWARD correctness gate.

No checkpoint, H3 inference, training, task cleanup, or mocks. Run only after
resource admission using torchrun --nproc-per-node=2; never a speed result.
Native async prefetch, stream events, hooks and collectives remain unchanged.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from h3_prepared_integration import ring_hook_plan
from prepared_shard_hook import PreparedBlock, canonical_metadata, prepare_all_meta_blocks, register_prepared_hook
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend, remove_distributed_block_hook,
)


@torch.inference_mode()
def run_case(count, rounds, rank, device, width=128):
    # Small, identical shapes match H3's 50 uniform main blocks. Values and
    # inputs are deterministic exact BF16 fractions, independent of RNG state.
    dtype = torch.bfloat16
    flat_index = torch.arange(width * width, dtype=torch.float32)
    full_cpu = [((flat_index + i * 3).remainder(31) - 15).div(64).to(dtype) for i in range(count)]
    with torch.device("meta"):
        blocks = [torch.nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(count)]
    references = []
    for full in full_cpu:
        layer = torch.nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        layer.weight.copy_(full.reshape(width, width).to(device))
        references.append(layer)
    entries = [dict(name="weight", dtype=dtype, shape=(width, width))]
    shard_size = width * width // 2
    bundles = []
    for index, full in enumerate(full_cpu):
        shard = torch.empty(shard_size, dtype=dtype, pin_memory=True)
        shard.copy_(full[rank * shard_size:(rank + 1) * shard_size])
        bundles.append(PreparedBlock(f"blocks.{index}", 2, rank, {dtype:shard}, canonical_metadata(entries)))
    prepare_all_meta_blocks(blocks, bundles, require_pinned=True)
    hooks = []
    copy_stream, comm_stream = torch.cuda.Stream(), torch.cuda.Stream()
    try:
        for current, following in ring_hook_plan(count):
            hooks.append(register_prepared_hook(blocks[current], blocks[following], prepared=bundles[following],
                device=device, dp_group=dist.group.WORLD, dp_size=2, rank=rank,
                copy_stream=copy_stream, comm_stream=comm_stream, pin_memory=True, shared_buffers=[None, None]))
        outputs = DistributedLayerwiseOffloadBackend._allocate_shared_buffers(hooks)
        inputs = DistributedLayerwiseOffloadBackend._allocate_shared_shard_buffers(hooks)
        slot_groups = [-1, -1]
        for index, hook in enumerate(hooks):
            hook._prev_hook = hooks[index - 1]
            hook.current_slot = index % 2
            hook.gpu_buffers, hook.gpu_shard_buffers = outputs, inputs
            hook._owns_buffers = False
            hook._group_id, hook._shared_slot_group = 0, slot_groups
        hooks[1]._is_group_first = True
        # EXACT integration/native initial order: last block initially primed,
        # followed by the first hook's native group-first synchronous fetch.
        hooks[-1].prefetch_layer(slot=hooks[0].current_slot, non_blocking=False)
        hooks[-1].get_weights(hooks[0].current_slot)
        fixed_input = ((torch.arange(8 * width, device=device).remainder(13) - 6).float() / 32).to(dtype).reshape(8, width)
        checks = []
        # No synchronize/item/cpu in this loop: preserve real overlap and let
        # native post-forward release/reuse storage using its own events.
        for _ in range(rounds):
            for block, reference in zip(blocks, references):
                actual = block(fixed_input)
                expected = reference(fixed_input)
                checks.append(torch.eq(actual, expected).all())
        passed = torch.stack(checks).cpu().tolist()
        failed = [dict(round=i // count, layer=i % count) for i, ok in enumerate(passed) if not ok]
        if failed:
            raise AssertionError(f"Native asynchronous forward mismatch: {failed[:8]}")
        pointers = dict(full=[slot[dtype].data_ptr() for slot in outputs],
                        half=[slot[dtype].data_ptr() for slot in inputs])
        if len(set(pointers["full"] + pointers["half"])) != 4:
            raise AssertionError("Expected two distinct full and two distinct half GPU allocations")
        return dict(blocks=count, rounds=rounds, checked_outputs=len(checks), bitwise_equal=True,
                    buffer_bytes=sum(x[dtype].numel() * x[dtype].element_size() for x in outputs + inputs),
                    device_buffer_pointers=pointers)
    finally:
        torch.cuda.synchronize()
        for block in blocks:
            remove_distributed_block_hook(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    if not 2 <= args.rounds <= 100:
        raise ValueError("Use 2..100 bounded rounds")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or rank != local_rank or rank not in (0, 1):
        raise ValueError("Single-node, exactly two ranks required")
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Hard bound on allocations owned by PyTorch, not CUDA/NCCL context memory.
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, (1 << 30) / total), device)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"ring_rank{rank}.json"
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    result = dict(status="running", rank=rank, cases=[], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="small BF16 Linear native async CUDA/NCCL ring; not H3 E2E, quality or timing")
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    try:
        for count in (3, 4, 50):
            result["cases"].append(run_case(count, args.rounds, rank, device))
            print(json.dumps(dict(rank=rank, case=result["cases"][-1])), flush=True)
        dist.barrier()
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
