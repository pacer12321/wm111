"""Minimal isolated DLO prepared-shard seam; no pipeline/backend enable().

AllGather, prefetch, slot rotation and repointing remain the upstream hook's
methods. Only _shard_and_pin is replaced by validated ownership transfer of
already-final rank-local shards. All blocks must get empty CPU placeholders
BEFORE any hook caches parameter references. No non-DiT module is touched.
"""
from dataclasses import dataclass, field
import math


@dataclass
class PreparedBlock:
    block_name: str
    world_size: int
    rank: int
    cpu_shards: dict
    metadata: dict
    provenance: dict = field(default_factory=dict)
    claimed: bool = False


def canonical_metadata(entries):
    """Exactly params-then-buffers order, grouped by dtype first occurrence."""
    grouped, offsets, names = {}, {}, set()
    for entry in entries:
        name, dtype, shape = entry["name"], entry["dtype"], tuple(entry["shape"])
        if name in names or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("Duplicate name or invalid shape")
        names.add(name)
        offset = offsets.get(dtype, 0)
        count = math.prod(shape)
        grouped.setdefault(dtype, []).append(dict(name=name, offset=offset, numel=count, shape=shape))
        offsets[dtype] = offset + count
    return grouped


def validate_prepared(entries, prepared, *, require_pinned):
    if prepared.world_size < 1 or not 0 <= prepared.rank < prepared.world_size:
        raise ValueError("Invalid rank ownership")
    if prepared.claimed:
        raise ValueError("Prepared block has already been claimed")
    expected = canonical_metadata(entries)
    if list(expected) != list(prepared.metadata) or list(expected) != list(prepared.cpu_shards):
        raise ValueError("Dtype group set/order mismatch")
    for dtype, items in expected.items():
        actual = [dict(x, shape=tuple(x["shape"])) for x in prepared.metadata[dtype]]
        if actual != items:
            raise ValueError("Parameter/buffer order, offset, shape or coverage mismatch")
        count = sum(x["numel"] for x in items)
        expected_size = (count + prepared.world_size - 1) // prepared.world_size
        shard = prepared.cpu_shards[dtype]
        if shard.device.type != "cpu" or shard.dtype != dtype or shard.ndim != 1 or shard.numel() != expected_size:
            raise ValueError("Shard device, dtype, rank-local size or shape mismatch")
        if not shard.is_contiguous():
            raise ValueError("Shard must be contiguous")
        if require_pinned and not shard.is_pinned():
            raise ValueError("Pinned shard required; implicit pinning/copying is forbidden")
    return expected


def bundle_from_streaming(block_name, world_size, rank, shards, metadata, *, provenance=None):
    """Convert string dtype/full manifest names to native hook conventions."""
    import torch
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    prefix = block_name + "."
    native_shards, native_metadata = {}, {}
    for dtype, shard in shards.items():
        native = dtypes[dtype]
        native_shards[native] = shard
        entries = []
        for item in metadata[dtype]:
            if not item["name"].startswith(prefix):
                raise ValueError("Streaming tensor does not belong to this main block")
            entries.append(dict(item, name=item["name"][len(prefix):], shape=tuple(item["shape"])))
        native_metadata[native] = entries
    return PreparedBlock(block_name, world_size, rank, native_shards, native_metadata, provenance or {})


def _block_entries(block):
    entries = []
    for name, tensor in list(block.named_parameters()) + list(block.named_buffers()):
        entries.append(dict(name=name, dtype=tensor.dtype, shape=tuple(tensor.shape)))
    return entries


def prepare_all_meta_blocks(blocks, bundles, *, require_pinned=True):
    """Validate every block, then install zero-element CPU placeholders.

    Never call module.to_empty('cpu'): that would allocate a full block.
    Replaces Parameter objects like upstream's mmap loader, preserving their
    Python attrs. This is forward-only/unquantized; weight loaders must never
    be reused after handoff. Tied tensors/previous hooks are rejected.
    """
    import torch
    blocks, bundles = list(blocks), list(bundles)
    if not blocks or len(blocks) != len(bundles):
        raise ValueError("One prepared bundle is required per main block")
    schemas, seen_tensors = [], set()
    for index, (block, bundle) in enumerate(zip(blocks, bundles)):
        if bundle.block_name != f"blocks.{index}":
            raise ValueError("Main-block bundle order mismatch")
        if hasattr(block, "_hook_registry") or hasattr(block, "_prepared_offload_schema"):
            raise ValueError("Block already has hooks/prepared state")
        all_tensors = list(block.named_parameters(remove_duplicate=False)) + list(block.named_buffers(remove_duplicate=False))
        for name, tensor in all_tensors:
            if tensor.device.type != "meta" or id(tensor) in seen_tensors:
                raise ValueError("Require untouched meta blocks without tied tensors")
            seen_tensors.add(id(tensor))
        entries = _block_entries(block)
        validate_prepared(entries, bundle, require_pinned=require_pinned)
        schemas.append(entries)
    # No mutations before all bundles above validate successfully.
    for block, bundle, entries in zip(blocks, bundles, schemas):
        for kind, members in (("parameter", list(block.named_parameters())), ("buffer", list(block.named_buffers()))):
            for name, old in members:
                owner, _, local_name = name.rpartition(".")
                parent = block.get_submodule(owner) if owner else block
                with torch.inference_mode(False):
                    empty = torch.empty((0,), device="cpu", dtype=old.dtype)
                if kind == "parameter":
                    with torch.inference_mode(False):
                        replacement = torch.nn.Parameter(empty, requires_grad=old.requires_grad)
                    replacement.__dict__.update(old.__dict__)
                    parent._parameters[local_name] = replacement
                else:
                    parent._buffers[local_name] = empty
        block._prepared_offload_schema = entries
        block._prepared_offload_bundle = bundle
        block._prepared_offload_tensor_ids = {
            name: id(tensor) for name, tensor in list(block.named_parameters()) + list(block.named_buffers())
        }


def validate_placeholder_block(block):
    if not hasattr(block, "_prepared_offload_schema"):
        raise ValueError("All blocks must be prepared before any hook registration")
    current = dict(list(block.named_parameters()) + list(block.named_buffers()))
    if set(current) != set(block._prepared_offload_tensor_ids):
        raise ValueError("Prepared block parameter/buffer set changed")
    for name, tensor in current.items():
        if id(tensor) != block._prepared_offload_tensor_ids[name] or tensor.device.type != "cpu" or tensor.numel() != 0:
            raise ValueError("Stale parameter identity or nonempty/non-CPU placeholder")


def make_prepared_hook_class(native_hook_class=None):
    """Injectable native class enables CPU protocol tests without CUDA imports."""
    if native_hook_class is None:
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadHook
        native_hook_class = DistributedLayerwiseOffloadHook

    class PreparedDistributedLayerwiseOffloadHook(native_hook_class):
        def __init__(self, *args, prepared, **kwargs):
            self.prepared = prepared
            super().__init__(*args, **kwargs)

        def initialize_hook(self, module):
            validate_placeholder_block(module)
            validate_placeholder_block(self.next_block)
            if self.next_block._prepared_offload_bundle is not self.prepared:
                raise ValueError("Hook bundle does not belong to next block")
            validate_prepared(self.next_block._prepared_offload_schema, self.prepared, require_pinned=self.pin_memory)
            # Native initialize caches refs, metadata repoint table, and
            # AllGather sizes. Its _shard_and_pin dispatches to our override.
            return super().initialize_hook(module)

        def _shard_and_pin(self, params, bufs, dp_size, rank, pin_memory):
            if dp_size != self.prepared.world_size or rank != self.prepared.rank:
                raise ValueError("Hook group/rank differs from prepared shard ownership")
            validate_prepared(self.next_block._prepared_offload_schema, self.prepared, require_pinned=pin_memory)
            supplied = dict(list(params.items()) + list(bufs.items()))
            if {name: id(tensor) for name, tensor in supplied.items()} != self.next_block._prepared_offload_tensor_ids:
                raise ValueError("Native hook captured stale parameter references")
            self.prepared.claimed = True
            # Zero-copy references: no flatten, full allocation, copy_,
            # pin_memory(), checkpoint loading or _shard_and_pin superclass.
            return self.prepared.cpu_shards, self.prepared.metadata

    return PreparedDistributedLayerwiseOffloadHook


def register_prepared_hook(module, next_block, *, prepared, device, dp_group, dp_size, rank,
                           copy_stream, comm_stream, pin_memory=True, shared_buffers=None):
    """A replacement only for apply_distributed_block_hook, not backend.enable."""
    from vllm_omni.diffusion.hooks import HookRegistry
    hook_class = make_prepared_hook_class()
    hook = hook_class(next_block=next_block, device=device, dp_group=dp_group, dp_size=dp_size, rank=rank,
                      copy_stream=copy_stream, comm_stream=comm_stream, pin_memory=pin_memory,
                      shared_buffers=shared_buffers, prepared=prepared)
    HookRegistry.get_or_create(module).register_hook(hook._HOOK_NAME, hook)
    return hook
