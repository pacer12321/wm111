"""CPU protocol mocks: never imports torch, vLLM or GPU libraries."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from prepared_shard_hook import PreparedBlock, canonical_metadata, make_prepared_hook_class, validate_prepared


class Tensor:
    def __init__(self, size, dtype="BF16", *, pinned=False):
        self.size, self.dtype, self.ndim = size, dtype, 1
        self.device = SimpleNamespace(type="cpu")
        self.pinned = pinned
    def numel(self): return self.size
    def is_contiguous(self): return True
    def is_pinned(self): return self.pinned


class Block:
    def __init__(self, bundle, entries):
        self.tensors = {entry["name"]: Tensor(0, entry["dtype"]) for entry in entries}
        self._prepared_offload_schema = entries
        self._prepared_offload_bundle = bundle
        self._prepared_offload_tensor_ids = {k:id(v) for k,v in self.tensors.items()}
    def named_parameters(self): return iter(self.tensors.items())
    def named_buffers(self): return iter(())


class NativeHookMock:
    """Mimics only native initialize seam; real runtime test is separate."""
    _HOOK_NAME = "distributed_layerwise_offload"
    def __init__(self, next_block, dp_size, rank, pin_memory=False, **kwargs):
        self.next_block, self.dp_size, self.rank, self.pin_memory = next_block, dp_size, rank, pin_memory
    def initialize_hook(self, module):
        self.block_parameters = dict(module.named_parameters())
        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.cpu_shards, self.metadata = self._shard_and_pin(self.next_block_parameters, {}, self.dp_size, self.rank, self.pin_memory)
        self.cached_targets = list(self.next_block_parameters.values())
        return module
    def _shard_and_pin(self, *args):
        raise AssertionError("Original full-parameter sharding must never run")


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.entries = [dict(name="qkv.weight", dtype="BF16", shape=(3, 2)),
                        dict(name="branch.weight", dtype="BF16", shape=(3,))]
        self.bundle = PreparedBlock("blocks.1", 2, 0, {"BF16":Tensor(5)}, canonical_metadata(self.entries))

    def test_zero_copy_handoff_bypasses_original_sharding(self):
        hook_cls = make_prepared_hook_class(NativeHookMock)
        block = Block(self.bundle, self.entries)
        hook = hook_cls(next_block=block, dp_size=2, rank=0, prepared=self.bundle)
        hook.initialize_hook(block)
        self.assertIs(hook.cpu_shards, self.bundle.cpu_shards)
        self.assertIs(hook.cpu_shards["BF16"], self.bundle.cpu_shards["BF16"])
        self.assertIs(hook.metadata, self.bundle.metadata)
        self.assertTrue(self.bundle.claimed)
        self.assertEqual(hook.cached_targets, list(block.tensors.values()))

    def test_duplicate_handoff_rejected(self):
        cls = make_prepared_hook_class(NativeHookMock)
        block = Block(self.bundle, self.entries)
        cls(next_block=block, dp_size=2, rank=0, prepared=self.bundle).initialize_hook(block)
        with self.assertRaisesRegex(ValueError, "already been claimed"):
            cls(next_block=block, dp_size=2, rank=0, prepared=self.bundle).initialize_hook(block)

    def test_rank_mismatch_rejected(self):
        cls = make_prepared_hook_class(NativeHookMock)
        block = Block(self.bundle, self.entries)
        with self.assertRaisesRegex(ValueError, "group/rank"):
            cls(next_block=block, dp_size=2, rank=1, prepared=self.bundle).initialize_hook(block)
        self.assertFalse(self.bundle.claimed)

    def test_no_implicit_pinning(self):
        with self.assertRaisesRegex(ValueError, "Pinned shard required"):
            validate_prepared(self.entries, self.bundle, require_pinned=True)

    def test_stale_meta_replacement_reference_rejected(self):
        cls = make_prepared_hook_class(NativeHookMock)
        block = Block(self.bundle, self.entries)
        block.tensors["qkv.weight"] = Tensor(0)
        with self.assertRaisesRegex(ValueError, "Stale parameter"):
            cls(next_block=block, dp_size=2, rank=0, prepared=self.bundle).initialize_hook(block)

    def test_metadata_order_and_offsets_rejected(self):
        self.bundle.metadata["BF16"].reverse()
        with self.assertRaisesRegex(ValueError, "order, offset"):
            validate_prepared(self.entries, self.bundle, require_pinned=False)

    def test_full_weights_not_accepted_as_shard(self):
        self.bundle.cpu_shards["BF16"] = Tensor(9)
        with self.assertRaisesRegex(ValueError, "rank-local size"):
            validate_prepared(self.entries, self.bundle, require_pinned=False)

    def test_missing_dtype_or_parameter_rejected(self):
        self.bundle.metadata["BF16"].pop()
        with self.assertRaises(ValueError):
            validate_prepared(self.entries, self.bundle, require_pinned=False)

    def test_mixed_dtype_offsets_reset(self):
        entries = self.entries + [dict(name="buffer", dtype="F32", shape=(2,))]
        result = canonical_metadata(entries)
        self.assertEqual(result["F32"][0]["offset"], 0)
        self.assertEqual(result["BF16"][1]["offset"], 6)

    def test_placeholder_must_be_empty_cpu(self):
        cls = make_prepared_hook_class(NativeHookMock)
        block = Block(self.bundle, self.entries)
        block.tensors["qkv.weight"].size = 6
        with self.assertRaisesRegex(ValueError, "nonempty/non-CPU"):
            cls(next_block=block, dp_size=2, rank=0, prepared=self.bundle).initialize_hook(block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
