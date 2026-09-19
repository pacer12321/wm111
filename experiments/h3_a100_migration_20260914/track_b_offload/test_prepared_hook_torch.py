"""Real torch/native-hook test with CPU buffers and fake collectives only.

This validates the seam/aliasing, NOT CUDA AllGather or async correctness.
Requires installed vLLM/vllm-omni; no GPU or process group is initialized.
"""
from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "vllm_omni"))
if HAS_RUNTIME:
    import torch
    from prepared_shard_hook import PreparedBlock, canonical_metadata, make_prepared_hook_class, prepare_all_meta_blocks
    from vllm_omni.diffusion.offloader import distributed_layerwise_backend as native


@unittest.skipUnless(HAS_RUNTIME, "torch/vllm-omni unavailable: real hook CPU seam NOT executed")
class NativePreparedTests(unittest.TestCase):
    def test_native_circular_forward_even_odd_and_full_depth(self):
        """Actual pre/post-forward methods, two turns and shared input/output slots.

        Fake collectives execute synchronously on CPU: this is a ring ownership
        gate, not evidence of CUDA stream or NCCL correctness.
        """
        from h3_prepared_integration import ring_hook_plan
        class Stream:
            def wait_stream(self, other): pass
            def wait_event(self, event): pass
        class Event:
            def record(self, stream): pass
        stream = Stream()
        platform = SimpleNamespace(current_stream=lambda:stream, stream=lambda s:nullcontext(), Event=Event)
        hook_cls = make_prepared_hook_class(native.DistributedLayerwiseOffloadHook)
        for count in (3, 4, 50):
            with self.subTest(block_count=count), torch.device("meta"):
                blocks = [torch.nn.Linear(2, 2, bias=False, dtype=torch.float32) for _ in range(count)]
            full = [torch.arange(4, dtype=torch.float32) + i * 10 for i in range(count)]
            entries = [dict(name="weight", dtype=torch.float32, shape=(2, 2))]
            bundles = [PreparedBlock(f"blocks.{i}", 2, 0, {torch.float32:full[i][:2].clone()},
                                     canonical_metadata(entries)) for i in range(count)]
            prepare_all_meta_blocks(blocks, bundles, require_pinned=False)
            def gather(output, local, group):
                output.copy_(torch.cat((local, local + 2)))
            with patch.object(native, "current_omni_platform", platform), \
                 patch.object(native.DistributedLayerwiseOffloadHook, "_shard_and_pin", side_effect=AssertionError("full sharding forbidden")), \
                 patch.object(torch.distributed, "all_gather_into_tensor", side_effect=gather):
                hooks = []
                by_block = {}
                for current, following in ring_hook_plan(count):
                    hook = hook_cls(next_block=blocks[following], device=torch.device("cpu"), dp_group=object(),
                                    dp_size=2, rank=0, copy_stream=stream, comm_stream=stream, pin_memory=False,
                                    shared_buffers=[None, None], prepared=bundles[following])
                    hook.initialize_hook(blocks[current])
                    hooks.append(hook)
                    by_block[current] = hook
                outputs = native.DistributedLayerwiseOffloadBackend._allocate_shared_buffers(hooks)
                inputs = native.DistributedLayerwiseOffloadBackend._allocate_shared_shard_buffers(hooks)
                slot_groups = [-1, -1]
                for index, hook in enumerate(hooks):
                    hook._prev_hook, hook.current_slot = hooks[index - 1], index % 2
                    hook.gpu_buffers, hook.gpu_shard_buffers = outputs, inputs
                    hook._group_id, hook._shared_slot_group = 0, slot_groups
                hooks[1]._is_group_first = True
                hooks[-1].prefetch_layer(hooks[0].current_slot, non_blocking=False)
                hooks[-1].get_weights(hooks[0].current_slot)
                tensor = torch.tensor([[1., 2.]])
                for turn in range(2):
                    for index, block in enumerate(blocks):
                        hook = by_block[index]
                        args, kwargs = hook.pre_forward(block, tensor)
                        self.assertTrue(torch.equal(block.weight, full[index].reshape(2, 2)), (count, turn, index))
                        output = block(*args, **kwargs)
                        self.assertTrue(torch.equal(output, tensor @ full[index].reshape(2, 2).t()))
                        self.assertIs(hook.post_forward(block, output), output)
                        self.assertEqual(block.weight.numel(), 0)
                self.assertEqual(len({x[torch.float32].data_ptr() for x in outputs}), 2)
                self.assertEqual(len({x[torch.float32].data_ptr() for x in inputs}), 2)

    def test_prepared_native_prefetch_repoint_and_reuse(self):
        with torch.device("meta"):
            block = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
            block.register_buffer("scale", torch.empty(2, dtype=torch.float32))
        old = block.weight
        old.custom_marker = "preserved"
        entries = [dict(name="weight", dtype=torch.bfloat16, shape=(2, 3)),
                   dict(name="scale", dtype=torch.float32, shape=(2,))]
        own = {torch.bfloat16: torch.tensor([1, 2, 3], dtype=torch.bfloat16),
               torch.float32: torch.tensor([.5], dtype=torch.float32)}
        peer = {torch.bfloat16: torch.tensor([4, 5, 6], dtype=torch.bfloat16),
                torch.float32: torch.tensor([.25], dtype=torch.float32)}
        bundle = PreparedBlock("blocks.0", 2, 0, own, canonical_metadata(entries))
        prepare_all_meta_blocks([block], [bundle], require_pinned=False)
        self.assertFalse(block.weight.is_meta)
        self.assertEqual(block.weight.numel(), 0)
        self.assertEqual(block.weight.custom_marker, "preserved")
        self.assertIsNot(block.weight, old)
        parameter_id = id(block.weight)

        class Stream:
            def wait_stream(self, other): pass
            def wait_event(self, event): pass
        class Event:
            def record(self, stream): pass
        stream = Stream()
        platform = SimpleNamespace(current_stream=lambda: stream, stream=lambda s: nullcontext(), Event=Event)
        outputs = [{torch.bfloat16: torch.zeros(6, dtype=torch.bfloat16), torch.float32: torch.zeros(2)} for _ in range(2)]
        inputs = [{torch.bfloat16: torch.zeros(3, dtype=torch.bfloat16), torch.float32: torch.zeros(1)} for _ in range(2)]
        calls = []
        def gather(output, local, group):
            calls.append((local.dtype, local.numel()))
            output.copy_(torch.cat((local, peer[local.dtype])))
        hook_cls = make_prepared_hook_class(native.DistributedLayerwiseOffloadHook)
        with patch.object(native, "current_omni_platform", platform), \
             patch.object(native.DistributedLayerwiseOffloadHook, "_shard_and_pin", side_effect=AssertionError("must bypass full sharding")), \
             patch.object(torch.distributed, "all_gather_into_tensor", side_effect=gather):
            hook = hook_cls(next_block=block, device=torch.device("cpu"), dp_group=object(), dp_size=2, rank=0,
                            copy_stream=stream, comm_stream=stream, pin_memory=False, shared_buffers=outputs, prepared=bundle)
            hook.initialize_hook(block)
            hook.gpu_shard_buffers = inputs
            self.assertIs(hook.cpu_shards, own)
            hook.prefetch_layer(0, non_blocking=False)
            self.assertTrue(torch.equal(block.weight, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(block.scale, torch.tensor([.5, .25])))
            self.assertEqual(block.weight.untyped_storage().data_ptr(), outputs[0][torch.bfloat16].untyped_storage().data_ptr())
            hook.offload_layer()
            self.assertEqual(block.weight.numel(), 0)
            hook.prefetch_layer(1, non_blocking=False)
            self.assertEqual(id(block.weight), parameter_id)
            self.assertEqual(block.weight.untyped_storage().data_ptr(), outputs[1][torch.bfloat16].untyped_storage().data_ptr())
        self.assertEqual(len(calls), 4)

    def test_all_bundle_validation_precedes_any_placeholder_mutation(self):
        with torch.device("meta"):
            blocks = [torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16) for _ in range(2)]
        entries = [dict(name="weight", dtype=torch.bfloat16, shape=(2, 3))]
        bundles = [PreparedBlock(f"blocks.{i}", 2, 0, {torch.bfloat16:torch.zeros(3, dtype=torch.bfloat16)},
                                 canonical_metadata(entries)) for i in range(2)]
        bundles[1].metadata[torch.bfloat16][0]["offset"] = 1
        with self.assertRaises(ValueError):
            prepare_all_meta_blocks(blocks, bundles, require_pinned=False)
        self.assertTrue(all(block.weight.is_meta for block in blocks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
