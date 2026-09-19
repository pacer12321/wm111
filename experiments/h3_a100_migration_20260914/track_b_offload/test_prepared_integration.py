"""CPU-only control-flow tests. No torch/model payload/GPU imports required."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h3_prepared_integration as integration


def config():
    return SimpleNamespace(enable_layerwise_offload=True, enable_distributed_layerwise_offload=False,
        enable_cpu_offload=False, quantization_config=None, pin_cpu_memory=True,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, ulysses_degree=2, ring_degree=1,
                                        data_parallel_size=1, cfg_parallel_size=1, use_hsdp=False))


class IntegrationTests(unittest.TestCase):
    def test_recipe_and_rejected_modes(self):
        integration.validate_options(config())
        for field, value in (("enable_distributed_layerwise_offload", True), ("enable_cpu_offload", True),
                             ("pin_cpu_memory", False), ("quantization_config", object()), ("enable_layerwise_offload", False),
                             ("dlo_use_allgather", False)):
            c = config()
            setattr(c, field, value)
            with self.assertRaises(ValueError): integration.validate_options(c)
        for field, value in (("tensor_parallel_size", 2), ("ulysses_degree", 1), ("ring_degree", 2),
                             ("data_parallel_size", 2), ("use_hsdp", True)):
            c = config()
            setattr(c.parallel_config, field, value)
            with self.assertRaises(ValueError): integration.validate_options(c)

    def test_ring_matches_native_order_and_unique_ownership(self):
        for count in (2, 3, 50):
            plan = integration.ring_hook_plan(count)
            self.assertEqual(plan[0], (count - 1, 0))
            self.assertEqual(sorted(x for x, _ in plan), list(range(count)))
            self.assertEqual(sorted(y for _, y in plan), list(range(count)))
            for index, (current, _) in enumerate(plan):
                self.assertEqual(plan[index - 1][1], current)

    def test_flags_fail_closed(self):
        with patch.dict("os.environ", {"ZHONGHAO_H3_PREPARED_OFFLOAD":"wrong"}):
            with self.assertRaises(ValueError): integration.enabled()
        with patch.dict("os.environ", {"ZHONGHAO_H3_PREPARED_OFFLOAD":"0"}):
            self.assertFalse(integration.enabled())

    def test_install_keeps_encoder_method_and_never_consumes_generic_weights(self):
        class Pipeline:
            def _load_weights_impl(self, weights, *, phase_timings=None): return {"old"}
            def _encode_text_hidden(self, *args): return "original lifecycle"
        class Loader:
            def _process_weights_after_loading(self, model, device): return "old_process"
        class Backend:
            def enable(self, model): return "old_enable"
            def disable(self): return "old_disable"
        class Unquantized: pass
        original_encode = Pipeline._encode_text_hidden
        constructor = Mock()
        module = SimpleNamespace(MiniMaxH3DiTModel=constructor, MiniMaxH3Pipeline=Pipeline)
        mocks = {
            "torch": SimpleNamespace(device=lambda device:nullcontext()),
            "vllm.logger": SimpleNamespace(init_logger=Mock(return_value=integration.LOG)),
            "vllm.model_executor.layers.linear": SimpleNamespace(UnquantizedLinearMethod=Unquantized),
            "vllm_omni.diffusion.model_loader.diffusers_loader": SimpleNamespace(DiffusersPipelineLoader=Loader),
            "vllm_omni.diffusion.offloader.layerwise_backend": SimpleNamespace(LayerWiseOffloadBackend=Backend),
            "vllm.platforms": SimpleNamespace(current_platform=SimpleNamespace(is_cuda_alike=lambda:True)),
        }
        class Poison:
            def __iter__(self): raise AssertionError("Generic full weights iterator was consumed")
        with patch.dict("sys.modules", mocks), patch.dict("os.environ", {"ZHONGHAO_H3_PREPARED_OFFLOAD":"1"}), \
             patch.object(integration, "_INSTALLED", False), \
             patch.object(integration, "load_prepared_pipeline_weights", return_value={"prepared"}) as load:
            integration.install_for_pipeline_module(module)
            instance = Pipeline()
            instance._openvdn_checkpoint = "/stageb"
            self.assertEqual(instance._load_weights_impl(Poison()), {"prepared"})
            self.assertIs(Pipeline._encode_text_hidden, original_encode)
            load.assert_called_once()
            plain = SimpleNamespace()
            self.assertEqual(Loader()._process_weights_after_loading(plain, "cpu"), "old_process")
            self.assertEqual(Backend().enable(plain), "old_enable")
            marked = SimpleNamespace(_h3_prepared_storage={}, od_config=config(),
                                      modules=lambda:[SimpleNamespace(quant_method=Unquantized())])
            self.assertIsNone(Loader()._process_weights_after_loading(marked, "cpu"))
            marked.modules = lambda:[SimpleNamespace(quant_method=object())]
            with self.assertRaises(ValueError): Loader()._process_weights_after_loading(marked, "cpu")

    def test_main_backend_uses_prepared_ring_only_and_preserves_component_moves(self):
        moves = []
        def component(name): return SimpleNamespace(to=lambda device:moves.append((name, device)))
        blocks = [object() for _ in range(50)]
        children = [("blocks", object()), ("refiner", component("refiner")), ("rope", component("rope"))]
        model = SimpleNamespace(blocks=blocks, named_children=lambda:iter(children), _parameters={}, _buffers={})
        group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=object())
        state = dict(group=group, bundles=[object() for _ in range(50)], offloader_enabled=False)
        pipeline = SimpleNamespace(od_config=config(), transformer=model, _h3_prepared_storage=state,
                                   text_encoder=component("TE"), video_vae=component("video"), audio_vae=component("audio"))
        backend = SimpleNamespace(enabled=False, config=SimpleNamespace(dp_size=2, pin_cpu_memory=True),
                                  device="cuda:logical", copy_stream=object())
        hooks = []
        def register(module, next_block, **kwargs):
            hook = SimpleNamespace(prefetch_layer=Mock(), get_weights=Mock())
            hooks.append(hook)
            self.assertIs(kwargs["prepared"], state["bundles"][blocks.index(next_block)])
            return hook
        native = SimpleNamespace(_allocate_shared_buffers=Mock(return_value=["full0", "full1"]),
                                 _allocate_shared_shard_buffers=Mock(return_value=["half0", "half1"]))
        with patch("prepared_shard_hook.prepare_all_meta_blocks") as prepare, \
             patch("prepared_shard_hook.register_prepared_hook", side_effect=register), \
             patch.dict("sys.modules", {
                 "vllm_omni.diffusion.offloader.distributed_layerwise_backend":SimpleNamespace(DistributedLayerwiseOffloadBackend=native),
                 "vllm_omni.platforms":SimpleNamespace(current_omni_platform=SimpleNamespace(Stream=lambda:object())),
             }):
            integration.enable_prepared_backend(backend, pipeline)
        prepare.assert_called_once_with(blocks, state["bundles"], require_pinned=True)
        self.assertEqual(len(hooks), 50)
        self.assertTrue(backend.enabled)
        self.assertTrue(state["offloader_enabled"])
        self.assertEqual([name for name,_ in moves], ["TE", "video", "audio", "refiner", "rope"])
        self.assertIs(hooks[1]._prev_hook, hooks[0])
        self.assertTrue(hooks[1]._is_group_first)
        hooks[-1].prefetch_layer.assert_called_once_with(slot=0, non_blocking=False)
        native._allocate_shared_buffers.assert_called_once_with(hooks)
        native._allocate_shared_shard_buffers.assert_called_once_with(hooks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
