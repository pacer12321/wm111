"""Instantiate ONLY the real H3 DiT on meta and emit validated order.

No model payload reads, pipeline, CUDA initialization, forward pass or
distributed process group. TP query functions are locally fixed to TP=1,
rank=0 for metadata; the real linear/branch constructors are retained.
Attention backend selection is SDPA solely to avoid GPU probing; Attention
contains no learned parameters. These metadata-only patches are recorded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validated-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.validated_manifest.exists():
        raise FileExistsError("Output must be new; do not overwrite prior audit")
    source = args.candidate_root / "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"
    expected = "a3dc6a0189fdd8f3a9e2aaeddf9544df1503a2f74af287b74b7e975dd7ce717c"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Candidate transformer changed; re-audit before metadata extraction")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(args.candidate_root.resolve()))
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from meta_model_metadata import extract_meta_model_metadata
    from manifest_builder import attach_model_metadata

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Refuse: CUDA already initialized")

    class MetaOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            device = kwargs.get("device")
            if device is not None and torch.device(device).type != "meta":
                raise RuntimeError(f"Non-meta tensor factory forbidden: {func}/{device}")
            result = func(*args, **kwargs)
            for value in tree_leaves(result):
                if isinstance(value, torch.Tensor) and value.device.type != "meta":
                    raise RuntimeError(f"Real tensor forbidden during metadata construction: {func}")
            return result

    def forbid_cuda(*args, **kwargs):
        raise RuntimeError("CUDA initialization forbidden in metadata harness")

    with patch("torch.cuda._lazy_init", forbid_cuda):
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
        import vllm.model_executor.layers.linear as linear
        import vllm.model_executor.parameter as parameter
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
        import vllm_omni.diffusion.attention.layer as attention_layer
        import vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer as h3

        if Path(h3.__file__).resolve() != source.resolve():
            raise RuntimeError("Imported the wrong H3 candidate")
        config = dict(manifest["config"], openvdn_enabled=True)
        od_config = SimpleNamespace(
            tf_model_config=config,
            parallel_config=SimpleNamespace(tensor_parallel_size=1, ulysses_degree=2, ring_degree=1),
            diffusion_attention_config=None,
            model_class_name="MiniMaxH3Pipeline",
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(linear, "get_tensor_model_parallel_world_size", return_value=1))
            stack.enter_context(patch.object(linear, "get_tensor_model_parallel_rank", return_value=0))
            stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0))
            stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1))
            stack.enter_context(patch.object(h3, "get_tensor_model_parallel_world_size", return_value=1))
            stack.enter_context(patch.object(attention_layer, "get_attn_backend_for_role", return_value=(SDPABackend, None)))
            stack.enter_context(set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))))
            stack.enter_context(set_current_diffusion_config(od_config))
            stack.enter_context(torch.device("meta"))
            stack.enter_context(MetaOnly())
            model = h3.MiniMaxH3DiTModel(od_config, quant_config=None)
            metadata = extract_meta_model_metadata(model)
        metadata["harness_patches"] = ["TP metadata queries=1/rank0", "parameter-free Attention backend selector=SDPA"]
        metadata["source_sha256"] = expected
        metadata["cuda_initialized"] = torch.cuda.is_initialized()
        if metadata["cuda_initialized"]:
            raise RuntimeError("CUDA unexpectedly initialized")
        validated = attach_model_metadata(manifest, metadata)
        # Record real per-block parameter+buffer order separately, including
        # nonpersistent buffers; it is the exact DLO flattening contract.
        metadata["block_orders"] = {}
        for index, block in enumerate(model.blocks):
            parameters = [(name, t) for name, t in block.named_parameters()]
            buffers = [(name, t) for name, t in block.named_buffers()]
            metadata["block_orders"][f"blocks.{index}"] = [name for name, _ in parameters + buffers]
        args.output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        args.validated_manifest.write_text(json.dumps(validated, indent=2), encoding="utf-8")
        print(json.dumps(dict(status="meta_shape_dtype_order_validated", entries=len(metadata["parameters_and_persistent_buffers"]),
                              nonpersistent=len(metadata["nonpersistent_buffers_requiring_reconstruction"]),
                              cuda_initialized=False, runtime_usable=False)))


if __name__ == "__main__":
    main()
