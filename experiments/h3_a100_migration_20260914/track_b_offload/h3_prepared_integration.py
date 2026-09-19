"""Opt-in H3 B/D storage integration for an ISOLATED candidate copy only.

Bootstrap at the bottom of that copy's pipeline_minimax_h3.py:
    from h3_prepared_integration import install_for_pipeline_module
    install_for_pipeline_module(sys.modules[__name__])

Activation requires ZHONGHAO_H3_PREPARED_OFFLOAD=1 and an absolute manifest
path in ZHONGHAO_H3_PREPARED_MANIFEST. Keep ordinary layerwise-offload enabled;
do NOT enable generic DLO. No attention methods, dtype, request or seed change.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import time

LOG = logging.getLogger(__name__)
_INSTALLED = False


def base_only_enabled():
    value = os.environ.get("ZHONGHAO_H3_PREPARED_BASE_ONLY", "0")
    if value not in ("0", "1"):
        raise ValueError("ZHONGHAO_H3_PREPARED_BASE_ONLY must be exactly 0 or 1")
    return value == "1"


def enabled():
    value = os.environ.get("ZHONGHAO_H3_PREPARED_OFFLOAD", "0")
    if value not in ("0", "1"):
        raise ValueError("ZHONGHAO_H3_PREPARED_OFFLOAD must be exactly 0 or 1")
    return value == "1"


def validate_options(config):
    parallel = config.parallel_config
    if not getattr(config, "enable_layerwise_offload", False):
        raise ValueError("Prepared H3 preserves ordinary layerwise lifecycle; flag must remain on")
    if getattr(config, "enable_distributed_layerwise_offload", False) or getattr(config, "enable_cpu_offload", False):
        raise ValueError("Do not mix generic distributed/model offload with prepared H3")
    if getattr(config, "quantization_config", None) is not None or getattr(parallel, "use_hsdp", False):
        raise ValueError("Prepared H3 supports unquantized DiT TP1, not HSDP/quantization")
    if int(parallel.tensor_parallel_size) != 1 or int(parallel.ulysses_degree) != 2 or int(parallel.ring_degree) != 1:
        raise ValueError("This verified storage recipe requires DiT TP1/Ulysses2/ring1")
    if int(getattr(parallel, "data_parallel_size", 1)) != 1 or int(getattr(parallel, "cfg_parallel_size", 1)) != 1:
        raise ValueError("Prepared H3 currently requires DP1/CFG1")
    if not getattr(config, "pin_cpu_memory", True):
        raise ValueError("Prepared H3 production shards require pinned CPU memory")
    if not getattr(config, "dlo_use_allgather", True):
        raise ValueError("Prepared SP2 storage requires AllGather; independent-rank mode is unsupported")


def ring_hook_plan(block_count):
    """Native order: last->first, then 0->1, 1->2, ..., N-2->N-1."""
    if block_count < 2:
        raise ValueError("At least two blocks required")
    return [(block_count - 1, 0)] + [(i, i + 1) for i in range(block_count - 1)]


def _read_manifest(pipeline):
    raw_path = os.environ.get("ZHONGHAO_H3_PREPARED_MANIFEST")
    if not raw_path or not Path(raw_path).is_absolute():
        raise ValueError("Set absolute ZHONGHAO_H3_PREPARED_MANIFEST")
    manifest = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if (manifest.get("base_count"), manifest.get("branch_count"), manifest.get("lora_tensor_count"), manifest.get("lora_pair_count")) != (535, 800, 416, 208):
        raise ValueError("Full Ref2VA/Stage-B manifest required")
    model_path = Path(pipeline.od_config.model).resolve()
    if base_only_enabled():
        base_root = model_path / "transformer"
        plans = []
        for original in manifest["plans"]:
            source_path = Path(original["source_file"]).resolve()
            if source_path.is_relative_to(base_root):
                item = dict(original)
                item["lora_pairs"] = []
                plans.append(item)
        if len(plans) != 535:
            raise ValueError(f"Expected 535 Ref2VA base plans, found {len(plans)}")
        manifest["plans"] = plans
        manifest["branch_count"] = 0
        manifest["lora_tensor_count"] = 0
        manifest["lora_pair_count"] = 0
        manifest["file_provenance"] = [
            item for item in manifest["file_provenance"] if item["group"] == "base"
        ]
        for item in manifest["file_provenance"]:
            if not Path(item["path"]).resolve().is_relative_to(base_root):
                raise ValueError("Base-only manifest points outside Ref2VA transformer")
        return manifest
    checkpoint = Path(pipeline._openvdn_checkpoint).resolve()
    for item in manifest["file_provenance"]:
        path = Path(item["path"]).resolve()
        allowed = model_path / "transformer" if item["group"] == "base" else checkpoint
        if not path.is_relative_to(allowed):
            raise ValueError("Manifest points outside selected model/Stage-B checkpoint")
    return manifest


def _reader_factory(manifest):
    from streaming_shards import RangeReader
    import struct
    provenance = {str(Path(x["path"]).resolve()): x for x in manifest["file_provenance"]}
    readers = {}

    def get(path):
        path = str(Path(path).resolve())
        if path not in readers:
            record = provenance[path]
            # Check exact collected header identity before payload reads.
            with Path(path).open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if length != record["header_size"]:
                    raise ValueError("Checkpoint header length changed")
                raw = stream.read(length)
            if hashlib.sha256(raw).hexdigest() != record["header_sha256"] or Path(path).stat().st_size != record["file_size"]:
                raise ValueError("Checkpoint header/file extent changed")
            readers[path] = RangeReader(path)
        return readers[path]
    return get


def _one_plan(item, reader):
    from streaming_shards import TensorPlan, TensorRef, LoRA
    def ref(source):
        result = TensorRef(reader(source["source_file"]), source["source_key"])
        if result.info["shape"] != source["shape"] or result.info["dtype"] != source["dtype"]:
            raise ValueError("Manifest/source shape or dtype changed")
        return result
    return TensorPlan(item["model_name"], ref(item), item.get("qkv_heads"), item.get("head_dim"),
                      tuple(LoRA(p["start_row"], p["end_row"], ref(p["a"]), ref(p["b"])) for p in item["lora_pairs"]))


def _assign_loaded_nonmain(model, name, value):
    """Replace a meta-only non-main tensor with its final CPU payload."""
    import torch
    prefix, _, local = name.rpartition(".")
    owner = model.get_submodule(prefix) if prefix else model
    if local in owner._parameters:
        old = owner._parameters[local]
        if old.device.type != "meta" or old.shape != value.shape or old.dtype != value.dtype:
            raise ValueError("Non-main target not untouched meta/matching shape-dtype")
        if torch.is_inference(value):
            raise ValueError("Persistent non-main payload must have a version counter")
        # Preserve the final payload without cloning and retain the original
        # flag. Explicitly override any outer loader inference context.
        with torch.inference_mode(False):
            new = torch.nn.Parameter(value, requires_grad=old.requires_grad)
        new.__dict__.update(old.__dict__)
        owner._parameters[local] = new
    elif local in owner._buffers:
        old = owner._buffers[local]
        if old.device.type != "meta" or old.shape != value.shape or old.dtype != value.dtype:
            raise ValueError("Non-main buffer shape/dtype mismatch")
        owner._buffers[local] = value
    else:
        raise ValueError("Non-main tensor missing in real model")


def load_prepared_pipeline_weights(pipeline, *, phase_timings=None):
    from manifest_builder import attach_model_metadata, tensor_plans_for_main_block
    from meta_model_metadata import extract_meta_model_metadata
    from prepared_shard_hook import bundle_from_streaming
    from torch_streaming_shards import build_torch_block_shard
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    if getattr(pipeline, "_h3_prepared_storage", None) is not None:
        raise RuntimeError("Refusing to load/merge prepared weights twice")
    validate_options(pipeline.od_config)
    base_only = base_only_enabled()
    if pipeline.partition.lower() != "ref2va":
        raise ValueError("Prepared storage requires the Ref2VA partition")
    if base_only:
        if pipeline._openvdn_checkpoint is not None:
            raise ValueError("Base-only prepared storage forbids a Stage-B checkpoint")
    elif pipeline._openvdn_checkpoint is None:
        raise ValueError("Stage-B prepared storage requires its selected checkpoint")
    group = get_sp_group()
    if group.world_size != 2 or group.rank_in_group not in (0, 1):
        raise ValueError("Live SP process group must match 2-rank storage recipe")
    started = time.monotonic()
    model = pipeline.transformer
    metadata = extract_meta_model_metadata(model, require_openvdn=not base_only)
    if metadata["nonpersistent_buffers_requiring_reconstruction"]:
        raise ValueError("Unexpected nonpersistent buffers; add explicit reconstruction before integration")
    manifest = attach_model_metadata(_read_manifest(pipeline), metadata)
    reader = _reader_factory(manifest)
    bundles = []
    for index in range(50):
        if base_only:
            selected = [
                item for item in manifest["plans"]
                if item["offload_unit"] == f"blocks.{index}"
            ]
            if len(selected) != 10:
                raise ValueError(
                    f"Expected 10 base tensors for block {index}, found {len(selected)}"
                )
            selected.sort(key=lambda item: item["model_iteration_index"])
            plans = [_one_plan(item, reader) for item in selected]
        else:
            plans = tensor_plans_for_main_block(manifest, index, reader)
        shards, layout = build_torch_block_shard(plans, 2, group.rank_in_group, pin_memory=True)
        bundles.append(bundle_from_streaming(f"blocks.{index}", 2, group.rank_in_group, shards, layout,
                                            provenance=dict(
                                                recipe=(
                                                    "H3_Ref2VA_base_layout"
                                                    if base_only
                                                    else "H3_StageB_final_model_layout"
                                                ),
                                                index=index,
                                            )))
        LOG.info("H3_PREPARED_MAIN rank=%s blocks=%d/50", group.rank_in_group, index + 1)
    main_done = time.monotonic()
    nonmain = [p for p in manifest["plans"] if p["offload_unit"] == "non_main_block_requires_explicit_policy"]
    if len(nonmain) != 35:
        raise ValueError("Expected 35 non-main DiT entries")
    for item in sorted(nonmain, key=lambda p: p["model_iteration_index"]):
        # Only this small set is replicated. Refiner QKV/LoRA still runs
        # through identical model-layout transform and production merge path.
        payload, _ = build_torch_block_shard([_one_plan(item, reader)], 1, 0, pin_memory=False)
        value = payload[item["dtype"]].reshape(item["shape"])
        _assign_loaded_nonmain(model, item["model_name"], value)
    model.post_load_weights()  # F32 persistent rope/embedder contract
    model._openvdn_checkpoint_loaded = not base_only
    pipeline._h3_prepared_storage = dict(bundles=bundles, group=group, metadata=metadata,
                                         manifest=manifest, offloader_enabled=False)
    if phase_timings is not None:
        phase_timings.update(prepared_main=main_done-started, prepared_nonmain=time.monotonic()-main_done)
    pipeline._openvdn_load_record = dict(base_tensor_count=535,
        branch_tensor_count=0 if base_only else 800,
        lora_pairs_merged=0 if base_only else 208,
        qkv_merge_layout="base grouped-QKV reorder only" if base_only else "post-base-loader contiguous Q/K/V thirds",
        merge_dtype="none" if base_only else "FP32 delta, cast then BF16 add",
        storage="rank-local prepared shards; nonmain replicated", rank=group.rank_in_group,
        base_only=base_only,
        load_and_merge_seconds=time.monotonic()-started)
    LOG.info("H3_PREPARED_LOAD_RECORD %s", json.dumps(pipeline._openvdn_load_record, sort_keys=True))
    loaded = {"transformer." + p["model_name"] for p in manifest["plans"]}
    for component_name in ("text_encoder", "video_vae", "audio_vae"):
        loaded.update(f"{component_name}.{name}" for name, _ in getattr(pipeline, component_name).named_parameters())
    return loaded


def enable_prepared_backend(backend, pipeline):
    """Main-block ring only; preserves original H3 non-main lifecycle."""
    import torch
    from prepared_shard_hook import prepare_all_meta_blocks, register_prepared_hook
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
    from vllm_omni.platforms import current_omni_platform
    validate_options(pipeline.od_config)
    state = pipeline._h3_prepared_storage
    if backend.enabled or state["offloader_enabled"]:
        raise RuntimeError("Prepared offloader already enabled")
    model, group = pipeline.transformer, state["group"]
    if group.world_size != 2 or backend.config.dp_size != 2:
        raise ValueError("Live/config SP size mismatch at offloader enable")
    if not backend.config.pin_cpu_memory:
        raise ValueError("Production H2D path requires pre-pinned shards")
    # Match ordinary layerwise initialization. H3 _encode_text_hidden still
    # loads encoder and explicitly returns it to CPU in its finally block.
    # Do not install DLO's generic permanently-resident encoder hooks.
    pipeline.text_encoder.to(backend.device)
    pipeline.video_vae.to(backend.device)
    pipeline.audio_vae.to(backend.device)
    for name, child in model.named_children():
        if name != "blocks":
            child.to(backend.device)
    for value in model._parameters.values():
        if value is not None:
            value.data = value.data.to(backend.device)
    for value in model._buffers.values():
        if value is not None:
            value.data = value.data.to(backend.device)
    blocks = list(model.blocks)
    prepare_all_meta_blocks(blocks, state["bundles"], require_pinned=True)
    backend.comm_stream = current_omni_platform.Stream()
    hooks = []
    # Keep a cleanup list immediately; a later registration/allocation
    # failure must not leave hooks silently active in this process.
    backend._h3_prepared_hooks = hooks
    backend._h3_prepared_blocks = blocks
    try:
        for current, following in ring_hook_plan(len(blocks)):
            hooks.append(register_prepared_hook(blocks[current], blocks[following], prepared=state["bundles"][following],
                device=backend.device, dp_group=group.device_group, dp_size=2, rank=group.rank_in_group,
                copy_stream=backend.copy_stream, comm_stream=backend.comm_stream, pin_memory=True,
                shared_buffers=[None, None]))
        # These buffers become live Parameter storage during forward and need
        # ordinary TensorImpl version counters, even if enable() is called
        # from an outer inference context.
        with torch.inference_mode(False):
            shared = DistributedLayerwiseOffloadBackend._allocate_shared_buffers(hooks)
            shard_buffers = DistributedLayerwiseOffloadBackend._allocate_shared_shard_buffers(hooks)
        slot_groups = [-1, -1]
        for index, hook in enumerate(hooks):
            hook._prev_hook = hooks[index - 1]
            hook.current_slot = index % 2
            hook.gpu_buffers, hook.gpu_shard_buffers = shared, shard_buffers
            hook._owns_buffers = False
            hook._group_id = 0
            hook._shared_slot_group = slot_groups
        hooks[1]._is_group_first = True
        # Same initialization sequence as audited native DLO backend. It
        # primes the last block; group-first then refetches the first block.
        hooks[-1].prefetch_layer(slot=hooks[0].current_slot, non_blocking=False)
        hooks[-1].get_weights(hooks[0].current_slot)
        backend._blocks = [blocks]
        backend.enabled = True
        state["offloader_enabled"] = True
        LOG.info("H3_PREPARED_OFFLOAD_ENABLED rank=%s blocks=50 buffers=2_full+2_half", group.rank_in_group)
    except BaseException:
        disable_prepared_backend(backend)
        raise


def disable_prepared_backend(backend):
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import remove_distributed_block_hook
    from vllm_omni.platforms import current_omni_platform
    if hasattr(backend, "_h3_prepared_blocks"):
        current_omni_platform.synchronize()
        for block in backend._h3_prepared_blocks:
            remove_distributed_block_hook(block)
        backend._h3_prepared_hooks.clear()
        backend._h3_prepared_blocks.clear()
        backend._blocks = []
        backend.enabled = False


def install_for_pipeline_module(pipeline_module):
    """Install process-local methods, only for explicitly enabled H3 instances."""
    global _INSTALLED, LOG
    if not enabled():
        return
    if _INSTALLED:
        return
    import torch
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
    LOG = init_logger(__name__)
    original_constructor = pipeline_module.MiniMaxH3DiTModel
    pipeline_cls = pipeline_module.MiniMaxH3Pipeline
    original_load = pipeline_cls._load_weights_impl
    original_process = DiffusersPipelineLoader._process_weights_after_loading
    original_enable, original_disable = LayerWiseOffloadBackend.enable, LayerWiseOffloadBackend.disable

    def meta_constructor(od_config, *args, **kwargs):
        mapping = od_config.tf_model_config
        mapping = mapping.to_dict() if hasattr(mapping, "to_dict") else dict(mapping)
        if not mapping.get("openvdn_enabled") and not base_only_enabled():
            return original_constructor(od_config, *args, **kwargs)
        validate_options(od_config)
        with torch.device("meta"):
            model = original_constructor(od_config, *args, **kwargs)
        if any(t.device.type != "meta" for t in list(model.parameters()) + list(model.buffers())):
            raise RuntimeError("Meta-first constructor allocated real DiT weights")
        return model

    def load(self, weights, *, phase_timings=None):
        if base_only_enabled():
            return load_prepared_pipeline_weights(self, phase_timings=phase_timings)
        if self._openvdn_checkpoint is None:
            return original_load(self, weights, phase_timings=phase_timings)
        # Deliberately DO NOT iterate weights: generic iterator would load
        # full checkpoint tensors before the rank-local reader gets control.
        return load_prepared_pipeline_weights(self, phase_timings=phase_timings)

    def process(self, model, target_device):
        if getattr(model, "_h3_prepared_storage", None) is None:
            return original_process(self, model, target_device)
        validate_options(model.od_config)
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None and not isinstance(method, UnquantizedLinearMethod):
                raise ValueError("Unexpected post-load transform; prepared path cannot bypass it")
        # Audited UnquantizedLinearMethod post-load is a CUDA no-op. Generic
        # method would call module.to(cpu) on meta parameters first, causing
        # either a failure or premature full materialization. No quantization
        # or shape-changing postprocess is permitted in this integration.
        from vllm.platforms import current_platform
        if not current_platform.is_cuda_alike():
            raise ValueError("Post-load bypass validated only for CUDA unquantized path")
        LOG.info("H3_PREPARED_POSTLOAD_NOOP: unquantized CUDA, no full meta materialization")

    def enable_backend(self, pipeline):
        if getattr(pipeline, "_h3_prepared_storage", None) is None:
            return original_enable(self, pipeline)
        return enable_prepared_backend(self, pipeline)

    def disable_backend(self):
        if hasattr(self, "_h3_prepared_blocks"):
            return disable_prepared_backend(self)
        return original_disable(self)

    pipeline_module.MiniMaxH3DiTModel = meta_constructor
    pipeline_cls._load_weights_impl = load
    DiffusersPipelineLoader._process_weights_after_loading = process
    LayerWiseOffloadBackend.enable = enable_backend
    LayerWiseOffloadBackend.disable = disable_backend
    _INSTALLED = True
    code_files = ("h3_prepared_integration.py", "prepared_shard_hook.py", "torch_streaming_shards.py",
                  "streaming_shards.py", "manifest_builder.py", "meta_model_metadata.py")
    code_hashes = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in code_files}
    LOG.info("H3_PREPARED_STORAGE_CODE_HASHES %s", json.dumps(code_hashes, sort_keys=True))
    LOG.info("H3_PREPARED_INTEGRATION_INSTALLED: meta-first direct-shard loading; ordinary non-DiT lifecycle")
