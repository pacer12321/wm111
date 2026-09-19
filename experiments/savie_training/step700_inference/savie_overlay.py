"""Isolated step700 overlay for the EXISTING DMD8 vLLM/odd-even/skip pipeline.

No changes to training, scheduler, base loading, VAE, text encoder or collectives.
SAViE weights are overlaid on the CPU shard-at-read plans after DMD adapters.
"""
import dataclasses
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, "/cache/zhonghao/h3/savie_step700_eval/repo")
sys.path.insert(0, "/cache/zhonghao/h3/savie_step700_eval/deps")

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

LOG = logging.getLogger(__name__)
_WEIGHTS = None
_MASKS = OrderedDict()
_MAX_MASKS = 8
_FLEX = None
_CHECKPOINT_STEP = int(os.environ.get("SAVIE_CHECKPOINT_STEP", "700"))


def training_mask_predicate():
    # Load the exact training predicate without importing its package __init__,
    # which requires the training-only newer Diffusers distribution.
    import importlib.util
    name = '_savie_training_dual_mask'
    if name not in sys.modules:
        path = Path('/cache/zhonghao/h3/savie_step700_eval/repo/src/models/softmax_attention/dual_stream_flex.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].make_dual_stream_mask_mod


def unpatchify_video_rows(rows, shape):
    b,c,t,h,w = shape
    assert rows.shape == (t*(h//2)*(w//2),c*4)
    grid=rows.reshape(b,t,h//2,w//2,c,1,2,2)
    return torch.einsum('nthwcrpq->nctrhpwq',grid).reshape(b,c,t,h,w)


def latent_selector_mask(source,first_x0):
    import torch.nn.functional as F
    score=(F.normalize(source.float(),dim=1,eps=1e-6)-F.normalize(first_x0.float(),dim=1,eps=1e-6)).square().sum(1,keepdim=True)
    score=F.avg_pool3d(score,(1,2,2),(1,2,2))
    ordered=score.detach().float().flatten().cpu().sort().values
    n=ordered.numel()
    prefix=ordered.cumsum(0).double();sq=ordered.square().cumsum(0).double()
    nl=torch.arange(1,n,dtype=torch.float64);nr=n-nl
    sse=sq[:-1]-prefix[:-1].square()/nl+sq[-1]-sq[:-1]-(prefix[-1]-prefix[:-1]).square()/nr
    split=int(sse.argmin())+1
    threshold=(float(ordered[split-1])+float(ordered[split]))/2
    active=F.max_pool3d((score>=threshold).float(),(1,3,3),1,(0,1,1)).bool()
    return active.flatten(),score.flatten(),threshold


class TensorReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.header = {
            k: {"shape": list(t.shape), "dtype": "BF16" if t.dtype == torch.bfloat16 else "F32"}
            for k, t in tensors.items()
        }

    def read_flat(self, key, start, stop):
        t = self.tensors[key].reshape(-1)[start:stop].contiguous()
        return (t.view(torch.uint16) if t.dtype == torch.bfloat16 else t).numpy().copy()


def weights():
    global _WEIGHTS
    if _WEIGHTS is None:
        path = os.environ["SAVIE_STEP700_CHECKPOINT"]
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        meta = obj["metadata"]
        if int(meta.get("step", -1)) != _CHECKPOINT_STEP:
            raise ValueError(f"Expected exactly step{_CHECKPOINT_STEP}, got metadata {meta}")
        state = obj["weights"]
        if len(state) != 1150:
            raise ValueError(f"Unexpected SAViE parameter count: {len(state)}")
        for key, value in state.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite step700 weight: {key}")
        _WEIGHTS = state
        LOG.warning("SAVIE_CHECKPOINT_LOADED step=%d tensors=%d path=%s", _CHECKPOINT_STEP, len(state), path)
    return _WEIGHTS


def overlay_plans(plans, index):
    from streaming_shards import TensorRef, LoRA
    state = weights()
    prefix = f"transformer_blocks.{index}.attn."
    used = set()
    result = []
    for plan in plans:
        suffix = plan.name.split(f"blocks.{index}.attn.", 1)[-1]
        native = prefix + suffix
        if native in state:
            dtype = torch.bfloat16 if plan.source.info["dtype"] == "BF16" else torch.float32
            value = state[native].to(dtype)
            if list(value.shape) != plan.source.info["shape"]:
                raise ValueError(f"Shape mismatch for {native}")
            reader = TensorReader({native: value})
            plan = dataclasses.replace(plan, source=TensorRef(reader, native))
            used.add(native)
        parts = []
        if suffix == "qkv_proj.weight":
            d = plan.source.info["shape"][0] // 3
            parts = [(f"orig.to_{part}", j*d, (j+1)*d) for j, part in enumerate("qkv")]
        elif suffix == "out_proj.weight":
            parts = [("orig.to_out.0", 0, plan.source.info["shape"][0])]
        elif suffix == "to_out_linear.weight":
            parts = [("to_out_linear", 0, plan.source.info["shape"][0])]
        loras = list(plan.loras)
        for name, lo, hi in parts:
            a, b = prefix + name + ".lora_A.savie.weight", prefix + name + ".lora_B.savie.weight"
            if a not in state or b not in state or state[a].shape[0] != 64:
                raise ValueError(f"Missing/rank-mismatched SAViE adapter: {a}")
            reader = TensorReader({a: state[a].float(), b: state[b].float()})
            loras.append(LoRA(lo, hi, TensorRef(reader, a), TensorRef(reader, b)))
            used.update((a, b))
        result.append(dataclasses.replace(plan, loras=tuple(loras)))
    expected = {k for k in state if k.startswith(prefix)}
    if used != expected:
        raise ValueError(f"Unmapped step700 keys: {sorted(expected-used)}; unexpected={sorted(used-expected)}")
    LOG.warning("SAVIE_OVERLAY block=%d tensors=%d exact_key_coverage=true", index, len(used))
    return result


def make_mask(layout, source, packed_length, device, permutation,
              active_indices=None, world_size=2):
    from savie_mask_arithmetic import make_arithmetic_mask
    return make_arithmetic_mask(layout, source, packed_length, permutation,
                                active_indices, world_size)


def savie_softmax(self, q, k, v, layout, source_layout=None,
                  active_query_mask=None, interleaved_map=None):
    global _FLEX
    if source_layout is None or interleaved_map is None:
        raise ValueError("SAViE evaluation requires source layout and odd/even Ulysses")
    if active_query_mask is not None:
        # Pack only after all-to-all; preserve ownership, RoPE and connectivity.
        from savie_partial_layout import partial_softmax
        if _FLEX is None:
            _FLEX = torch.compile(flex_attention, dynamic=True)
        return partial_softmax(q, k, v, layout, source_layout, active_query_mask,
                               interleaved_map, _FLEX, self.softmax_scale)
    length = q.shape[0]
    world = getattr(interleaved_map, 'world_size', 2)
    permutation = interleaved_map.physical_to_logical
    active = (None if active_query_mask is None
              else torch.nonzero(active_query_mask, as_tuple=False).flatten())
    if active is not None and active.numel() == 0:
        return torch.zeros_like(q)
    qlen = length if active is None else active.numel()
    key = (layout, source_layout, length, str(q.device), active is None, world)
    cached = _MASKS.get(key)
    if (cached is None or cached[2] is not permutation
            or (active is not None and not torch.equal(cached[0], active))):
        predicate = make_mask(layout, source_layout, length, q.device,
                              permutation, active, world_size=world)
        block_mask = create_block_mask(predicate, B=None, H=None,
                                       Q_LEN=qlen, KV_LEN=length,
                                       device=q.device, BLOCK_SIZE=128, _compile=True)
        cached = (None if active is None else active.clone(), block_mask, permutation)
        _MASKS[key] = cached
        if len(_MASKS) > _MAX_MASKS:
            _MASKS.popitem(last=False)
        LOG.warning("SAVIE_FLEX_MASK_READY q=%d kv=%d ss=vdn_local ts=same_frame st=absent mapping=arithmetic_c5r1", qlen, length)
    _MASKS.move_to_end(key)
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=True)
    query = q.index_select(0, active) if active_query_mask is not None else q
    out = _FLEX(query.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                block_mask=cached[1], scale=self.softmax_scale)[0].transpose(0, 1)
    if active_query_mask is None:
        return out
    result = torch.zeros_like(q)
    return result.index_copy_(0, active, out)


def install(module):
    if os.environ.get("SAVIE_STEP700_CHECKPOINT") is None:
        return
    import manifest_builder
    original_plans = manifest_builder.tensor_plans_for_main_block
    def plans(manifest, index, reader):
        return overlay_plans(original_plans(manifest, index, reader), index)
    manifest_builder.tensor_plans_for_main_block = plans
    module.MiniMaxH3Attention._openvdn_softmax = savie_softmax
    original_forward = module.MiniMaxH3DiTModel.forward
    def forward(self, **kwargs):
        update = kwargs["update_mask"].flatten().bool()
        positions = self._pos_ids(kwargs["img_pos_info"], "img_pos_info")
        targets = positions[update.to(positions.device)]
        inverse = kwargs["inverse_indices"].flatten().long()
        ts = kwargs["unique_timesteps"].flatten()
        t = float(ts[inverse[targets[0]]].item())
        if t == 0.0:
            self._spotedit_step = 0
            self._spotedit_last_target_t = None
            self._spotedit_input_cache = None
            self._spotedit_payload = None  # Reload the newly calibrated step700 mask.
        video, audio = original_forward(self, **kwargs)
        marker = Path(os.environ["SAVIE_SELECTOR_READY"])
        if t == 0.0 and self._spotedit_payload.get('selector_checkpoint_step') != _CHECKPOINT_STEP:
            # First all-active forward from THIS checkpoint, not the old B mask.
            shape = tuple(self._spotedit_payload["source_shape"])
            device = video.device
            output_positions = self._pos_ids(kwargs['img_pos_for_infer_output_info'], 'img_pos_for_infer_output_info')
            if not torch.equal(positions.to(device),output_positions.to(device)):
                raise ValueError('Selector requires identical input/output visual row order')
            visual = kwargs["x"][0].index_select(0, positions.to(device))
            target_rows = visual[update.to(device)]
            predicted = target_rows.float() + video[update.to(device)].float()
            source_rows = self._spotedit_payload["source_clean_normalized_rows"].to(device)
            source = unpatchify_video_rows(source_rows, shape)
            first_x0 = unpatchify_video_rows(predicted, shape)
            active, score, threshold = latent_selector_mask(source, first_x0)
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
            import torch.distributed as dist
            group = get_sp_group()
            # Identical global gathered output, but use authoritative rank0 mask.
            dist.broadcast(active, src=0, group=group.device_group)
            if group.rank_in_group == 0:
                payload = dict(self._spotedit_payload)
                payload.update(active_target_mask=active.cpu(), selector_threshold=threshold,
                               selector_checkpoint_step=_CHECKPOINT_STEP, selector_score=score.cpu())
                path = Path(self._spotedit_payload_path)
                temporary = path.with_suffix(".new.pt")
                torch.save(payload, temporary)
                temporary.replace(path)
                marker.write_text(json.dumps({"step":_CHECKPOINT_STEP,"active_ratio":float(active.float().mean()),
                                              "stable_ratio":float((~active).float().mean()),
                                              "threshold":threshold}))
            dist.barrier(group=group.device_group)
            LOG.warning("SAVIE_SELECTOR_CALIBRATED checkpoint=%d active=%.6f", _CHECKPOINT_STEP, active.float().mean())
        return video, audio
    module.MiniMaxH3DiTModel.forward = forward
    LOG.warning("SAVIE_STEP700_OVERLAY_INSTALLED existing_dmd8_sp2_skip_pipeline=true partial_layout=post_ulysses_logical_pack odd_even_ownership_unchanged=true")
