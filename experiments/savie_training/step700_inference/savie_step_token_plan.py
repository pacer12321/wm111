"""Candidate: share mask metadata across blocks of ONE denoising forward.

No activation/KV caching and no selector change. A new plan must be attached
after SP slicing on every model.forward. Mask values must remain immutable
inside that forward. The first consumer still does the original all-gather
and checks it against the pre-sharding global mask; later blocks reuse it.
"""
from dataclasses import dataclass
import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return None


@dataclass
class StepTokenPlan:
    local_mask: torch.Tensor
    expected_global: torch.Tensor
    active_indices: torch.Tensor
    stable_indices: torch.Tensor
    local_version: int | None
    global_version: int | None
    gathered_mask: torch.Tensor | None = None
    group: object = None
    world: int | None = None

    def validate(self, local):
        if local is not self.local_mask:
            raise ValueError('Token plan belongs to a different forward/mask')
        if (_version(local) != self.local_version or
                _version(self.expected_global) != self.global_version):
            raise ValueError('Active mask mutated after token-plan construction')


def attach_step_plan(local_mask, global_mask, shard_start):
    if local_mask.dtype != torch.bool or global_mask.dtype != torch.bool:
        raise ValueError('Boolean active masks required')
    if local_mask.ndim != 1 or global_mask.ndim != 1 or local_mask.device != global_mask.device:
        raise ValueError('Expected one-dimensional masks on one device')
    stop = shard_start + local_mask.numel()
    if shard_start < 0 or stop > global_mask.numel():
        raise ValueError('Invalid SP shard interval')
    if not torch.equal(local_mask, global_mask[shard_start:stop]):
        raise ValueError('Local active mask does not match this physical shard')
    plan = StepTokenPlan(local_mask, global_mask,
                         torch.nonzero(local_mask).flatten(),
                         torch.nonzero(~local_mask).flatten(),
                         _version(local_mask), _version(global_mask))
    local_mask._savie_step_token_plan = plan
    return plan


def token_indices(local_mask, stable=False):
    plan = getattr(local_mask, '_savie_step_token_plan', None)
    if plan is None:
        return torch.nonzero(~local_mask if stable else local_mask).flatten()
    plan.validate(local_mask)
    return plan.stable_indices if stable else plan.active_indices


def gathered_active_mask(local_mask, *, world, group, all_gather):
    plan = getattr(local_mask, '_savie_step_token_plan', None)
    if plan is not None:
        plan.validate(local_mask)
        if plan.gathered_mask is not None:
            if plan.group is not group or plan.world != world:
                raise ValueError('Cannot reuse token plan across communication groups')
            return plan.gathered_mask
    local_u8 = local_mask.to(dtype=torch.uint8).contiguous()
    parts = [torch.empty_like(local_u8) for _ in range(world)]
    all_gather(parts, local_u8, group=group)
    gathered = torch.cat(parts).bool()
    if plan is not None:
        if not torch.equal(gathered, plan.expected_global):
            raise ValueError('Ranks disagree on active mask; refusing cached metadata')
        plan.gathered_mask = gathered
        plan.group, plan.world = group, world
    return gathered
