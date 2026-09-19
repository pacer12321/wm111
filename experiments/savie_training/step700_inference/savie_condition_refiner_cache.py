"""Exact, request-local condition projection/refinement memoization candidate.

The cached subgraph has NO timestep or noisy-video input. This does NOT
cache source hidden states inside DiT. Value-compare cloned inputs so even
inference tensors without version counters cannot silently go stale.
"""
import torch
import logging

LOG = logging.getLogger(__name__)


def clear_request(model):
    model._savie_condition_cache = None
    model._savie_condition_cache_hits = 0
    model._savie_condition_cache_misses = 0


def refined_condition(model, text, cu, maximum):
    if torch.is_grad_enabled():
        raise RuntimeError('Frozen inference only; never cache a training graph')
    cached = getattr(model, '_savie_condition_cache', None)
    if cached is not None:
        old_text, old_cu, old_maximum, result = cached
        if (old_maximum == maximum and old_text.shape == text.shape and
                old_text.device == text.device and old_text.dtype == text.dtype and
                old_cu.device == cu.device and old_cu.dtype == cu.dtype and
                torch.equal(old_text, text) and torch.equal(old_cu, cu)):
            model._savie_condition_cache_hits = getattr(model, '_savie_condition_cache_hits', 0) + 1
            return result
    embedded, _ = model.condition_proj(text)
    result = model.token_refiner(embedded, cu_seqlens=cu, max_seqlen=maximum)
    model._savie_condition_cache = (text.detach().clone(), cu.detach().clone(), maximum, result)
    model._savie_condition_cache_misses = getattr(model, '_savie_condition_cache_misses', 0) + 1
    return result


def install(module):
    import inspect,linecache,textwrap
    cls=module.MiniMaxH3DiTModel
    if getattr(cls,'_savie_condition_cache_installed',False):
        raise RuntimeError('Condition cache already installed')
    source=textwrap.dedent(inspect.getsource(cls._embed))
    old='''    text_embed, _ = self.condition_proj(text_rows)
    text_embed = self.token_refiner(
        text_embed,
        cu_seqlens=refiner_cu_seqlens,
        max_seqlen=refiner_max_seqlen,
    )'''
    assert source.count(old)==1
    source=source.replace(old,'''    text_embed = _savie_refined_condition(
        self, text_rows, refiner_cu_seqlens, refiner_max_seqlen)''')
    source='from __future__ import annotations\n'+source
    filename='<savie_condition_refiner_cache>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    module.__dict__['_savie_refined_condition']=refined_condition
    namespace={};exec(compile(source,filename,'exec'),module.__dict__,namespace)
    cls._embed=namespace['_embed']
    original=cls.forward
    def forward(self,**kwargs):
        positions=self._pos_ids(kwargs['img_pos_info'],'img_pos_info')
        update=kwargs['update_mask'].flatten().bool().to(positions.device)
        first=positions[update][0]
        inverse=kwargs['inverse_indices'].flatten().long()
        timestep=float(kwargs['unique_timesteps'].flatten()[inverse[first]].item())
        if timestep==0.0:clear_request(self)
        output = original(self,**kwargs)
        LOG.warning('SAVIE_CONDITION_CACHE t=%.9f hits=%d misses=%d', timestep,
                    self._savie_condition_cache_hits, self._savie_condition_cache_misses)
        return output
    cls.forward=forward
    cls._savie_condition_cache_installed=True
