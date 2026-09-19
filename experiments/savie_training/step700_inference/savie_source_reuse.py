"""Isolated approximate S-query reuse; T selector and connectivity unchanged.

Current K/V projections and communication are retained. This is NOT full KV
reuse. One cache backend supports whole-S and selective-S policies.
"""
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F


def drift_score(previous, current):
    """Per-row relative L2; magnitude changes are not normalized away."""
    a, b = previous.float(), current.float()
    return (b-a).square().sum(-1).sqrt() / a.square().sum(-1).sqrt().clamp_min(1e-6)


def choose_active(scores, frames, height, width):
    if scores.numel() != frames*height*width or not bool(torch.isfinite(scores).all()):
        raise ValueError('Invalid S drift scores')
    ordered = scores.detach().float().flatten().cpu().sort().values.double()
    # No evidence of separable populations: conservative, do not skip S.
    if ordered.numel() < 2 or float(ordered[-1]-ordered[0]) < 1e-8:
        return torch.ones_like(scores, dtype=torch.bool), None
    prefix, square = ordered.cumsum(0), ordered.square().cumsum(0)
    left = torch.arange(1, ordered.numel(), dtype=torch.float64)
    right = ordered.numel()-left
    sse = square[:-1]-prefix[:-1].square()/left + square[-1]-square[:-1]-(prefix[-1]-prefix[:-1]).square()/right
    split = int(sse.argmin())+1
    threshold = float((ordered[split-1]+ordered[split])/2)
    active = (scores >= threshold).float().reshape(1,1,frames,height,width)
    active = F.max_pool3d(active,(1,3,3),1,(0,1,1)).flatten().bool()
    return active, threshold


def partial_source_batches(t, s, n, mapping, active_mask=None, group_size=4):
    from savie_source_query_flash import _source_reuse_original_prepare
    batches = _source_reuse_original_prepare(t,s,n,mapping,group_size)
    if active_mask is None:
        return batches
    result = []
    for qidx, kidx, qc, kc in batches:
        qparts, kparts, newq, newk = [], [], [], []
        q0 = k0 = 0
        for q1, k1 in zip(qc,kc):
            selected = qidx[q0:q1]
            selected = selected[active_mask[selected]]
            if selected.numel():
                qparts.append(selected); kparts.append(kidx[k0:k1])
                newq.append((newq[-1] if newq else 0)+selected.numel())
                newk.append((newk[-1] if newk else 0)+k1-k0)
            q0, k0 = q1, k1
        if qparts:
            result.append((torch.cat(qparts),torch.cat(kparts),newq,newk))
    return result


def reused_block(block, x, *, module, kwargs, plan, source_cache):
    """Compute active queries/MLP only; fill S and T from separate caches."""
    if torch.is_grad_enabled():
        raise RuntimeError('Source reuse is inference-only')
    from savie_step_token_plan import token_indices
    live = plan['live']; active = token_indices(live)
    stable_t = plan['stable_t']; stable_s = plan['stable_s']
    cached_t = getattr(block,'_spotedit_stable_output',None)
    if cached_t is None or cached_t.shape != (stable_t.numel(),x.shape[-1]):
        raise RuntimeError('T cache contract violated by S reuse')
    if source_cache.shape != (plan['source_rows'].numel(),x.shape[-1]):
        raise RuntimeError('Missing/mismatched S output cache')
    profile, dtype = module._profile_scope, module._BF16_DTYPE
    with profile('adaln_projection'):
        a,b,c,d,e,f = block.adaln_proj(kwargs['t_emb'])
    with profile('norm1_and_modulation'):
        h = module._modulate_scale_shift(block.norm1(x),a,b,kwargs['combined_indices'],dtype=dtype)
    attention_kwargs = {key:value for key,value in kwargs.items() if key not in ('t_emb','combined_indices')}
    attention_kwargs.update(spotedit_active_mask=live,spotedit_refresh=False)
    attn = block.attn(h,**attention_kwargs)
    indices = kwargs['combined_indices'].index_select(0,active)
    with profile('attention_residual_gate'):
        residual = module._modulate_gate(x.index_select(0,active),c,attn.index_select(0,active),indices,dtype=dtype)
    with profile('norm2_and_modulation'):
        h = module._modulate_scale_shift(block.norm2(residual),d,e,indices,dtype=dtype)
    with profile('mlp'):
        h = block.mlp(h)
    with profile('mlp_residual_gate'):
        fresh = module._modulate_gate(residual,f,h,indices,dtype=dtype)
    with profile('stable_cache_update'):
        output = torch.empty_like(x)
        output.index_copy_(0,active,fresh)
        output.index_copy_(0,stable_t,cached_t)
        output.index_copy_(0,stable_s,source_cache.index_select(0,plan['stable_s_in_source']))
    return output


class Controller:
    def __init__(self, mode):
        if mode not in ('whole','selective'):
            raise ValueError(mode)
        self.mode = mode
        self.reset()

    def reset(self):
        self.cache = {}
        self.mask = None
        self.step = 0
        self.plan = None
        self.scores = None
        self.observed = set()
        self.last_t = None

    def begin(self,t):
        if t == 0 or self.last_t is None or t < self.last_t:
            self.reset()
        elif t == self.last_t:
            raise RuntimeError('Repeated diffusion timestep; ambiguous S cache request')
        self.last_t = t
        self.step += 1
        if self.step > 8:
            raise RuntimeError('Expected at most 8 DMD forwards')
        self.refresh = self.step in (1,2,5)
        self.plan = None
        self.scores = None
        self.observed = set()

    def make_plan(self,x,kw):
        from savie_step_token_plan import attach_step_plan,token_indices
        local = kw['spotedit_active_mask']
        original = getattr(local,'_savie_step_token_plan',None)
        if original is None:
            raise RuntimeError('S reuse needs verified per-forward token plan')
        m,s = kw['interleaved_map'],kw['openvdn_source_layout']
        logical = kw['local_logical_indices']
        rows = torch.nonzero((logical>=s.video_start)&(logical<s.video_end)).flatten()
        source_indices = logical[rows]-s.video_start
        if self.refresh:
            return dict(source_rows=rows,source_indices=source_indices,layout=s)
        if kw['spotedit_refresh']:
            raise RuntimeError('This candidate requires T refresh to be a subset of S refresh')
        if self.mode=='selective' and self.mask is None:
            raise RuntimeError('Selective S-skip has no own-request drift calibration')
        source_active = torch.zeros(s.num_frames*s.tokens_per_frame,device=x.device,dtype=torch.bool) if self.mode=='whole' else self.mask
        global_live = original.expected_global.clone()
        source_global = torch.arange(s.video_start,s.video_end,device=x.device)
        global_live[m.logical_to_physical[source_global]] = source_active
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
        start = int(get_sp_group().ulysses_rank)*x.shape[0]
        local_live = global_live[start:start+x.shape[0]]
        attach_step_plan(local_live,global_live,start)
        stable_source_positions = torch.nonzero(~source_active[source_indices]).flatten()
        return dict(source_rows=rows,source_indices=source_indices,layout=s,live=local_live,
                    stable_t=token_indices(local,True),
                    stable_s=rows[stable_source_positions],stable_s_in_source=stable_source_positions)

    def block(self,block,x,module,reference,kw):
        if self.plan is None:
            self.plan = self.make_plan(x,kw)
        p = self.plan; index = block.attn._block_index
        if self.refresh:
            out = reference(block,x,**kw)
            current = out.index_select(0,p['source_rows']).detach()
            if self.step==2 and index in (8,24,41):
                previous = self.cache.get(index)
                if previous is None:
                    raise RuntimeError('S calibration missing first-forward states')
                score = drift_score(previous,current)
                self.scores = score if self.scores is None else torch.maximum(self.scores,score)
                self.observed.add(index)
            # source rows are copied, never aliases of transient block output.
            self.cache[index] = current
            return out
        return reused_block(block,x,module=module,kwargs=kw,plan=p,source_cache=self.cache[index])

    def finish(self):
        import torch.distributed as dist
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
        group = get_sp_group(); p = self.plan; s = p['layout']
        threshold = None
        if self.step==2:
            if self.observed != {8,24,41}:
                raise RuntimeError('S drift probes not all evaluated')
            scores = torch.zeros(s.num_frames*s.tokens_per_frame,device=self.scores.device)
            scores[p['source_indices']] = self.scores
            dist.all_reduce(scores,op=dist.ReduceOp.MAX,group=group.device_group)
            mask,threshold = choose_active(scores,s.num_frames,s.frame_height,s.frame_width)
            dist.broadcast(mask,src=0,group=group.device_group)
            self.mask = mask
            self.calibration = dict(threshold=threshold,active_ratio=float(mask.float().mean()),
                                    stable_ratio=float((~mask).float().mean()),
                                    drift_mean=float(scores.mean()),drift_max=float(scores.max()),
                                    rule='max relative-L2 at layers8/24/41; exact2means+3x3 active dilation',
                                    quality_calibrated=False)
        cache_bytes = sum(t.numel()*t.element_size() for t in self.cache.values())
        record = dict(event='S_REUSE',mode=self.mode,forward=self.step,refresh=self.refresh,
                      cache_bytes_per_rank=cache_bytes,rank=int(group.ulysses_rank),
                      skipped_source_rows=0 if self.refresh else int(p['stable_s'].numel()),
                      kv_recomputed=True,source_refresh_forwards=[1,2,5])
        if self.step>=2: record['calibration'] = self.calibration
        print(json.dumps(record),flush=True)
        if int(group.ulysses_rank)==0:
            target = Path(os.environ['SAVIE_S_REUSE_LOG'])
            with target.open('a') as out: out.write(json.dumps(record)+'\n')


def install(module):
    import savie_source_query_flash as source
    if hasattr(source,'_source_reuse_original_prepare'):
        raise RuntimeError('Source reuse installed twice')
    source._source_reuse_original_prepare = source.prepare_source_batches
    source.prepare_source_batches = partial_source_batches
    mode = os.environ['SAVIE_S_REUSE_MODE']
    old_model = module.MiniMaxH3DiTModel.forward
    old_block = module.MiniMaxH3DiTBlock.forward
    def model_forward(self,**kwargs):
        if torch.is_grad_enabled(): raise RuntimeError('S reuse cannot run during training')
        update = kwargs['update_mask'].flatten().bool()
        positions = self._pos_ids(kwargs['img_pos_info'],'img_pos_info')
        targets = positions[update.to(positions.device)]
        inverse = kwargs['inverse_indices'].flatten().long()
        ts = kwargs['unique_timesteps'].flatten()
        t = float(ts[inverse[targets[0]]].item())
        controller = getattr(self,'_savie_source_reuse',None)
        if controller is None:
            controller = Controller(mode)
            self._savie_source_reuse = controller
            for block in self.blocks: block._savie_source_reuse = controller
        controller.begin(t)
        result = old_model(self,**kwargs)
        controller.finish()
        return result
    def block_forward(self,x,**kw):
        controller = getattr(self,'_savie_source_reuse',None)
        if controller is None: raise RuntimeError('Missing source request controller')
        return controller.block(self,x,module,old_block,kw)
    module.MiniMaxH3DiTModel.forward = model_forward
    module.MiniMaxH3DiTBlock.forward = block_forward
    print('SAVIE_SOURCE_REUSE_INSTALLED mode='+mode+' refresh=1,2,5 current_KV=true',flush=True)
