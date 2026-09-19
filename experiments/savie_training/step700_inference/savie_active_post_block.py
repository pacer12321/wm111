"""Inference-only candidate: omit discarded stable-row post-attention math.

Norm1 and attention stay full-input so current stable K/V remain available.
The existing refresh creates per-block stable-output caches; no new cache
approximation is introduced. Only row-independent operations are narrowed.
"""
import torch


def run_active_post_block(self, x, *, reference_forward, module, **kwargs):
    active_mask=kwargs.get('spotedit_active_mask')
    if active_mask is None or kwargs.get('spotedit_refresh',False):
        return reference_forward(self,x,**kwargs)
    if torch.is_grad_enabled():
        raise RuntimeError('Active post-block candidate is inference-only')
    active_mask=active_mask.to(device=x.device,dtype=torch.bool)
    if active_mask.shape!=(x.shape[0],):
        raise ValueError('Mask must cover every local sequence row')
    from savie_step_token_plan import token_indices
    active=token_indices(active_mask)
    stable=token_indices(active_mask,True)
    cached=getattr(self,'_spotedit_stable_output',None)
    if cached is None or cached.shape!=(stable.numel(),x.shape[-1]):
        raise RuntimeError('Missing or mismatched stable block-output cache')
    profile=module._profile_scope
    dtype=module._BF16_DTYPE
    combined=kwargs['combined_indices']
    with profile('adaln_projection'):
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaln_proj(kwargs['t_emb'])
    with profile('norm1_and_modulation'):
        h=module._modulate_scale_shift(self.norm1(x),shift_msa,scale_msa,combined,dtype=dtype)
    attn_kwargs={key:value for key,value in kwargs.items() if key not in ('t_emb','combined_indices')}
    attn_kwargs['spotedit_active_mask']=active_mask
    attention=self.attn(h,**attn_kwargs)
    active_combined=combined.index_select(0,active)
    with profile('attention_residual_gate'):
        residual=module._modulate_gate(x.index_select(0,active),gate_msa,
            attention.index_select(0,active),active_combined,dtype=dtype)
    with profile('norm2_and_modulation'):
        h=module._modulate_scale_shift(self.norm2(residual),shift_mlp,scale_mlp,active_combined,dtype=dtype)
    with profile('mlp'):
        h=self.mlp(h)
    with profile('mlp_residual_gate'):
        output_active=module._modulate_gate(residual,gate_mlp,h,active_combined,dtype=dtype)
    with profile('stable_cache_update'):
        output=torch.empty_like(x)
        output.index_copy_(0,active,output_active)
        output.index_copy_(0,stable,cached.to(device=x.device,dtype=x.dtype))
    return output


def install(module):
    if getattr(module.MiniMaxH3DiTBlock,'_savie_active_post_installed',False):
        raise RuntimeError('Active post-block installed twice')
    reference=module.MiniMaxH3DiTBlock.forward
    def forward(self,x,**kwargs):
        return run_active_post_block(self,x,reference_forward=reference,module=module,**kwargs)
    module.MiniMaxH3DiTBlock.forward=forward
    module.MiniMaxH3DiTBlock._savie_active_post_installed=True
