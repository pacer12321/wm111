"""Exact-support pruning of TT linear output gate/projection only.

The linear scan and every K/V remain unchanged. The deployed scan writes
only interior TARGET rows into an otherwise zero tensor. Its no-bias output
projection therefore need not run on source/text/audio/anchor/padding rows.
Never use this helper with source-hybrid enabled.
"""
import torch


def live_rows(logical,layout,active=None):
    keep=(logical>=layout.video_start+layout.tokens_per_frame)&(logical<layout.video_end-layout.tokens_per_frame)
    if active is not None:keep=keep&active
    return torch.nonzero(keep).flatten()


def gate_live_rows(attn,x,readout,layout,logical,active=None):
    if attn.openvdn_source_hybrid_enabled:
        raise ValueError('TT-only support pruning cannot be used with source linear')
    if logical is None or logical.shape!=(x.shape[0],):
        raise ValueError('Explicit local logical coordinates required')
    rows=live_rows(logical,layout,active)
    output=torch.zeros_like(readout)
    if rows.numel():
        gate=attn.linear_attention.output_gate(x.index_select(0,rows)).to(readout.dtype)
        output.index_copy_(0,rows,readout.index_select(0,rows)*gate)
    return output,rows


def add_live_projection(attn,readout,out,rows):
    if getattr(attn.to_out_linear,'bias',None) is not None:
        raise ValueError('Support pruning requires zero-bias linear projection')
    if rows.numel():
        projection=attn.to_out_linear(readout.index_select(0,rows).to(out.dtype))
        out.index_copy_(0,rows,out.index_select(0,rows)+projection)
    return out


def transformed_sources(module):
    """Fail closed if deployment code no longer has these exact branches."""
    import inspect
    import textwrap
    ulysses=textwrap.dedent(inspect.getsource(module.MiniMaxH3Attention._run_openvdn_ulysses))
    needle='        readout_local = readout_local * self.linear_attention.output_gate(x).to(readout_local.dtype)'
    assert ulysses.count(needle)==1
    ulysses=ulysses.replace(needle,
        '        readout_local, self._savie_linear_live_rows = _savie_gate_live_rows(\n'
        '            self, x, readout_local, layout, local_logical_indices, active_query_mask_local)')
    forward=textwrap.dedent(inspect.getsource(module.MiniMaxH3Attention.forward))
    begin=forward.index('            if active_indices is not None:',forward.index('if linear_readout_local is not None:'))
    end=forward.index('        else:\n            linear_readout = self.linear_attention',begin)
    forward=forward[:begin]+'''            with _profile_scope("linear_out_projection"):
                out = _savie_add_live_projection(self, linear_readout_local, out,
                                                 self._savie_linear_live_rows)
'''+forward[end:]
    return ulysses,forward


def install(module):
    import linecache
    if getattr(module.MiniMaxH3Attention,'_savie_live_rows_installed',False):
        raise RuntimeError('Linear live-row pruning installed twice')
    ulysses,forward=transformed_sources(module)
    module.__dict__.update(_savie_gate_live_rows=gate_live_rows,
                           _savie_add_live_projection=add_live_projection)
    namespace={}
    for name,source in (('ulysses',ulysses),('forward',forward)):
        source='from __future__ import annotations\n'+source
        filename=f'<savie_linear_live_rows_{name}>'
        linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
        exec(compile(source,filename,'exec'),module.__dict__,namespace)
    module.MiniMaxH3Attention._run_openvdn_ulysses=namespace['_run_openvdn_ulysses']
    module.MiniMaxH3Attention.forward=namespace['forward']
    module.MiniMaxH3Attention._savie_live_rows_installed=True
