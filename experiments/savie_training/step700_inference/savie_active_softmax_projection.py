"""Candidate: move softmax output gating into already-active projection rows.

Only used on partial-query forwards. Stable rows are discarded by the
existing projection path, not used as K/V here. All QKV and key access stay
unchanged. Keep original full-query path. Not installed by default.
"""


def active_softmax_projection(attn, x, out_flat, rows):
    selected = out_flat.index_select(0, rows).reshape(-1, attn.num_heads, attn.head_dim)
    gate = attn.softmax_gate(x.index_select(0, rows)).to(selected.dtype)
    projected, _ = attn.out_proj((selected * gate).flatten(1))
    return projected


def install(module):
    import inspect
    import linecache
    import textwrap
    cls = module.MiniMaxH3Attention
    if getattr(cls, '_savie_active_softmax_projection', False):
        raise RuntimeError('Already installed')
    source = textwrap.dedent(inspect.getsource(cls.forward))
    old = '                out = out * self.softmax_gate(x).to(out.dtype)'
    assert source.count(old) == 1
    source = source.replace(old,
        '                if active_query_mask is None:\n'
        '                    out = out * self.softmax_gate(x).to(out.dtype)')
    old = '                projected, _ = self.out_proj(out.index_select(0, active_indices))'
    assert source.count(old) == 1
    source = source.replace(old,
        '                if self.openvdn_enabled:\n'
        '                    projected = _savie_active_softmax_projection(self, x, out, active_indices)\n'
        '                else:\n'
        '                    projected, _ = self.out_proj(out.index_select(0, active_indices))')
    source = 'from __future__ import annotations\n' + source
    filename = '<savie_active_softmax_projection>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module.__dict__['_savie_active_softmax_projection'] = active_softmax_projection
    namespace = {}
    exec(compile(source, filename, 'exec'), module.__dict__, namespace)
    cls.forward = namespace['forward']
    cls._savie_active_softmax_projection = True
