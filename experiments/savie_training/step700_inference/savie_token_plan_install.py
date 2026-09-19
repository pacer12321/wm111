"""Per-forward mask metadata only: no historical K/V or activation reuse."""
import inspect
import linecache
import textwrap
from savie_step_token_plan import attach_step_plan,gathered_active_mask,token_indices


def _replace(module,cls,name,source):
    filename=f'<savie_token_plan_{cls.__name__}_{name}>'
    source='from __future__ import annotations\n'+source
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace={}
    exec(compile(source,filename,'exec'),module.__dict__,namespace)
    setattr(cls,name,namespace[name])


def install(module):
    if getattr(module.MiniMaxH3DiTModel,'_savie_token_plan_installed',False):
        raise RuntimeError('Step token plan installed twice')
    module.__dict__.update(_savie_attach_step_plan=attach_step_plan,
                           _savie_gathered_active_mask=gathered_active_mask,
                           _savie_token_indices=token_indices)
    source=textwrap.dedent(inspect.getsource(module.MiniMaxH3DiTModel.forward))
    needle='        spot_active_local = spot_active_global[start : start + local_rows]'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'\n        _savie_attach_step_plan(spot_active_local, spot_active_global, start)')
    _replace(module,module.MiniMaxH3DiTModel,'forward',source)
    source=textwrap.dedent(inspect.getsource(module.MiniMaxH3Attention._run_openvdn_ulysses))
    old='''            local_u8 = active_query_mask_local.to(device=x.device, dtype=torch.uint8).contiguous()
            gathered = [torch.empty_like(local_u8) for _ in range(world)]
            dist.all_gather(gathered, local_u8, group=ctx.ulysses_pg)
            active_query_mask_global = torch.cat(gathered).to(torch.bool)'''
    new='''            active_query_mask_global = _savie_gathered_active_mask(
                active_query_mask_local, world=world, group=ctx.ulysses_pg, all_gather=dist.all_gather)'''
    assert source.count(old)==1
    _replace(module,module.MiniMaxH3Attention,'_run_openvdn_ulysses',source.replace(old,new))
    source=textwrap.dedent(inspect.getsource(module.MiniMaxH3Attention.forward))
    old='torch.nonzero(spotedit_active_mask, as_tuple=False).view(-1)'
    assert source.count(old)==1
    _replace(module,module.MiniMaxH3Attention,'forward',source.replace(old,'_savie_token_indices(spotedit_active_mask)'))
    module.MiniMaxH3DiTModel._savie_token_plan_installed=True
