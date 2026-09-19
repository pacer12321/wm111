"""Create isolated CUDA copies from frozen ABCD; preserve all attention routing."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/cache/zhonghao/h3')
WORK = ROOT / 'a100_v1'
REL = Path('vllm_omni/diffusion/models/minimax_h3')
SOURCES = {'A': 'src/vllm-omni', 'B': 'candidates/b_v1/vllm-omni',
           'C': 'candidates/c_v1/vllm-omni', 'D': 'dualstream_v1/candidate/vllm-omni'}

CUDA_BRANCH = '''    if query.device.type == "cuda":
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
        if flash_attn_varlen_func is None:
            raise RuntimeError("CUDA varlen FlashAttention unavailable; no slow fallback allowed")
        cache_key = (tuple(q_cu), tuple(kv_cu), str(query.device))
        if cache_key not in _CUDA_CU_CACHE:
            if len(_CUDA_CU_CACHE) >= 128:
                _CUDA_CU_CACHE.clear()
            q_bounds, k_bounds = (0, *q_cu), (0, *kv_cu)
            _CUDA_CU_CACHE[cache_key] = (
                torch.tensor(q_bounds, dtype=torch.int32, device=query.device),
                torch.tensor(k_bounds, dtype=torch.int32, device=query.device),
                max(b - a for a, b in zip(q_bounds, q_bounds[1:])),
                max(b - a for a, b in zip(k_bounds, k_bounds[1:])),
            )
        cu_q, cu_k, max_q, max_k = _CUDA_CU_CACHE[cache_key]
        output = flash_attn_varlen_func(
            q=query.contiguous(), k=key.contiguous(), v=value.contiguous(),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=scale, causal=False,
        )
        return output[0] if isinstance(output, tuple) else output
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_kernel(text):
    needle = '    if query.device.type != "npu" or torch_npu is None:\n'
    if text.count(needle) != 1:
        raise RuntimeError('Frozen fusion wrapper signature changed')
    changed = text.replace(needle, CUDA_BRANCH + needle)
    changed += '\n# CUDA cumulative-length cache only; no activation or KV caching.\n_CUDA_CU_CACHE = {}\n'
    # AST proof: removing CUDA branch and metadata cache restores original file.
    before, after = ast.parse(text), ast.parse(changed)
    for node in after.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_fusion_attention_tnd':
            node.body.pop(0)
    after.body.pop()
    if ast.dump(before) != ast.dump(after):
        raise RuntimeError('CUDA adaptation changed non-backend statements')
    return changed


def main():
    manifest = {'purpose': 'CUDA backend port; original model policies unchanged', 'cases': {}}
    for case, relative in SOURCES.items():
        src = ROOT / 'frozen' / relative
        dst = WORK / 'candidates' / case
        if dst.exists():
            raise RuntimeError(f'Refusing overwrite: {dst}')
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '.git', '*.egg-info'))
        changes = {}
        if case != 'A':
            kernel = dst / REL / 'openvdn_npu.py'
            before = digest(kernel)
            kernel.write_text(patch_kernel(kernel.read_text()))
            changes[str(REL / kernel.name)] = {'before': before, 'after': digest(kernel),
                                              'change': 'CUDA varlen backend only; original routes and linear math unchanged'}
        if case == 'D':
            adapter = dst / REL / 'dual_stream_adapter.py'
            original = adapter.read_text()
            assert original.count('world not in (1, 8)') == 1
            adapter.write_text(original.replace('world not in (1, 8)', 'world not in (1, 2, 4, 8)'))
            changes[str(REL / adapter.name)] = {'before': hashlib.sha256(original.encode()).hexdigest(),
                'after': digest(adapter), 'change': 'Allow world2 execution logging; no math changes'}
        for path in (dst / REL).glob('*.py'):
            compile(path.read_text(), str(path), 'exec')
        manifest['cases'][case] = {'frozen': str(src), 'candidate': str(dst), 'changes': changes,
            'model_files': {p.name: digest(p) for p in (dst / REL).glob('*.py')}}
    # Same input; preserve the original actual VLM decision and its hashes.
    src = ROOT / 'frozen/data/shirt_red_couple_124'
    sample = json.loads((src / 'manifest.json').read_text())
    assert digest(src / 'source.mp4') == sample['source_sha256']
    manifest['sample'] = sample
    manifest['source_path'] = str(src / 'source.mp4')
    policy = json.loads((ROOT / 'frozen/dualstream_v1/deploy_d/reviewed_policy.json').read_text())
    policy['candidate_vendor'] = str(WORK / 'candidates/D')
    policy['candidate_files'] = {str(REL / name): value for name, value in manifest['cases']['D']['model_files'].items()}
    policy['execution_contract'].update(device_type='cuda', heads_per_rank=28, world=2)
    policy['migration_note'] = 'Only CUDA backend and world2 logging adapted; attention rules unchanged.'
    (WORK / 'cuda_d_policy.json').write_text(json.dumps(policy, indent=2))
    manifest['d_policy_sha256'] = digest(WORK / 'cuda_d_policy.json')
    (WORK / 'cuda_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'prepared': list(SOURCES), 'gpu_validation': 'pending'}))


if __name__ == '__main__':
    main()
