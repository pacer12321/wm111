"""Make fresh private diagnostic copies; never edit frozen B/D sources."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/cache/zhonghao/h3/profiling_d4_v1')
MODEL_REL = Path('vllm_omni/diffusion/models/minimax_h3')
ORIGINALS = {
    'D': Path('/cache/zhonghao/h3/dualstream_v1/candidate/vllm-omni'),
}


def main():
    assert Path(__file__).resolve().parent == ROOT
    sys.path.insert(0, '/cache/zhonghao/h3/color_trial_v1/b')
    import b_trial_gates as b
    sys.path.insert(0, '/cache/zhonghao/h3/dualstream_v1/deploy_d')
    import d_trial_gates as d
    host = b.profiles.require_host()
    policy = d.policy_gate('a412150f34a5f3c564e908e5ba85dcebe32ca641b2d139ce6963cbfc5d3949cd')
    proofs = {'B': b.source_code_manifest(), 'D': d.source_code_manifest(policy['value'])}
    sample = b.sample_profile('shirt_red_couple_124')
    manifest = {'host': host, 'original_proofs': proofs, 'copies': {},
                'source_video': str(sample.source), 'prompt': sample.prompt,
                'source_sha256': b.profiles.digest(sample.source),
                'generation': b.GENERATION, 'physical_cards': [0, 1, 2, 3], 'world_size': 4, 'policy_sha256': policy['sha256'] if 'sha256' in policy else 'a412150f34a5f3c564e908e5ba85dcebe32ca641b2d139ce6963cbfc5d3949cd'}
    for case, source in ORIGINALS.items():
        destination = ROOT / 'candidates' / case / 'vllm-omni'
        if destination.exists():
            raise RuntimeError('Refusing to overwrite a diagnostic candidate')
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '.git'))
        target = destination / MODEL_REL / 'minimax_h3_transformer.py'
        original = target.read_text()
        suffix = '\n# Diagnostic observers only: original statements above are unchanged.\nfrom profile_hooks import install as _install_profile_observers\n_install_profile_observers(globals())\n'
        transformed = original + suffix
        before, after = ast.parse(original), ast.parse(transformed)
        assert ast.dump(ast.Module(body=after.body[:-2], type_ignores=[])) == ast.dump(before)
        target.write_text(transformed)
        compile(transformed, str(target), 'exec')
        # Diagnostic logging previously admitted only world 1 or 8. Admit 4
        # in the private copy only; all attention and layout math is unchanged.
        adapter = destination / MODEL_REL / 'dual_stream_adapter.py'
        adapter_before = adapter.read_text()
        needle = 'world not in (1, 8)'
        assert adapter_before.count(needle) == 1
        adapter_after = adapter_before.replace(needle, 'world not in (1, 4, 8)')
        adapter.write_text(adapter_after)
        compile(adapter_after, str(adapter), 'exec')
        manifest['copies'][case] = {'vendor': str(destination), 'original': str(source),
                                    'files': {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in destination.rglob('*.py')}}
    manifest['diagnostic_code'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh'))}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'host': host, 'cases': list(ORIGINALS)}))


if __name__ == '__main__':
    main()
