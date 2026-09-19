"""Finalize only the private CUDA environment; never touch system drivers."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/cache/zhonghao/h3')
WORK = ROOT / 'a100_v1'
ENV = ROOT / 'env_cuda_v1'


def main():
    assert Path(sys.executable).parent == ENV / 'bin'
    assert importlib.metadata.version('vllm') == '0.26.0'
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
               UV_CACHE_DIR=str(ROOT / 'uv_cache'), PIP_CACHE_DIR=str(ROOT / 'pip_cache'),
               SETUPTOOLS_SCM_PRETEND_VERSION='0.26.0', VLLM_TARGET_DEVICE='cuda',
               TMPDIR=str(ROOT / 'build_tmp'))
    uv = str(ENV / 'bin/uv')
    install_state = json.loads((WORK / 'environment_status.json').read_text())
    index = install_state.get('index', 'https://pypi.org/simple')
    if 'mirrors.huaweicloud.com' in index:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name, None)
    fallback = install_state.get('fallback_index')
    index_args = (['--index', index, '--default-index', fallback,
                   '--index-strategy', 'unsafe-first-match'] if fallback else ['--index-url', index])
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, *index_args,
                    'wheel', 'setuptools-scm>=8.0', 'setuptools>=77.0.3,<81.0.0'], env=env, check=True)
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--no-deps', '--no-build-isolation',
                    '-e', str(WORK / 'candidates/A')], env=env, check=True)
    # Ensure the package operation didn't mutate frozen computation modules.
    manifest = json.loads((WORK / 'cuda_manifest.json').read_text())
    import hashlib
    for case in 'ABCD':
        for name, expected in manifest['cases'][case]['model_files'].items():
            path = WORK / f'candidates/{case}/vllm_omni/diffusion/models/minimax_h3' / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    (WORK / 'finalize_status.json').write_text(json.dumps({'installed_omni': True, 'model_math_unchanged': True}))


if __name__ == '__main__':
    main()
