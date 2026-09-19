"""Private A100 preparation only. Never launches a model or alters system drivers."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/cache/zhonghao/h3')
WORK = ROOT / 'a100_v1'
ENV = ROOT / 'env_cuda_v1'

# These two local tiny-test slices were copied into the old asset inventory.
# Formal B/C/D loads adapter_model.safetensors and model.safetensors instead.
EXCLUDED_TEST_ARTIFACTS = {
    'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000/adapters/default/openvdn_lora_5block.safetensors',
    'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000/linear_branch/openvdn_5block.safetensors',
}


def model_rows(manifest):
    return [item for name, item in manifest.items()
            if name.startswith('models/') and item['kind'] == 'file' and name not in EXCLUDED_TEST_ARTIFACTS]


def save(name, value):
    tmp = WORK / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(WORK / name)


def get_json(url):
    return json.loads(subprocess.check_output(
        ['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '60', url]))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def weights():
    save('weights_status.json', {'status': 'resolving_public_revisions'})
    manifest = json.loads((ROOT / 'frozen/transfer_verified.json').read_text())
    repositories = {'MiniMax-H3': 'MiniMaxAI/MiniMax-H3',
                    'OpenVDN-vdn-minimax-h3': 'OpenVDN/vdn-minimax-h3'}
    revision_file = WORK / 'model_revisions.json'
    revisions = json.loads(revision_file.read_text()) if revision_file.exists() else {
        folder: get_json('https://hf-mirror.com/api/models/' + repo)['sha'] for folder, repo in repositories.items()}
    save('model_revisions.json', revisions)
    rows = model_rows(manifest)
    save('weights_scope.json', {'excluded_local_test_artifacts': sorted(EXCLUDED_TEST_ARTIFACTS),
         'reason': 'Absent from official repository; not loaded by formal BCD checkpoint loader',
         'formal_file_count': len(rows), 'original_manifest_unchanged': True})
    save('weights_status.json', {'status': 'downloading', 'files': len(rows),
                                'bytes': sum(item['size'] for item in rows)})

    def download(item):
        relative = Path(item['path'])
        assert relative.parts[0] == 'models' and '..' not in relative.parts
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size == item['size'] and sha(destination) == item['sha256']:
                return relative.as_posix()
            raise RuntimeError(f'Refusing different existing asset: {relative}')
        repo_folder = relative.parts[1]
        remote_path = '/'.join(relative.parts[2:])
        url = f'https://hf-mirror.com/{repositories[repo_folder]}/resolve/{revisions[repo_folder]}/{remote_path}'
        partial = destination.with_name(destination.name + '.partial')
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item['size']:
                    raise RuntimeError('Oversize partial file')
                if offset < item['size']:
                    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                                    '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                                    '--continue-at', '-', '--output', str(partial), url], check=True)
                if partial.stat().st_size != item['size'] or sha(partial) != item['sha256']:
                    raise RuntimeError(f'Original-experiment SHA mismatch: {relative}')
                partial.rename(destination)
                return relative.as_posix()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5)

    completed = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for future in concurrent.futures.as_completed([executor.submit(download, row) for row in rows]):
                result = future.result()
                completed.append(result)
                print('Verified', result, flush=True)
                save('weights_status.json', {'status': 'downloading', 'completed': len(completed), 'total': len(rows)})
        save('weights_status.json', {'status': 'verified', 'files': completed, 'original_sha256_match': True})
    except Exception as exc:
        save('weights_status.json', {'status': 'failed', 'error': repr(exc), 'completed': completed})
        raise


def environment(index='https://pypi.org/simple', locked=False, proxy=None, fallback_index=None):
    lockfile = WORK / 'cuda_environment.lock.txt'
    if locked and not lockfile.is_file():
        raise RuntimeError('Missing frozen dependency lock')
    save('environment_status.json', {'status': 'installing', 'index': index,
         'fallback_index': fallback_index, 'lock_sha256': sha(lockfile) if locked else None})
    env = dict(os.environ)
    env.update(PIP_CACHE_DIR=str(ROOT / 'pip_cache'), UV_CACHE_DIR=str(ROOT / 'uv_cache'),
               TMPDIR=str(ROOT / 'build_tmp'), CUDA_VISIBLE_DEVICES='', MAX_JOBS='4',
               PYTHONNOUSERSITE='1', UV_HTTP_TIMEOUT='120')
    for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(name, None)
    if proxy:
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, http_proxy=proxy, https_proxy=proxy,
                   NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    else:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name, None)
    Path(env['TMPDIR']).mkdir(exist_ok=True)

    def run(*args):
        subprocess.run(args, check=True, env=env)

    try:
        if not ENV.exists():
            run('/cache/envs/swiftcam/bin/python', '-m', 'venv', str(ENV))
        python = str(ENV / 'bin/python')
        if not (ENV / 'bin/uv').is_file():
            run(python, '-m', 'pip', 'install', '--index-url', index, 'uv')
        if locked:
            index_args = (['--index', index, '--default-index', fallback_index,
                           '--index-strategy', 'unsafe-first-match'] if fallback_index
                          else ['--index-url', index])
            run(str(ENV / 'bin/uv'), 'pip', 'install', '--python', python,
                *index_args, '--require-hashes', '-r', str(lockfile))
        else:
            run(str(ENV / 'bin/uv'), 'pip', 'install', '--python', python,
                '--index-url', index, 'vllm==0.26.0',
                '-r', str(ROOT / 'frozen/src/vllm-omni/requirements/common.txt'), 'pytest', 'requests')
        # NVIDIA-supported application-local compatibility libraries; no apt,
        # kernel-driver changes, system linker changes, or shared environment edits.
        repo = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {'http': proxy, 'https': proxy} if proxy else {}))
        listing = opener.open(repo, timeout=60).read().decode()
        candidates = sorted(set(re.findall(r'cuda-compat-13-0_[0-9.]+-[0-9a-z.]+_amd64\.deb', listing)),
                            key=lambda name: tuple(int(x) for x in name.split('_')[1].split('-')[0].split('.')))
        if not candidates:
            raise RuntimeError('Official CUDA compatibility package not found')
        package = WORK / candidates[-1]
        if not package.exists():
            partial = package.with_suffix('.deb.partial')
            run('curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '600',
                '--output', str(partial), repo + package.name)
            partial.rename(package)
        compat = ROOT / 'cuda_compat13'
        compat.mkdir(exist_ok=True)
        run('dpkg-deb', '-x', str(package), str(compat))
        with (WORK / 'pip_freeze.txt').open('w') as stream:
            subprocess.run([python, '-m', 'pip', 'freeze'], stdout=stream, check=True, env=env)
        save('environment_status.json', {'status': 'installed_not_gpu_validated', 'env': str(ENV),
             'compat': str(compat / 'usr/local/cuda-13.0/compat'), 'package': package.name,
             'index': index, 'proxy': proxy, 'fallback_index': fallback_index,
             'lock_sha256': sha(lockfile) if locked else None})
    except Exception as exc:
        save('environment_status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    assert Path(__file__).resolve().parent == WORK
    if sys.argv[1:] == ['--launch']:
        for mode in ('weights', 'environment'):
            with (WORK / f'{mode}.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', __file__, mode], stdout=log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
                print(json.dumps({'mode': mode, 'pid': process.pid}), flush=True)
    elif sys.argv[1:] == ['weights']:
        weights()
    elif sys.argv[1:] == ['environment']:
        environment()
    else:
        raise SystemExit('Explicit --launch, weights or environment required')
