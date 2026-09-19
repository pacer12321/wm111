"""Wait for verified import, relocate cloned venv, validate, then launch only D."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/cache/zhonghao/h3')
CONTROL = ROOT / 'runtime_setup_30674_20260914'
ENV = ROOT / 'env_cuda_v1'
RUNTIME = ROOT / 'python312_runtime'


def status(phase, **fields):
    data = dict(phase=phase, pid=os.getpid(), updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)
    temp = CONTROL / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(CONTROL / 'status.json')
    print(json.dumps(data), flush=True)


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    CONTROL.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ['--launch']:
        with (CONTROL / 'setup.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'runtime_setup_pid': proc.pid}))
        return
    try:
        status('waiting_for_verified_import')
        deadline = time.monotonic() + 12 * 3600
        while True:
            path = ROOT / 'migration_30674_tar_20260914/status.json'
            stage = json.loads(path.read_text()) if path.exists() else {}
            if stage.get('phase') == 'assets_imported_runtime_setup_required':
                break
            if stage.get('phase') == 'failed' or time.monotonic() > deadline or (CONTROL / 'STOP').exists():
                raise RuntimeError('Migration failed, timed out or STOP requested; do not launch D')
            time.sleep(10)
        status('relocating_cloned_venv')
        config = ENV / 'pyvenv.cfg'
        assert 'home = /cache/envs/swiftcam/bin' in config.read_text()
        shutil.copy2(config, CONTROL / 'original_pyvenv.cfg')
        recorded = {}
        for name in ('python', 'python3', 'python3.12'):
            path = ENV / 'bin' / name
            if path.is_symlink():
                target = os.readlink(path)
                assert target in ('python', 'python3', 'python3.12', '/cache/envs/swiftcam/bin/python',
                                  '/cache/envs/swiftcam/bin/python3.12'), target
                recorded[name] = target
        assert set(recorded) == {'python', 'python3', 'python3.12'}, 'Unexpected cloned interpreter layout'
        (CONTROL / 'original_python_links.json').write_text(json.dumps(recorded, indent=2))
        # Only replace known symlinks in our newly imported private venv, not source files.
        for name in recorded:
            (ENV / 'bin' / name).unlink()
        environment = dict(os.environ)
        for name in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'LD_AUDIT'):
            environment.pop(name, None)
        environment.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES=
                           'GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670,GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3',
                           LD_LIBRARY_PATH=str(ROOT / 'cuda_compat13/usr/local/cuda-13.0/compat') + ':' +
                           str(ENV / 'lib') + ':' + str(RUNTIME / 'lib'))
        subprocess.run([str(RUNTIME / 'bin/python3.12'), '-m', 'venv', '--without-pip', str(ENV)],
                       env=environment, check=True, timeout=60)
        status('checking_locked_runtime')
        subprocess.run([str(ENV / 'bin/python'), '-m', 'pip', 'check'], env=environment, check=True, timeout=120)
        probe = ('import sys,torch,vllm,transformers,diffusers,av,json; '
                 'assert sys.version_info[:2]==(3,12); '
                 'assert torch.__version__=="2.11.0+cu130"; '
                 'assert vllm.__version__=="0.26.0"; '
                 'assert transformers.__version__=="5.17.0"; '
                 'assert diffusers.__version__=="0.38.0"; '
                 'print(json.dumps({"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda}))')
        subprocess.run([str(ENV / 'bin/python'), '-c', probe], env=environment, check=True, timeout=120)
        status('runtime_verified')
        subprocess.run([str(ENV / 'bin/python'), str(ROOT / 'run_d_30674.py'), '--launch'],
                       env=environment, check=True, timeout=120)
        # Keep runtime_verified for the child's admission check.
        (CONTROL / 'd_dispatch_succeeded.json').write_text(json.dumps({'launched': True}))
    except BaseException as exc:
        status('failed', error=repr(exc), automatic_model_retry=False)
        raise


if __name__ == '__main__':
    main()
