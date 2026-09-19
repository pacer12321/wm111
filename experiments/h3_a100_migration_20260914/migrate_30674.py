"""Detached, non-destructive transfer of our H3 assets via the shared /temp volume."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/cache/zhonghao/h3')
STAGE = Path('/temp/zhonghao/h3_to_30674_20260914')
CONTROL = ROOT / 'migration_30674_20260914'
PARTS = ('a100_v1', 'frozen', 'cuda_compat13', 'env_cuda_v1', 'models')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def status(phase, **fields):
    data = dict(phase=phase, pid=os.getpid(), host=socket.gethostname(),
                updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)
    write(CONTROL / 'status.json', data)
    print(json.dumps(data), flush=True)


def copy(source, destination, extra=()):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-aH', '--info=progress2', '--stats', *extra,
                    str(source) + '/', str(destination) + '/'], check=True)


def export():
    assert socket.gethostname() == 'os-node-created-dvv7x'
    assert not STAGE.exists(), 'Single-use export: shared stage already exists'
    assert shutil.disk_usage('/temp').free > 220 * 1024**3
    STAGE.mkdir(parents=True)
    for part in PARTS:
        status('exporting', part=part)
        copy(ROOT / part, STAGE / 'h3' / part)
    status('exporting_python_runtime')
    runtime = STAGE / 'h3/python312_runtime'
    (runtime / 'bin').mkdir(parents=True)
    shutil.copy2('/cache/envs/swiftcam/bin/python3.12', runtime / 'bin/python3.12')
    copy(Path('/cache/envs/swiftcam/lib/python3.12'), runtime / 'lib/python3.12',
         ('--exclude=site-packages/',))
    # Model bytes are checked end-to-end by rsync, and SHA256 is frozen here
    # for a second independent check after importing onto the destination disk.
    entries = []
    model_files = sorted(p for p in (STAGE / 'h3/models').rglob('*') if p.is_file())
    for i, path in enumerate(model_files):
        entries.append(dict(path=str(path.relative_to(STAGE / 'h3')),
                            size=path.stat().st_size, sha256=sha(path)))
        status('hashing_exported_models', completed=i + 1, total=len(model_files))
    write(STAGE / 'models_sha256.json', entries)
    write(STAGE / 'READY.json', dict(source_host=socket.gethostname(),
                                   exported_parts=list(PARTS), model_files=len(entries)))
    status('export_ready', stage=str(STAGE))


def receive():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    assert all(not (ROOT / part).exists() for part in PARTS), 'Destination already has assets'
    assert shutil.disk_usage('/cache').free > 220 * 1024**3
    status('waiting_for_shared_export')
    deadline = time.monotonic() + 12 * 3600
    while not (STAGE / 'READY.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Shared export not ready within12h')
        if (CONTROL / 'STOP').exists():
            raise RuntimeError('Migration STOP requested')
        time.sleep(10)
    for part in (*PARTS, 'python312_runtime'):
        status('importing', part=part)
        copy(STAGE / 'h3' / part, ROOT / part)
    entries = json.loads((STAGE / 'models_sha256.json').read_text())
    for i, entry in enumerate(entries):
        path = ROOT / entry['path']
        assert path.stat().st_size == entry['size'] and sha(path) == entry['sha256'], str(path)
        status('verifying_imported_models', completed=i + 1, total=len(entries))
    status('assets_imported_runtime_setup_required', model_files=len(entries))


def main():
    mode = sys.argv[1]
    assert mode in ('export', 'receive')
    CONTROL.mkdir(parents=True, exist_ok=True)
    if sys.argv[2:] == ['--launch']:
        with (CONTROL / (mode + '.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, mode],
                                     stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'mode': mode, 'pid': child.pid}))
        return
    try:
        (export if mode == 'export' else receive)()
    except BaseException as exc:
        status('failed', error=repr(exc), automatic_retry=False)
        raise


if __name__ == '__main__':
    main()
