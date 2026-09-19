"""Transfer our verified H3 tree using sequential archives (SFS disallows rename)."""
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
CONTROL = ROOT / 'migration_30674_tar_20260914'
STAGE = Path('/temp/zhonghao/h3_to_30674_tar_20260914')
PARTS = ['a100_v1', 'frozen', 'cuda_compat13', 'env_cuda_v1', 'models']


def status(phase, **fields):
    data = dict(phase=phase, pid=os.getpid(), host=socket.gethostname(),
                updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)
    tmp = CONTROL / 'status.tmp'
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(CONTROL / 'status.json')
    print(json.dumps(data), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(32 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def export():
    assert socket.gethostname() == 'os-node-created-dvv7x'
    assert not STAGE.exists()
    assert shutil.disk_usage('/temp').free > 220 * 1024**3
    STAGE.mkdir(parents=True)
    status('exporting_h3_archive', stage=str(STAGE))
    with (STAGE / 'h3.tar').open('xb') as out:
        subprocess.run(['tar', '-cf', '-', '-C', str(ROOT), *PARTS], stdout=out, check=True)
    status('exporting_python_runtime')
    runtime = Path('/cache/envs/swiftcam')
    libraries = [str(p.relative_to(runtime)) for p in sorted((runtime / 'lib').glob('*.so*'))]
    with (STAGE / 'python312.tar').open('xb') as out:
        subprocess.run(['tar', '--exclude=lib/python3.12/site-packages', '-cf', '-',
                        '-C', str(runtime), 'bin/python3.12', 'lib/python3.12', *libraries],
                       stdout=out, check=True)
    entries = []
    for name in ('h3.tar', 'python312.tar'):
        status('hashing_export', archive=name)
        path = STAGE / name
        entries.append(dict(name=name, size=path.stat().st_size, sha256=sha(path)))
    with (STAGE / 'manifest.json').open('x') as stream:
        json.dump(entries, stream, indent=2)
    (STAGE / 'READY').mkdir()
    status('export_ready', stage=str(STAGE), archives=entries)


def receive():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    assert all(not (ROOT / part).exists() for part in PARTS)
    assert shutil.disk_usage('/cache').free > 220 * 1024**3
    status('waiting_for_export', stage=str(STAGE))
    deadline = time.monotonic() + 12 * 3600
    while not (STAGE / 'READY').is_dir():
        if time.monotonic() > deadline or (CONTROL / 'STOP').exists():
            raise RuntimeError('Export wait expired or STOP requested')
        time.sleep(10)
    entries = json.loads((STAGE / 'manifest.json').read_text())
    assert [e['name'] for e in entries] == ['h3.tar', 'python312.tar']
    for entry in entries:
        path = STAGE / entry['name']
        status('verifying_archive', archive=entry['name'])
        assert path.stat().st_size == entry['size'] and sha(path) == entry['sha256']
        destination = ROOT if entry['name'] == 'h3.tar' else ROOT / 'python312_runtime'
        destination.mkdir(parents=True, exist_ok=True)
        status('extracting_archive', archive=entry['name'])
        subprocess.run(['tar', '-xf', str(path), '-C', str(destination)], check=True)
    status('assets_imported_runtime_setup_required', archives=entries)


def main():
    mode = sys.argv[1]
    assert mode in ('export', 'receive')
    CONTROL.mkdir(parents=True, exist_ok=True)
    if sys.argv[2:] == ['--launch']:
        # Stop only the previous cooperative migration receiver, not GPU jobs.
        previous = ROOT / 'migration_30674_20260914'
        if mode == 'receive' and previous.exists():
            (previous / 'STOP').touch(exist_ok=True)
        with (CONTROL / (mode + '.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__, mode],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'mode': mode, 'pid': proc.pid}))
    else:
        try:
            (export if mode == 'export' else receive)()
        except BaseException as exc:
            status('failed', error=repr(exc), automatic_retry=False)
            raise


if __name__ == '__main__':
    main()
