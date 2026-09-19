"""Refresh observer/launcher digests only while the diagnostic lease is free.

Prior run manifests are immutable snapshots. Original and copied model code
must still match exactly; this cannot authorize changes to a candidate model.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path

root = Path('/cache/zhonghao/h3/profiling_bd_v1')
assert Path(__file__).resolve().parent == root
with (root / 'run.lock').open('r+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'failed'
    assert status['cleanup_completed'] and status['selected_cards_verified_idle_after_cleanup']
    manifest = json.loads((root / 'manifest.json').read_text())
    for copy in manifest['copies'].values():
        for relative, expected in copy['files'].items():
            assert hashlib.sha256((Path(copy['vendor']) / relative).read_bytes()).hexdigest() == expected
    manifest['diagnostic_code'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list(root.glob('*.py')) + list(root.glob('*.sh'))}
    temporary = root / 'manifest.refreshed.json'
    with temporary.open('x') as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary, root / 'manifest.json')
    print('Diagnostic digests refreshed; prior run manifest preserved; candidate files unchanged.')
