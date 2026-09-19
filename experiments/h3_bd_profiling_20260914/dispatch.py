"""Detach one explicit, bounded diagnostic supervisor from the SSH session."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

root = Path('/cache/zhonghao/h3/profiling_bd_v1')
assert Path(__file__).resolve().parent == root
assert sys.argv[1:] == ['--start-b-then-d']
status = json.loads((root / 'status.json').read_text())
assert status['phase'] == 'failed' and status['cleanup_completed']
log = root / ('dispatch_' + uuid.uuid4().hex + '.log')
with log.open('xb') as stream:
    child = subprocess.Popen(['/cache/zhonghao/h3/env/bin/python', '-u', str(root / 'run_profiles.py'),
                              '--run-b-then-d'], cwd=root, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'supervisor_pid': child.pid, 'log': str(log)}))
