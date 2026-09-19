"""Install the explicitly requested >250 decimal GB threshold; no model launch."""
import fcntl
import hashlib
from pathlib import Path
import socket


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    root = Path('/cache/zhonghao/h3')
    with (root/'a100_v1/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = root/'d_memory_guard.py'
        staged = root/'d_memory_guard_250_staged.py'
        old, new = current.read_bytes(), staged.read_bytes()
        assert hashlib.sha256(old).hexdigest() == '1ee7021274784293344206686f25a8cba726daffcd135b926a85f4a9ddabd792'
        assert hashlib.sha256(new).hexdigest() == 'db57170026764be6203f9288216cb8b425a139a5605e68862869b64649c97ca0'
        backup = root/'a100_v1/history/d_memory_guard_before_250GB_20260915.py'
        with backup.open('xb') as handle:
            handle.write(old)
        staged.replace(current)
        print('Installed strict usage >250000000000 bytes guard; original preserved:', backup)


if __name__ == '__main__':
    main()
