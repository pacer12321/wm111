"""CPU-only command tests: no model imports, network access or GPU allocations."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

if os.name == 'nt':
    # No flock calls occur in run_case; stub import only for Windows CPU tests.
    with patch.dict(sys.modules, {'fcntl': types.ModuleType('fcntl')}):
        import run_abcd_cuda as q
else:
    import run_abcd_cuda as q


class SmokeEnd(Exception):
    pass


class Reply:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Opener:
    def open(self, *args, **kwargs): return Reply()


class Server:
    pid = 999999999
    def __init__(self, command, **kwargs): self.command = command
    def poll(self): return None


class Socket:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def bind(self, *args): pass


class TestSerialCommand(unittest.TestCase):
    def capture(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'a100_v1'
            work.mkdir()
            calls = []
            def request(*args):
                calls.append(args[3])
                raise SmokeEnd()
            with patch.multiple(q, ROOT=root, WORK=work, SERVER_EXTRA_ARGS=extra,
                                assert_idle=lambda: None, env_for=lambda *_: {},
                                pid_identity=lambda *_: None, capture_tree=lambda *_: None,
                                cleanup=lambda *_: None, request=request, update=lambda *a, **k: None), \
                 patch.object(q.socket,'socket',Socket), \
                 patch.object(q.subprocess,'Popen',Server), \
                 patch.object(q.urllib.request,'build_opener',lambda *_: Opener()):
                with self.assertRaises(SmokeEnd):
                    q.run_case('D', {'cases': {'D': {'model_files': {}}}, 'sample': {}})
            self.assertEqual(calls, [2])
            cmd = json.loads((work/'results/D/launch.json').read_text())['command']
            # Normalize temporary model path, everything else must remain identical.
            cmd[4] = '<model>'
            return cmd

    def test_only_one_added_startup_argument(self):
        default = self.capture(())
        serial = self.capture(('--disable-multithread-weight-load',))
        self.assertEqual(serial, default + ['--disable-multithread-weight-load'])


if __name__ == '__main__':
    unittest.main()
