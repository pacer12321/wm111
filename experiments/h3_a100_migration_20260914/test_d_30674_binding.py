"""CPU-only resource guard regression; never imports CUDA or starts a model."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

stub = types.ModuleType('run_abcd_cuda')
stub.env_for = lambda case='D': {'LD_LIBRARY_PATH': '/private_cuda:/env/lib'}
sys.modules['run_abcd_cuda'] = stub
if sys.platform == 'win32':
    sys.modules.setdefault('fcntl', types.ModuleType('fcntl'))
path = Path(__file__).with_name('run_d_30674.py')
spec = importlib.util.spec_from_file_location('d_queue', path)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.rows = [['0', d.UUIDS[0], '856', '0'], ['1', d.UUIDS[1], '0', '0']]

    def test_empty_accepted(self):
        with patch.object(d, 'gpu_rows', return_value=self.rows), patch.object(d.subprocess, 'check_output', return_value=''):
            d.assert_ready()

    def test_explicit_residual_accepted(self):
        with patch.object(d, 'gpu_rows', return_value=self.rows), patch.object(d.subprocess, 'check_output', return_value=d.UUIDS[0] + ', 1550469\n'):
            d.assert_ready()

    def test_unknown_task_rejected(self):
        with patch.object(d, 'gpu_rows', return_value=self.rows), patch.object(d.subprocess, 'check_output', return_value=d.UUIDS[1] + ', 1550469\n'):
            with self.assertRaises(RuntimeError):
                d.assert_ready()

    def test_busy_residual_rejected(self):
        self.rows[0][3] = '1'
        with patch.object(d, 'gpu_rows', return_value=self.rows), patch.object(d.subprocess, 'check_output', return_value=''):
            with self.assertRaises(RuntimeError):
                d.assert_ready()

    def test_own_holders_not_mistaken_for_idle(self):
        self.rows[1][2] = '72000'
        with patch.object(d, 'gpu_rows', return_value=self.rows), patch.object(d.subprocess, 'check_output', return_value=''):
            with self.assertRaises(RuntimeError):
                d.assert_ready()

    def test_wrong_mapping_rejected(self):
        text = '\n'.join(', '.join(r) for r in self.rows).replace('0, GPU-', '2, GPU-', 1)
        with patch.object(d.subprocess, 'check_output', return_value=text):
            with self.assertRaises(AssertionError):
                d.gpu_rows()


if __name__ == '__main__':
    unittest.main()
