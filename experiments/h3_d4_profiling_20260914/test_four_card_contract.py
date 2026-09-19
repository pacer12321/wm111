"""Static allocation/observer checks without importing torch or remote code."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class ContractTests(unittest.TestCase):
    def test_launcher_only_allocates_four(self):
        text = (ROOT / 'launch_server.sh').read_text()
        self.assertIn('export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3\n', text)
        self.assertNotIn('0,1,2,3,4', text)
        self.assertIn('--num-gpus 4 --usp 4', text)
        self.assertIn('--text-encoder-tp-size 4', text)
        self.assertIn('--vae-patch-parallel-size 4', text)
        self.assertLess(len('/cache/zhonghao/h3/tmp/p4_' + '0'*32 + '_D/' + '0'*36), 107)

    def test_supervisor_only_runs_d(self):
        text = (ROOT / 'run_profiles.py').read_text()
        ast.parse(text)
        self.assertIn("for case in ('D',):", text)
        self.assertIn('cards=(0, 1, 2, 3)', text)
        self.assertNotIn('for rank in range(8)', text)

    def test_head_partition(self):
        self.assertEqual(56 % 4, 0)
        self.assertEqual(56 // 4, 14)


if __name__ == '__main__':
    unittest.main()
