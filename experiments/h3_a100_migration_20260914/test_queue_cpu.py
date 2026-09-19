"""Read-only CPU gates for the BCD queue, before detached launch."""
import os
import unittest
from unittest.mock import patch
import prepare_cuda as prepare
import run_abcd_cuda as queue


class QueueTests(unittest.TestCase):
    def test_user_selected_only_bcd(self):
        self.assertEqual(queue.CASES, ('B', 'C', 'D'))

    def test_original_to_cuda_math_unchanged(self):
        for case in ('B', 'C', 'D'):
            original = queue.ROOT / 'frozen' / prepare.SOURCES[case] / prepare.REL / 'openvdn_npu.py'
            expected = prepare.patch_kernel(original.read_text())
            deployed = queue.WORK / f'candidates/{case}' / prepare.REL / 'openvdn_npu.py'
            self.assertEqual(expected, deployed.read_text())

    def test_real_vlm_evidence(self):
        manifest = queue.read(queue.WORK / 'cuda_manifest.json')
        self.assertTrue(queue.admission(manifest)['reused_real_vlm_decision'])

    def test_reject_changed_edit(self):
        manifest = queue.read(queue.WORK / 'cuda_manifest.json')
        manifest['sample']['edit_prompt'] = 'different edit'
        with self.assertRaises(RuntimeError):
            queue.admission(manifest)

    def test_reject_busy_cards(self):
        rows = [['2', queue.UUIDS[0], '70000', '0'], ['3', queue.UUIDS[1], '0', '0']]
        with patch.object(queue, 'gpu_rows', return_value=rows), patch.object(queue.subprocess, 'check_output', return_value=''):
            with self.assertRaises(RuntimeError):
                queue.assert_idle()

    def test_selected_mask(self):
        for case in queue.CASES:
            self.assertEqual(queue.env_for(case)['CUDA_VISIBLE_DEVICES'], ','.join(queue.UUIDS))
            self.assertEqual(queue.env_for(case)['ZHONGHAO_H3_OPENVDN'], '1')

    def test_owned_pid_binding(self):
        identity = queue.pid_identity(os.getpid())
        self.assertGreater(identity[0], 0)
        self.assertEqual(identity[1], os.getppid())


if __name__ == '__main__':
    unittest.main(verbosity=2)
