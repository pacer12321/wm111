"""CPU-only tests: wait for shared cards without operating on occupants."""
import unittest
from unittest.mock import patch
import continue_cd as cd


class WaitTests(unittest.TestCase):
    @patch.object(cd.q, 'gpu_rows', return_value=[])
    @patch.object(cd.time, 'sleep')
    @patch.object(cd.Path, 'exists', return_value=False)
    def test_busy_then_two_idle_observations(self, exists, sleep, rows):
        with patch.object(cd.q, 'assert_idle', side_effect=[RuntimeError('busy'), None, None]) as idle:
            cd.wait_idle(90)
            self.assertEqual(idle.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    @patch.object(cd.Path, 'exists', return_value=True)
    def test_stop_prevents_launch(self, exists):
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            cd.wait_idle(90)

    @patch.object(cd.Path, 'exists', return_value=False)
    @patch.object(cd.q, 'gpu_rows', side_effect=RuntimeError('UUID changed'))
    def test_changed_binding_fails_immediately(self, rows, exists):
        with self.assertRaisesRegex(RuntimeError, 'UUID changed'):
            cd.wait_idle(90)


if __name__ == '__main__':
    unittest.main(verbosity=2)
