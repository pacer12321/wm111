import unittest
from summarize_events import summarize


class SummaryTests(unittest.TestCase):
    def test_no_double_counting(self):
        value = {'case': 'D', 'rank': 0, 'forward': 4, 'metric': 'test', 'rows': [
            {'id': 0, 'parent': None, 'name': 'dit_forward', 'elapsed_ms': 100},
            {'id': 1, 'parent': 0, 'name': 'block', 'elapsed_ms': 80},
            {'id': 2, 'parent': 1, 'name': 'linear/S', 'elapsed_ms': 30},
            {'id': 3, 'parent': 2, 'name': 'linear/scan', 'elapsed_ms': 20}]}
        result = summarize(value)
        self.assertEqual(sum(r['self_ms'] for r in result['nonoverlapping_scope_self_intervals']), 100)
        self.assertEqual(result['warnings'], [])

    def test_invalid_parent_rejected(self):
        value = {'rows': [{'id': 0, 'parent': 4, 'name': 'dit_forward', 'elapsed_ms': 1}]}
        with self.assertRaises(AssertionError):
            summarize(value)


if __name__ == '__main__':
    unittest.main()
