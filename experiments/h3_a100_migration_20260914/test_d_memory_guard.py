import unittest
from d_memory_guard import MAX_USAGE_BYTES, pressure_reason


def sample(usage, cache=0, shmem=0, oom=2):
    return {'usage_in_bytes': usage, 'limit_in_bytes': 250999996416,
            'oom_control': {'oom_kill': oom},
            'stat': {'cache': cache, 'shmem': shmem}}


class TestPressure(unittest.TestCase):
    def test_decimal_gb(self):
        self.assertEqual(MAX_USAGE_BYTES, 250_000_000_000)

    def test_one_byte_below(self):
        self.assertIsNone(pressure_reason(sample(MAX_USAGE_BYTES-1), 2))

    def test_exact_threshold_does_not_stop(self):
        self.assertIsNone(pressure_reason(sample(MAX_USAGE_BYTES), 2))

    def test_one_byte_above(self):
        self.assertIn('proactive', pressure_reason(sample(MAX_USAGE_BYTES+1), 2))

    def test_previous_trigger_no_longer_stops(self):
        self.assertIsNone(pressure_reason(sample(240278036480, cache=27036033024,
                                               shmem=3704295424), 2))

    def test_total_usage_threshold_not_non_file(self):
        self.assertIn('proactive', pressure_reason(sample(MAX_USAGE_BYTES+1, cache=40_000_000_000), 2))

    def test_new_oom(self):
        self.assertIn('increased', pressure_reason(sample(100, oom=3), 2))

    def test_old_oom_not_current_failure(self):
        self.assertIsNone(pressure_reason(sample(100, oom=2), 2))


if __name__ == '__main__':
    unittest.main()
