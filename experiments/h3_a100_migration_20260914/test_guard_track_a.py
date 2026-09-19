import unittest
from d_memory_guard_track_a import pressure_reason, MAX_USAGE_BYTES

def sample(usage, cache=0, shmem=0, oom=2):
    return {'usage_in_bytes': usage, 'limit_in_bytes': 250999996416,
            'oom_control': {'oom_kill': oom},
            'stat': {'cache': cache, 'shmem': shmem}}

class TrackAGuardTests(unittest.TestCase):
    def test_previous_failure_allowed(self):
        self.assertIsNone(pressure_reason(sample(250117574656,33529815040,1003544576),2))
    def test_threshold_exact_allowed(self):
        self.assertIsNone(pressure_reason(sample(MAX_USAGE_BYTES+5,5),2))
    def test_threshold_above_stops(self):
        self.assertIn('non-file',pressure_reason(sample(MAX_USAGE_BYTES+6,5),2))
    def test_shmem_not_deducted(self):
        self.assertIn('non-file',pressure_reason(sample(MAX_USAGE_BYTES+1,100,100),2))
    def test_new_oom_always_stops(self):
        self.assertIn('increased',pressure_reason(sample(200,100,0,3),2))
    def test_original_guard_unchanged(self):
        from d_memory_guard import pressure_reason as original
        self.assertIn('proactive',original(sample(MAX_USAGE_BYTES+1,100),2))
    def test_invalid_cache_does_not_hide_usage(self):
        self.assertIn('non-file',pressure_reason(sample(MAX_USAGE_BYTES+1,1,2),2))

if __name__ == '__main__':
    unittest.main()
