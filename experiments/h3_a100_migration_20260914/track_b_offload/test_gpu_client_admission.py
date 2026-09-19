import unittest
from gpu_client_admission import check_clients


class AdmissionTests(unittest.TestCase):
    def test_own_container_id_does_not_need_to_equal_host_id(self):
        self.assertEqual(check_clients({10:100,20:200},{10:100},{20:200}),{20:200})
    def test_foreign_client_rejected(self):
        with self.assertRaises(RuntimeError): check_clients({30:300},{10:100},{20:200})
    def test_reused_baseline_pid_rejected(self):
        with self.assertRaises(RuntimeError): check_clients({10:999},{10:100},{20:200})
    def test_reused_own_pid_rejected(self):
        with self.assertRaises(RuntimeError): check_clients({20:999},{10:100},{20:200})
    def test_empty_is_not_an_ownership_claim(self):
        self.assertEqual(check_clients({}, {}, {20:200}), {})


if __name__=='__main__': unittest.main()
