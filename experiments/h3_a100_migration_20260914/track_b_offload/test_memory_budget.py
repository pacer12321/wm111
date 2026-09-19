import json
from pathlib import Path
import unittest

from memory_budget import budget, encoder_budget


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        cls.manifest = json.loads((root / "real_manifest.json").read_text(encoding="utf-8"))
        cls.aux = json.loads((root / "aux_budget_headers.json").read_text(encoding="utf-8"))

    def test_encoder_replication_not_divided_by_two(self):
        result = encoder_budget(self.aux["components"]["text_encoder"])
        self.assertEqual(result["per_rank_bytes"], 26_348_887_520)
        self.assertEqual(result["vision_replicated_bytes"], 1_190_533_600)

    def test_pinned_weights_not_added_twice_and_all_gpu_buffers_counted(self):
        result = budget(self.manifest, self.aux)
        self.assertEqual(result["dit"]["main_pinned_shard_bytes_each_rank"] * 2, 68_836_504_000)
        self.assertEqual(result["dit"]["gpu_allgather_buffers_total_each_rank"], 4_130_190_240)
        case = result["cases"]["UNVERIFIED_if_checkpoint_16bit_approx_includes_header"]
        self.assertEqual(case["cpu_payload_plus_provided_baseline"], 188_853_577_288)

    def test_video_unknown_does_not_pass_gate(self):
        result = budget(self.manifest, self.aux)
        self.assertEqual(result["status"], "budget_gate_incomplete")
        self.assertIsNone(result["video_vae_runtime_fp32_bytes_each_rank"])
        self.assertTrue(all(name.startswith("UNVERIFIED") for name in result["cases"]))

    def test_provided_baseline_is_separate_term(self):
        low = budget(self.manifest, self.aux, baseline_bytes=0)
        high = budget(self.manifest, self.aux, baseline_bytes=21_000_000_000)
        for name in low["cases"]:
            self.assertEqual(high["cases"][name]["cpu_payload_plus_provided_baseline"] -
                             low["cases"][name]["cpu_payload_plus_provided_baseline"], 21_000_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
