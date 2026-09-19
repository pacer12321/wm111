import copy
import json
from pathlib import Path
import tempfile
import unittest
from retrain_readiness import REQUIRED, sha, verify


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.manifest = root / "manifest.jsonl"
        self.manifest.write_text("\n".join(json.dumps(dict(sample_id=str(i))) for i in range(2000)))
        self.samples = root / "samples";self.samples.mkdir()
        self.output = root / "out"
        for i in range(4):
            for kind in ("video", "prompt"):
                for ext in ("pt", "done"):
                    (self.samples/f"{kind}_{i:06d}.{ext}").write_bytes(b"unit-test-placeholder")
        code = {}
        for i in range(5):
            path = root/f"code{i}.py";path.write_text("# fixture\n")
            code[str(path)]=sha(path)
        receipt=root/"receipt.json";receipt.write_text('{"fixture":true}')
        self.report=dict(schema="savie-retrain-readiness-v1",audio_input_policy="fixed-silent",
            starting_point="fresh-dmd8-new-lora",train_token_skip=False,audio_objective=False,
            sample_dir=str(self.samples),manifest_sha256=sha(self.manifest),code_sha256=code,
            initial_buffer_size=4,checks={k:dict(passed=True,receipt_path=str(receipt),
                receipt_sha256=sha(receipt)) for k in REQUIRED})

    def run_gate(self, report=None):
        return verify(report or self.report,self.manifest,self.samples,self.output,"fixed-silent")

    def test_streaming_does_not_require_all_2000_encodes(self):
        self.assertTrue(self.run_gate()["passed"])

    def test_tag_only_old_preflight_fails(self):
        with self.assertRaises(ValueError):
            self.run_gate(dict(passed=True))

    def test_every_required_check_must_have_passed(self):
        for name in REQUIRED:
            report=copy.deepcopy(self.report);report["checks"][name]["passed"]=False
            with self.assertRaisesRegex(ValueError,name):self.run_gate(report)

    def test_changed_code_and_missing_evidence_rejected(self):
        path=Path(next(iter(self.report["code_sha256"])));path.write_text("# changed")
        with self.assertRaisesRegex(ValueError,"code changed"):self.run_gate()

    def test_old_checkpoint_cannot_silently_resume(self):
        self.output.mkdir();(self.output/"savie_step001000.pt").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError,"old training"):self.run_gate()

    def test_wrong_policy_or_data_directory_rejected(self):
        for field,value in [("audio_input_policy","other"),("sample_dir","/old-cache")]:
            report=copy.deepcopy(self.report);report[field]=value
            with self.assertRaises(ValueError):self.run_gate(report)


if __name__ == "__main__":unittest.main()
