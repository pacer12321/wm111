"""CPU unit tests of the real admission function, NOT actual VLM evidence.

Only a/vlm_gate.py executes as production code here. The lower-level completed
VLM verifier is explicitly mocked: its own tests cover full inference evidence.
All receipts/status/results below are synthetic files in TemporaryDirectory;
no model, accelerator, remote host, real admission, or real result is accessed.
"""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "color_vlm_admission_under_test", HERE / "a" / "vlm_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        temporary = stack.enter_context(tempfile.TemporaryDirectory())
        root = Path(temporary).resolve()
        code = root / "vlm"
        code.mkdir()
        (root / "a").mkdir()
        run_id = "a" * 32
        run = root / "vlm_results" / "synthetic_only" / "runs" / ("20260914T010000Z_" + run_id)
        run.mkdir(parents=True)
        receipt = root / "vlm_admission.json"

        def write(path, value):
            path.write_text(json.dumps(value), encoding="utf-8")

        def record(path):
            path = Path(path)
            if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
                raise RuntimeError("Noncanonical CPU fixture evidence")
            payload = path.read_bytes()
            return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))

        source = dict(path=str(root / "source.synthetic"), sha256="b" * 64, size_bytes=123)
        prompt = "SYNTHETIC TEST ONLY: preserve timing and change shirt color."
        result = dict(source=copy.deepcopy(source), edit_prompt=prompt,
                      classification=dict(classification="preserve"),
                      fixture_only=True, actual_inference=False)
        write(run / "vlm_status.json", dict(fixture_only=True, actual_inference=False, run_id=run_id))
        write(run / "worker_result.json", result)
        row = dict(schema_version=1, run_id=run_id, run_directory=str(run),
                   status_sha256=record(run / "vlm_status.json")["sha256"],
                   result_sha256=record(run / "worker_result.json")["sha256"],
                   source=copy.deepcopy(source), edit_prompt=prompt, classification="preserve",
                   root_reviewed_real_evidence=True)
        # This flag exercises the real envelope schema, NOT a real review claim:
        # the entire file lives in a disposable fixture with a mocked verifier.
        write(receipt, row)
        verified = dict(result=result, fixture_only=True, actual_inference=False)
        checker = SimpleNamespace(__file__=str(code / "run_vlm.py"),
                                  verify_completed=mock.Mock(return_value=verified))
        contract = SimpleNamespace(__file__=str(code / "vlm_contract.py"),
                                   run_directory=mock.Mock(return_value=run),
                                   record=mock.Mock(side_effect=record))
        modules = {"run_vlm": checker, "vlm_contract": contract}
        # Keep production import resolution and sys.path mutations local to the test.
        stack.enter_context(mock.patch.object(gate, "importlib", SimpleNamespace(
            import_module=mock.Mock(side_effect=lambda name: modules[name]))))
        stack.enter_context(mock.patch.object(gate, "sys", SimpleNamespace(path=list(sys.path))))
        for name, value in (("ROOT", root), ("VLM_CODE", code), ("ADMISSION", receipt),
                            ("__file__", str(root / "a" / "vlm_gate.py"))):
            stack.enter_context(mock.patch.object(gate, name, value))
        return SimpleNamespace(root=root, code=code, run=run, run_id=run_id, receipt=receipt,
                               source=source, prompt=prompt, result=result, row=row, verified=verified,
                               checker=checker, contract=contract, write=write, record=record)

    def test_valid_synthetic_envelope_delegates_to_real_admission_function(self):
        f = self.fixture()
        admitted = gate.verify_admission(f.source, f.prompt)
        self.assertEqual(admitted["admission"], f.record(f.receipt))
        self.assertIs(admitted["evidence"], f.verified)
        self.assertTrue(admitted["evidence"]["fixture_only"])
        self.assertFalse(admitted["evidence"]["actual_inference"])
        f.contract.run_directory.assert_called_once_with(str(f.run), f.run_id)
        f.checker.verify_completed.assert_called_once_with(str(f.run), f.run_id)
        self.assertEqual(f.contract.record.call_args_list, [
            mock.call(f.run / "vlm_status.json"), mock.call(f.run / "worker_result.json"),
            mock.call(f.receipt)])

    def test_missing_receipt_fails_before_verifier(self):
        f = self.fixture()
        with mock.patch.object(gate, "ADMISSION", f.root / "missing.json"):
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_receipt_source_mismatch_fails(self):
        f = self.fixture(); f.row["source"]["sha256"] = "c" * 64
        f.write(f.receipt, f.row)
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_receipt_prompt_mismatch_fails(self):
        f = self.fixture(); f.row["edit_prompt"] = "other prompt"
        f.write(f.receipt, f.row)
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_receipt_change_and_uncertain_fail(self):
        for label in ("change", "uncertain"):
            with self.subTest(label=label):
                f = self.fixture(); f.row["classification"] = label
                f.write(f.receipt, f.row)
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_not_called()

    def test_actual_verifier_source_mismatch_fails(self):
        f = self.fixture(); f.result["source"]["sha256"] = "c" * 64
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_called_once()

    def test_actual_verifier_prompt_mismatch_fails(self):
        f = self.fixture(); f.result["edit_prompt"] = "other prompt"
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_called_once()

    def test_actual_classifier_change_and_uncertain_cannot_use_preserve_receipt(self):
        for label in ("change", "uncertain"):
            with self.subTest(label=label):
                f = self.fixture(); f.result["classification"]["classification"] = label
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_called_once()

    def test_status_sha_mismatch_fails_before_verifier(self):
        f = self.fixture(); f.row["status_sha256"] = "0" * 64
        f.write(f.receipt, f.row)
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_result_sha_mismatch_fails_before_verifier(self):
        f = self.fixture(); f.row["result_sha256"] = "0" * 64
        f.write(f.receipt, f.row)
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_malformed_sha_fails_before_verifier(self):
        for name in ("status_sha256", "result_sha256"):
            with self.subTest(name=name):
                f = self.fixture(); f.row[name] = "not-a-sha"
                f.write(f.receipt, f.row)
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_not_called()

    def test_changed_status_and_result_bytes_fail(self):
        for name in ("vlm_status.json", "worker_result.json"):
            with self.subTest(name=name):
                f = self.fixture(); f.write(f.run / name, dict(fixture_only=True, changed=True))
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_not_called()

    def test_wrong_checker_or_contract_path_fails(self):
        for name in ("checker", "contract"):
            with self.subTest(name=name):
                f = self.fixture()
                dependency = getattr(f, name)
                dependency.__file__ = str(f.root / "wrong_directory" / Path(dependency.__file__).name)
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_not_called()

    def test_completed_verifier_failure_propagates_without_fallback(self):
        f = self.fixture(); f.checker.verify_completed.side_effect = RuntimeError("synthetic cleanup failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic cleanup failure"):
            gate.verify_admission(f.source, f.prompt)

    def test_missing_review_flag_fails(self):
        f = self.fixture(); f.row["root_reviewed_real_evidence"] = False
        f.write(f.receipt, f.row)
        with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
        f.checker.verify_completed.assert_not_called()

    def test_missing_source_and_prompt_fields_fail(self):
        for name in ("source", "edit_prompt"):
            with self.subTest(name=name):
                f = self.fixture(); del f.row[name]
                f.write(f.receipt, f.row)
                with self.assertRaises(RuntimeError): gate.verify_admission(f.source, f.prompt)
                f.checker.verify_completed.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
