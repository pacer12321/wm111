"""CPU toy tests for fresh-execution-only, failure-invalidating report publishing."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import run_runtime_validation as reporter

HOST = {'hostname': reporter.HOSTNAME, 'boot_id': '12345678-abcd-abcd-abcd-123456789abc', 'machine': 'aarch64'}


def checker_pass():
    return {'status': reporter.PASSED, 'scope': 'actual CPU checker scope', 'npu_inference_verified': False,
            'environment': {'PATH': ['/private/bin']}, 'metadata': {'torch': {'version': '2.10.0'}},
            'runtime_prefix_files': {'scanned': ['private.pth']}, 'modules': {'torch': {'file': 'private'}},
            'abi': [{'file': 'private.so', 'elf_machine': 183}], 'warnings': [],
            'device_guard': {'python_npu_initialized': False, 'device_operations_requested': 0,
                             'npu_lazy_init_and_c_init_guarded': True}}


class ReportTests(unittest.TestCase):
    def execute(self, *, result=None, error=None, change_binding=False, second_run=False, missing_input=False):
        with tempfile.TemporaryDirectory(prefix='h3_report_cpu_') as directory, contextlib.ExitStack() as stack:
            root = Path(directory).resolve()
            latest, history = root / 'runtime_validation.json', root / 'runtime_validation_runs'
            old = {'status': reporter.PASSED, 'run_id': 'historical-do-not-reuse'}
            latest.write_text(json.dumps(old))
            binding = {'host': HOST, 'env_script_sha256': 'env-sha',
                       'source_sha256': {'minimax_h3_transformer.py': 'transformer-sha', 'pipeline_minimax_h3.py': 'pipeline-sha'},
                       'validator_sha256': 'checker-sha', 'reporter_sha256': 'reporter-sha'}
            for name, value in (('ROOT', root), ('LATEST', latest), ('HISTORY', history),
                                ('ENV', Path(sys.prefix).resolve()), ('CODE', HERE)):
                stack.enter_context(mock.patch.object(reporter, name, value))
            stack.enter_context(mock.patch.object(reporter, 'host_identity', return_value=HOST))
            stack.enter_context(mock.patch.object(reporter.os, 'O_NOFOLLOW', 0, create=True))
            stack.enter_context(mock.patch.object(reporter.os, 'O_NONBLOCK', 0, create=True))
            stack.enter_context(mock.patch.object(reporter.os, 'getuid', return_value=root.stat().st_uid, create=True))
            stack.enter_context(mock.patch.object(reporter.fcntl, 'flock'))
            stack.enter_context(mock.patch.object(reporter.signal, 'signal'))
            stack.enter_context(mock.patch('sys.stdout', new=io.StringIO()))
            if missing_input:
                identity = stack.enter_context(mock.patch.object(reporter, 'input_binding', side_effect=FileNotFoundError('missing private source')))
            elif change_binding:
                identity = stack.enter_context(mock.patch.object(reporter, 'input_binding', side_effect=[binding, {**binding, 'env_script_sha256': 'changed'}]))
            else:
                identity = stack.enter_context(mock.patch.object(reporter, 'input_binding', return_value=binding))
            captured = []
            def run(command, **kwargs):
                interim = json.loads(latest.read_text())
                self.assertEqual(interim['status'], 'running_cpu_runtime_checks')
                self.assertNotEqual(interim['run_id'], old['run_id'])
                captured.append((command, kwargs))
                if error:
                    raise error
                return result or SimpleNamespace(returncode=0, stdout=json.dumps(checker_pass()), stderr='')
            mock_run = stack.enter_context(mock.patch.object(reporter.subprocess, 'run', side_effect=run))
            code = reporter.run_once()
            published = json.loads(latest.read_text())
            records = [json.loads(path.read_text()) for path in history.glob('*/report.json')]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0], published)
            self.assertEqual(published['host'], HOST)
            if second_run:
                prior_id = published['run_id']
                code = reporter.run_once()
                published = json.loads(latest.read_text())
                self.assertNotEqual(prior_id, published['run_id'])
                self.assertEqual(len(list(history.glob('*/report.json'))), 2)
            return code, published, captured, mock_run.call_count, identity.call_count

    def test_fresh_checker_invoked_after_old_pass_invalidated(self):
        code, report, calls, count, _ = self.execute()
        self.assertEqual(code, 0); self.assertEqual(count, 1)
        self.assertEqual(report['status'], reporter.PASSED)
        self.assertEqual(report['checker_report'], checker_pass())
        for key, value in checker_pass().items():
            self.assertEqual(report[key], value)
        self.assertEqual(report['env_script_sha256'], 'env-sha')
        self.assertEqual(report['source_sha256']['pipeline_minimax_h3.py'], 'pipeline-sha')
        command, kwargs = calls[0]
        self.assertIn('source /cache/zhonghao/h3/env_h3_31731.sh', command[-1])
        self.assertIn('validate_h3_runtime.py', command[-1])
        self.assertEqual(kwargs['env']['TORCH_DEVICE_BACKEND_AUTOLOAD'], '0')
        self.assertEqual(kwargs['env']['VLLM_CONFIGURE_LOGGING'], '1')
        self.assertEqual(kwargs['env']['VLLM_LOGGING_STREAM'], 'ext://sys.stderr')
        self.assertEqual(kwargs['env']['VLLM_LOGGING_COLOR'], '0')
        self.assertNotIn('VLLM_LOGGING_CONFIG_PATH', kwargs['env'])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(len(kwargs['pass_fds']), 1)
        self.assertEqual(kwargs['timeout'], 300)

    def test_repeated_run_creates_unique_history(self):
        code, _, _, count, _ = self.execute(second_run=True)
        self.assertEqual(code, 0); self.assertEqual(count, 2)

    def test_failed_exit_does_not_reuse_previous_pass(self):
        original = {'status': 'failed', 'error': 'ABI dependency missing'}
        code, report, _, _, _ = self.execute(result=SimpleNamespace(returncode=1, stdout=json.dumps(original), stderr='detail'))
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['checker_report'], original)
        self.assertIn('finished_at', report)

    def test_claimed_pass_with_bad_exit_is_rejected(self):
        code, report, _, _, _ = self.execute(result=SimpleNamespace(returncode=1, stdout=json.dumps(checker_pass()), stderr='bad-exit'))
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')

    def test_missing_sections_or_device_guard_cannot_publish_pass(self):
        for mutation in ('abi', 'modules', 'device_guard'):
            original = checker_pass(); original[mutation] = {}
            with self.subTest(mutation=mutation):
                code, report, _, _, _ = self.execute(result=SimpleNamespace(returncode=0, stdout=json.dumps(original), stderr=''))
                self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')

    def test_timeout_and_launch_failure_publish_failed_latest(self):
        for error in (subprocess.TimeoutExpired('checker', 300), FileNotFoundError('bash'), InterruptedError('cancelled')):
            code, report, _, _, _ = self.execute(error=error)
            self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')
            self.assertIn(type(error).__name__, report['error'])

    def test_input_binding_failure_invalidates_old_pass_without_checker(self):
        code, report, _, count, _ = self.execute(missing_input=True)
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')
        self.assertEqual(count, 0)

    def test_host_code_or_env_changed_during_check_rejects_pass(self):
        code, report, _, _, _ = self.execute(change_binding=True)
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['checker_report']['status'], reporter.PASSED)

    def test_non_json_output_cannot_be_misread_as_pass(self):
        code, report, _, _, _ = self.execute(result=SimpleNamespace(returncode=0, stdout='not a checker report', stderr=''))
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')

    def test_prefixed_stdout_plugin_logs_are_not_loose_parsed_as_pass(self):
        mixed = 'INFO Available plugins for group vllm.platform_plugins:\n' + json.dumps(checker_pass())
        code, report, _, _, _ = self.execute(result=SimpleNamespace(returncode=0, stdout=mixed, stderr=''))
        self.assertEqual(code, 1); self.assertEqual(report['status'], 'failed')
        self.assertNotIn('checker_report', report)

    def test_plugin_logs_on_stderr_leave_strict_json_stdout_valid(self):
        logs = 'INFO Available plugins for group vllm.platform_plugins:\nINFO Ascend plugin registered\n'
        with mock.patch.dict(os.environ, {'VLLM_LOGGING_STREAM': 'ext://sys.stdout',
                                         'VLLM_CONFIGURE_LOGGING': '0', 'VLLM_LOGGING_CONFIG_PATH': '/foreign/logging.json'}):
            code, report, calls, _, _ = self.execute(result=SimpleNamespace(returncode=0, stdout=json.dumps(checker_pass()), stderr=logs))
        self.assertEqual(code, 0); self.assertEqual(report['status'], reporter.PASSED)
        self.assertEqual(report['checker_logging']['stream'], 'ext://sys.stderr')
        environment = calls[0][1]['env']
        self.assertEqual(environment['VLLM_LOGGING_STREAM'], 'ext://sys.stderr')
        self.assertEqual(environment['VLLM_CONFIGURE_LOGGING'], '1')
        self.assertNotIn('VLLM_LOGGING_CONFIG_PATH', environment)

    def test_supported_logging_stream_configuration_isolates_real_python_streams(self):
        # CPU integration of the exact logging.config StreamHandler mechanism
        # read from real vLLM logger.py, without importing torch/vLLM/NPU.
        code = '''import json, logging, logging.config, os
logging.config.dictConfig({"version": 1, "handlers": {"vllm": {"class": "logging.StreamHandler", "stream": os.environ["VLLM_LOGGING_STREAM"]}}, "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": False}}})
logging.getLogger("vllm").info("Available plugins for group vllm.platform_plugins:")
print(json.dumps({"machine_report": True}))
'''
        environment = os.environ.copy()
        environment['VLLM_LOGGING_STREAM'] = 'ext://sys.stderr'
        process = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                                 env=environment, timeout=10, check=True)
        self.assertEqual(json.loads(process.stdout), {'machine_report': True})
        self.assertIn('Available plugins', process.stderr)
        self.assertNotIn('Available plugins', process.stdout)

    def test_binding_reads_real_host_arch_and_boot(self):
        with mock.patch.object(reporter.socket, 'gethostname', return_value=HOST['hostname']), \
             mock.patch.object(reporter.os, 'uname', return_value=SimpleNamespace(machine='aarch64'), create=True), \
             mock.patch.object(Path, 'read_text', return_value=HOST['boot_id']):
            self.assertEqual(reporter.host_identity(), HOST)
        with mock.patch.object(reporter.socket, 'gethostname', return_value='wrong-host'), \
             mock.patch.object(reporter.os, 'uname', return_value=SimpleNamespace(machine='aarch64'), create=True), \
             mock.patch.object(Path, 'read_text', return_value=HOST['boot_id']):
            with self.assertRaises(RuntimeError):
                reporter.host_identity()

    def test_no_arguments_or_force_pass_override(self):
        with mock.patch.object(sys, 'argv', ['reporter', '--passed']), mock.patch.object(reporter, 'run_once') as run:
            with self.assertRaises(SystemExit):
                reporter.main()
            run.assert_not_called()

    def test_canonical_rejects_external_and_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch.object(reporter, 'ROOT', root):
                for path in (root.parent / 'outside.json', root / 'a/../outside.json', Path('relative.json')):
                    with self.assertRaises(RuntimeError):
                        reporter.canonical(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
