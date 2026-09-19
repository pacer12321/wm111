"""Run the private bootstrap against CPU-only toy vendor scripts, never CANN/NPU.

Only the fixed install literal is substituted into an ephemeral test directory.
Production env_h3_31731.sh has no path override or test bypass. On Windows this
uses Git Bash; no remote writes, vendor imports or hardware operations occur.
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BASH = shutil.which('bash') if os.name != 'nt' else r'E:\Git\bin\bash.exe'
KEYS = ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CMAKE_PREFIX_PATH', 'LIBRARY_PATH', 'CPATH',
        'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'PKG_CONFIG_PATH', 'LD_PRELOAD', 'LD_AUDIT',
        'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME',
        'ASCEND_AICPU_PATH', 'ASCEND_OPP_PATH', 'ASCEND_DRIVER_PATH', 'ASCEND_AUTOML_PATH',
        'ASCEND_CUSTOM_OPP_PATH', 'ATB_HOME_PATH', 'ATB_CACHE_DIR', 'CANN_HOME', 'NNAE_HOME_PATH',
        'DDK_PATH', 'TOOLCHAIN_HOME', 'TBE_IMPL_PATH', 'ASCEND_RT_VISIBLE_DEVICES', 'HCCL_CONNECT_TIMEOUT',
        'H3_ROOT', 'H3_OUTPUT', 'ATB_SHARE_MEMORY_NAME_SUFFIX', 'VLLM_OMNI_VIDEO_SYNC_TIMEOUT',
        'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR')


CANN_TOY = '''#!/usr/bin/env bash
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export TOOLCHAIN_HOME="$ASCEND_HOME_PATH/toolkit"
export CMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/lib64/cmake:${CMAKE_PREFIX_PATH:-}"
'''
ATB_TOY = '''#!/usr/bin/env bash
export ATB_HOME_PATH="$H3_ENV/Ascend/nnal/nnal/atb/latest/atb/cxx_abi_1"
export PATH="$ATB_HOME_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ATB_HOME_PATH/lib:$LD_LIBRARY_PATH"
export ATB_SHARE_MEMORY_NAME_SUFFIX=""
'''


class EnvironmentTests(unittest.TestCase):
    def run_bootstrap(self, values=None, *, repeats=1, cann_extra='', atb_extra='', omit_atb=False, root_escape=False):
        with tempfile.TemporaryDirectory(prefix='h3_env_cpu_') as directory:
            root = Path(directory).resolve()
            if os.name == 'nt':
                mapped = subprocess.run([BASH, '--noprofile', '--norc', '-c', '/usr/bin/cygpath -u "$1"', 'path', str(root)],
                                        capture_output=True, text=True, check=True)
                bash_root = mapped.stdout.strip()
            else:
                bash_root = str(root)
            cann = root / 'env/Ascend/cann-9.0.1/set_env.sh'
            atb = root / 'env/Ascend/nnal/nnal/atb/set_env.sh'
            cann.parent.mkdir(parents=True)
            atb.parent.mkdir(parents=True)
            cann.write_text(CANN_TOY + cann_extra, encoding='utf-8')
            if not omit_atb:
                atb.write_text(ATB_TOY + atb_extra, encoding='utf-8')
            original = (HERE / 'env_h3_31731.sh').read_text(encoding='utf-8')
            literal = 'export H3_INSTALL=/cache/zhonghao/h3'
            self.assertEqual(original.count(literal), 1)
            bootstrap = original.replace(literal, 'export H3_INSTALL=' + shlex.quote(bash_root))
            assignments = '\n'.join('export ' + name + '=' + shlex.quote(value.replace('{root}', bash_root))
                                    for name, value in (values or {}).items())
            prefix = 'set -euo pipefail\n' + assignments + '\n'
            if root_escape:
                prefix += 'realpath() { printf "/outside-install\\n"; }\n'
            # End with a fixed selected-key inventory, never dump a real env.
            report = '\nfor h3_test_key in ' + ' '.join(KEYS) + '; do printf "%s\\0%s\\0" "$h3_test_key" "${!h3_test_key-}"; done\n'
            script = prefix + ('\n' + bootstrap) * repeats + report
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob('*'))
            result = subprocess.run([BASH, '--noprofile', '--norc', '-s'], input=script.encode(), capture_output=True,
                                    env={key: value for key, value in os.environ.items() if key not in ('LD_PRELOAD', 'LD_AUDIT')}, timeout=20)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob('*'))
            self.assertEqual(before, after, 'The environment bootstrap must not create/edit installation or output files')
            if result.returncode:
                return result, {}, bash_root
            fields = result.stdout.decode().split('\0')
            self.assertEqual(fields[-1], '')
            data = dict(zip(fields[0:-1:2], fields[1:-1:2]))
            self.assertEqual(set(data), set(KEYS))
            return result, data, bash_root

    def polluted(self):
        return {
            'PATH': '/usr/local/Ascend/cann-8.5.2/bin:/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/driver/tools:/usr/local/ffmpeg/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/bin:/bin',
            'LD_LIBRARY_PATH': '/usr/local/Ascend/cann-8.5.2/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/host-lib:/usr/host-lib64:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/toolbox/latest/Ascend-DMI/lib64:/usr/local/seccomponent/lib/',
            'PYTHONPATH': '/usr/local/Ascend/cann-8.5.2/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:/usr/local/seccomponent/lib:/home/ma-user/infer/model/1',
            'CMAKE_PREFIX_PATH': '/usr/local/Ascend/cann-8.5.2/lib64/cmake:/usr/lib/cmake',
            'ASCEND_HOME_PATH': '/usr/local/Ascend/cann-8.5.2',
            'ASCEND_TOOLKIT_HOME': '/usr/local/Ascend/cann-8.5.2',
            'ASCEND_AICPU_PATH': '/usr/local/Ascend/cann-8.5.2',
            'ASCEND_OPP_PATH': '/usr/local/Ascend/cann-8.5.2/opp',
            'ASCEND_AUTOML_PATH': '/usr/local/Ascend/ascend-toolkit/latest/tools',
            'ATB_HOME_PATH': '/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1',
        }

    def test_reproduced_inherited_old_toolkit_paths_are_removed(self):
        result, data, root = self.run_bootstrap(self.polluted())
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        all_values = '\n'.join(data.values())
        for forbidden in ('cann-8.5.2', '/usr/local/Ascend/ascend-toolkit', '/usr/local/Ascend/nnal',
                          '/usr/host-lib', 'seccomponent', '/home/ma-user', '/cache/yunfeng', 'Ascend-DMI'):
            self.assertNotIn(forbidden, all_values)
        self.assertEqual(data['ASCEND_HOME_PATH'], root + '/env/Ascend/cann-9.0.1')
        self.assertEqual(data['ASCEND_OPP_PATH'], data['ASCEND_HOME_PATH'] + '/opp')
        self.assertEqual(data['ASCEND_AICPU_PATH'], data['ASCEND_HOME_PATH'])

    def test_required_driver_system_and_private_torch_libraries_remain(self):
        result, data, root = self.run_bootstrap({'LD_LIBRARY_PATH': '/lib64:/usr/lib/aarch64-linux-gnu/hdf5/serial:{root}/env/lib/python3.12/site-packages/torch/lib'})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        libraries = data['LD_LIBRARY_PATH'].split(':')
        for expected in ('/lib64', '/usr/lib/aarch64-linux-gnu/hdf5/serial', '/usr/local/Ascend/driver/lib64',
                         '/usr/local/Ascend/driver/lib64/common', '/usr/local/Ascend/driver/lib64/driver',
                         root + '/env/lib', root + '/env/lib/python3.12/site-packages/torch/lib',
                         root + '/env/lib/python3.12/site-packages/torch_npu/lib'):
            self.assertIn(expected, libraries)
        self.assertEqual(data['ASCEND_DRIVER_PATH'], '/usr/local/Ascend/driver')
        self.assertEqual(data['PATH'].split(':')[:2], [root + '/bin', root + '/env/bin'])

    def test_group_cards_and_run_controls_preserved(self):
        result, data, root = self.run_bootstrap({'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
            'HCCL_CONNECT_TIMEOUT': '333', 'H3_ROOT': '{root}/test_run', 'H3_OUTPUT': '{root}/test_run/output',
            'H3_GROUP_ID': 'group8_unique_run', 'VLLM_OMNI_VIDEO_SYNC_TIMEOUT': '1800'})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(data['ASCEND_RT_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(data['HCCL_CONNECT_TIMEOUT'], '333')
        self.assertEqual(data['H3_ROOT'], root + '/test_run')
        self.assertEqual(data['H3_OUTPUT'], root + '/test_run/output')
        self.assertEqual(data['ATB_SHARE_MEMORY_NAME_SUFFIX'], 'group8_unique_run')
        self.assertEqual(data['XDG_CACHE_HOME'], root + '/test_run/cache')

    def test_does_not_allocate_default_devices(self):
        result, data, _ = self.run_bootstrap({'ASCEND_RT_VISIBLE_DEVICES': ''})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(data['ASCEND_RT_VISIBLE_DEVICES'], '')

    def test_repeated_source_is_deduplicated_and_still_private(self):
        result, data, root = self.run_bootstrap(self.polluted(), repeats=2)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        for key in ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CMAKE_PREFIX_PATH'):
            entries = data[key].split(':')
            self.assertNotIn('', entries)
            self.assertEqual(len(entries), len(set(entries)))
        self.assertEqual(data['PATH'].split(':')[:2], [root + '/bin', root + '/env/bin'])

    def test_empty_relative_traversal_and_foreign_python_entries_removed(self):
        result, data, root = self.run_bootstrap({'PYTHONPATH': ':.:relative:/usr/lib/../local/Ascend/cann-8.5.2/python:{root}/candidates/b_v1/vllm-omni:{root}/src/vllm:',
                                               'LD_LIBRARY_PATH': '::.:relative:/lib64:/lib64:'})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertNotIn('/candidates/', data['PYTHONPATH'])
        for key in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
            self.assertTrue(all(value.startswith('/') and '/..' not in value and value != '.' for value in data[key].split(':')))
        self.assertIn(root + '/src/vllm', data['PYTHONPATH'].split(':'))

    def test_preload_interpreter_home_and_old_accelerator_bindings_removed(self):
        result, data, _ = self.run_bootstrap({'LD_PRELOAD': '/foreign/lib.so', 'LD_AUDIT': '/foreign/audit.so',
            'PYTHONHOME': '/foreign/python', 'PYTHONSTARTUP': '/foreign/start.py', 'PYTHONUSERBASE': '/foreign/user',
            'ASCEND_AUTOML_PATH': '/old/tools', 'ATB_CACHE_DIR': '/old/cache', 'CANN_HOME': '/old/cann',
            'NNAE_HOME_PATH': '/old/nnae', 'DDK_PATH': '/old/ddk', 'TBE_IMPL_PATH': '/old/tbe'})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE',
                    'ASCEND_AUTOML_PATH', 'ATB_CACHE_DIR', 'CANN_HOME', 'NNAE_HOME_PATH', 'DDK_PATH', 'TBE_IMPL_PATH'):
            self.assertEqual(data[key], '')

    def test_build_search_paths_are_filtered(self):
        values = {key: '/old/cann:/usr/include:/usr/lib:{root}/env/include:' for key in
                  ('LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'PKG_CONFIG_PATH')}
        result, data, root = self.run_bootstrap(values)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        for key in values:
            self.assertEqual(data[key], '/usr/include:/usr/lib:' + root + '/env/include')

    def test_vendor_reintroducing_foreign_runtime_path_fails_closed(self):
        result, data, _ = self.run_bootstrap(cann_extra='export LD_LIBRARY_PATH="/usr/local/Ascend/cann-8.5.2/lib64:$LD_LIBRARY_PATH"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(data)
        self.assertIn('Foreign/noncanonical LD_LIBRARY_PATH', result.stderr.decode())

    def test_vendor_wrong_toolkit_home_is_rejected(self):
        result, _, _ = self.run_bootstrap(cann_extra='export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2\n')
        self.assertNotEqual(result.returncode, 0)

    def test_foreign_atb_home_is_rejected(self):
        result, _, _ = self.run_bootstrap(atb_extra='export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest\n')
        self.assertNotEqual(result.returncode, 0)

    def test_missing_private_vendor_script_and_noncanonical_root_rejected(self):
        for kwargs in ({'omit_atb': True}, {'root_escape': True}):
            result, _, _ = self.run_bootstrap(**kwargs)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
