"""Validate identical working-B package inventory; acknowledge only its exact known gaps."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import setup_runtime_30674 as setup

EXPECTED_INVENTORY = '590ce80e7c9c4abb038a0ef2c20e16a345307b9a5da19bea7d45bd0af36f0a78'
EXPECTED_GAPS = ['vllm-omni 0.26.0 requires fa3-fwd, which is not installed.',
                 'vllm-omni 0.26.0 requires onnxruntime, which is not installed.']


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    old = json.loads((setup.CONTROL / 'status.json').read_text())
    assert old['phase'] == 'failed' and "'pip', 'check'" in old['error']
    assert not Path(f'/proc/{old["pid"]}').exists()
    assert not (setup.ROOT / 'a100_v1/results/D').exists()
    assert not (setup.CONTROL / 'd_dispatch_succeeded.json').exists()
    if sys.argv[1:] == ['--launch']:
        with (setup.CONTROL / 'baseline_validation_resume.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'validation_pid': proc.pid}))
        return
    try:
        environment = dict(os.environ)
        for key in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'LD_AUDIT'):
            environment.pop(key, None)
        environment.update(PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
            CUDA_VISIBLE_DEVICES='GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670,GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3',
            LD_LIBRARY_PATH=str(setup.ROOT / 'cuda_compat13/usr/local/cuda-13.0/compat') + ':' +
                            str(setup.ENV / 'lib') + ':' + str(setup.RUNTIME / 'lib'))
        python = str(setup.ENV / 'bin/python')
        setup.status('comparing_working_B_baseline')
        inventory = subprocess.check_output([python, '-m', 'pip', 'list', '--format=json',
                                             '--disable-pip-version-check'], env=environment)
        assert hashlib.sha256(inventory).hexdigest() == EXPECTED_INVENTORY, 'Package inventory drift'
        checked = subprocess.run([python, '-m', 'pip', 'check'], env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        assert checked.returncode == 1 and sorted(checked.stdout.strip().splitlines()) == EXPECTED_GAPS
        (setup.CONTROL / 'baseline_dependency_evidence.json').write_text(json.dumps({
            'source_host_port': 32209, 'source_B_completed': True,
            'identical_source_destination_inventory_sha256': EXPECTED_INVENTORY,
            'exact_existing_baseline_pip_check_gaps': EXPECTED_GAPS,
            'packages_installed_removed_or_upgraded': False}, indent=2))
        probe = ('import sys,torch,vllm,transformers,diffusers,av,json; '
                 'assert sys.base_prefix=="/cache/zhonghao/h3/python312_runtime"; '
                 'assert torch.__version__=="2.11.0+cu130"; '
                 'assert vllm.__version__=="0.26.0"; '
                 'assert transformers.__version__=="5.17.0"; '
                 'assert diffusers.__version__=="0.38.0"; '
                 'print(json.dumps({"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda}))')
        subprocess.run([python, '-c', probe], env=environment, check=True, timeout=120)
        setup.status('runtime_verified', exact_working_B_inventory=True,
                     baseline_pip_check_gaps_acknowledged=EXPECTED_GAPS)
        subprocess.run([python, str(setup.ROOT / 'run_d_30674.py'), '--launch'],
                       env=environment, check=True, timeout=120)
        (setup.CONTROL / 'd_dispatch_succeeded.json').write_text(json.dumps({'launched': True}))
    except BaseException as exc:
        setup.status('failed', error=repr(exc), automatic_model_retry=False)
        raise


if __name__ == '__main__':
    main()
