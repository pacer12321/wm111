"""Print D policy only after read-only verification of an explicit finished tiny run.

No files, locks, process launches, NPU initialization, or submission claims.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import d_contract as c
from d_execution_verifier import EXPECTED_EXECUTION_CONTRACT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--pins-sha256', required=True)
    args = parser.parse_args()
    code = c.EXPERIMENT / 'validation_d'
    pins_path = code / 'pins.json'
    if digest(pins_path) != args.pins_sha256:
        raise RuntimeError('Explicit validation pins differ')
    pins = json.loads(pins_path.read_text())
    status = json.loads((args.run_directory / 'd_validation_status.json').read_text())
    if status.get('phase') != 'completed' or status.get('validation_passed') is not True:
        raise RuntimeError('Cannot prepare an inference policy before successful D tiny cleanup')
    verifier = code / 'd_validation_verifier.py'
    if digest(verifier) != pins['validation']['d_validation_verifier.py']:
        raise RuntimeError('Tiny verifier hash mismatch')
    policy = dict(
        schema_version=1, case=c.CASE, mode=c.MODE, review_status='approved_for_inference',
        policy_revision='D_v1_seed4101_inference_only', semantic_decisions_complete=True,
        training_performed=False, sample_id=c.SAMPLE, group=c.GROUP,
        candidate_vendor=c.VENDOR.as_posix(),
        candidate_files={'vllm_omni/diffusion/models/minimax_h3/' + name: sha
                         for name, sha in pins['candidate'].items()},
        execution_verifier_sha256=digest(c.VERIFIER), tiny_verifier_sha256=digest(verifier),
        validation_pins_path=pins_path.as_posix(), validation_pins_sha256=args.pins_sha256,
        tiny_proof=dict(case='D_tiny_SP8', mode=c.MODE, run_id=status['run_id'],
                        run_directory=str(args.run_directory),
                        status_sha256=digest(args.run_directory / 'd_validation_status.json'),
                        result_sha256=digest(args.run_directory / 'd_tiny.json'),
                        source_manifest_sha256=digest(args.run_directory / 'source_manifest.json')),
        attention=dict(endpoint_policy='vdn_anchors', auxiliary_policy='preserve_existing',
                       source_linear_text='text', target_linear_text='text', chunk_frames=5,
                       chunk_radius=1, share_parameters=True, independent_stream_states=True),
        execution_contract=EXPECTED_EXECUTION_CONTRACT)
    c.validate_policy(policy)
    spec = importlib.util.spec_from_file_location('_d_prepare_verified', verifier)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.verify_completed(run_directory=str(args.run_directory), run_id=status['run_id'],
                            host=status['host'], policy=policy)
    print(json.dumps(policy, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
