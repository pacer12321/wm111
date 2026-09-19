"""Read-only admission of one real VLM run; never substitute intended labels."""
import importlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/cache/zhonghao/h3/color_trial_v1')
VLM_CODE = ROOT / 'vlm'
ADMISSION = ROOT / 'vlm_admission.json'


def verify_admission(source, prompt):
    if (Path(__file__).resolve() != ROOT / 'a/vlm_gate.py'
            or ADMISSION.resolve(strict=True) != ADMISSION
            or not ADMISSION.is_file() or ADMISSION.is_symlink()):
        raise RuntimeError('Missing/noncanonical real VLM admission; no manual-label fallback')
    row = json.loads(ADMISSION.read_text(encoding='utf-8'))
    if (set(row) != {'schema_version', 'run_id', 'run_directory', 'status_sha256', 'result_sha256',
                     'source', 'edit_prompt', 'classification', 'root_reviewed_real_evidence'}
            or type(row['schema_version']) is not int or row['schema_version'] != 1
            or row['source'] != source or row['edit_prompt'] != prompt
            or row['classification'] != 'preserve' or row['root_reviewed_real_evidence'] is not True
            or any(not isinstance(row[key], str) or re.fullmatch('[0-9a-f]{64}', row[key]) is None
                   for key in ('status_sha256', 'result_sha256'))):
        raise RuntimeError('Real VLM admission does not match this frozen color source and edit')
    sys.path.insert(0, str(VLM_CODE))
    checker = importlib.import_module('run_vlm')
    contract = importlib.import_module('vlm_contract')
    if (Path(checker.__file__).resolve() != VLM_CODE / 'run_vlm.py'
            or Path(contract.__file__).resolve() != VLM_CODE / 'vlm_contract.py'):
        raise RuntimeError('VLM verifier loaded outside the reviewed private directory')
    run = contract.run_directory(row['run_directory'], row['run_id'])
    for name, key in (('vlm_status.json', 'status_sha256'), ('worker_result.json', 'result_sha256')):
        if contract.record(run / name)['sha256'] != row[key]:
            raise RuntimeError('Pinned actual VLM run evidence changed after review')
    verified = checker.verify_completed(str(run), row['run_id'])
    result = verified['result']
    if (result['source'] != source or result['edit_prompt'] != prompt
            or result['classification']['classification'] != 'preserve'):
        raise RuntimeError('This fixed same-frame color trial requires actual preserve decision; review change/uncertain')
    return {'admission': contract.record(ADMISSION), 'evidence': verified,
            'policy': 'One shared actual source+edit VLM decision for matched ABC; generator definitions unchanged.',
            'limitation': 'Pre-generation classification is not a guarantee of output quality or preserved timing.'}
