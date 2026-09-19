#!/usr/bin/env python3
"""Genuine local Qwen3VL generation; no device import before explicit opt-in."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import time

import vlm_contract as contract


def inherited_run():
    host = contract.profiles.require_host()
    if Path(__file__).resolve().parent != contract.CODE:
        raise RuntimeError('VLM worker is outside the fixed code directory')
    run_id = os.environ.get('H3_MINIMAL_RUN_ID', '')
    directory = contract.run_directory(os.environ.get('H3_ROOT', ''), run_id)
    supervisor = int(os.environ.get('H3_VLM_SUPERVISOR_PID', '0'))
    if (os.getppid() != supervisor or supervisor <= 1 or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '0,1'
            or contract.json_read(directory/'host_identity.json') != host):
        raise RuntimeError('Missing live direct supervisor, exact host or physical cards 0,1')
    expected = [str(contract.OUTPUT/'run.lock')] + [str(contract.profiles.profile('01234567').lease_root/f'device{i}.lock') for i in range(8)]
    descriptors = os.environ.get('H3_VLM_LEASE_FDS', '').split(',')
    if len(descriptors) != 9 or len(set(descriptors)) != 9: raise RuntimeError('All nine inherited personal lease handles are required')
    actual = []
    for value in descriptors:
        fd = int(value); st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != os.getuid():
            raise RuntimeError('Nonprivate inherited lease')
        actual.append(os.readlink(f'/proc/self/fd/{fd}'))
    if actual != expected: raise RuntimeError('Inherited locks do not cover the fixed group and all eight cards')
    identity = contract.base.proc_identity(os.getpid())
    if identity is None or identity['pgrp'] != os.getpid() or identity['session'] != os.getpid():
        raise RuntimeError('VLM worker must own its dedicated process group/session')
    return host, run_id, directory, identity


def tensor_record(value):
    if str(value.device) != 'cpu' or value.numel() <= 0:
        raise RuntimeError('Record nonempty CPU preprocessing tensors before device transfer')
    raw = value.detach().contiguous().view(__import__('torch').uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(), sha256=hashlib.sha256(raw).hexdigest())


def preprocess(video, preparation):
    import torch
    from transformers import Qwen3VLProcessor
    from transformers.video_utils import VideoMetadata
    processor = Qwen3VLProcessor.from_pretrained(str(contract.MODEL), local_files_only=True, trust_remote_code=False)
    conversation = contract.messages()
    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    metadata = VideoMetadata(total_num_frames=124, fps=24.0, width=1280, height=720, duration=124/24,
                             video_backend='pyav', frames_indices=list(contract.FRAME_INDICES))
    inputs = processor(text=[rendered], videos=[video], return_tensors='pt', return_metadata=True,
                       videos_kwargs=dict(video_metadata=[metadata], do_sample_frames=False,
                                          size=contract.VIDEO_SIZE.copy(), device='cpu'))
    returned = inputs.pop('video_metadata', None)
    if (not isinstance(returned, list) or len(returned) != 1 or returned[0].fps != 24
            or list(returned[0].frames_indices) != list(contract.FRAME_INDICES)):
        raise RuntimeError('Processor resampled or lost original video timestamps')
    allowed = {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values_videos', 'video_grid_thw'}
    if set(inputs) - allowed or not {'input_ids', 'pixel_values_videos', 'video_grid_thw'}.issubset(inputs):
        raise RuntimeError('Processor output is not the expected actual video-conditioned input')
    records = {key: tensor_record(value) for key, value in inputs.items()}
    grid = inputs['video_grid_thw'].tolist()
    tokens = int((inputs['input_ids'] == processor.video_token_id).sum().item())
    if (len(grid) != 1 or len(grid[0]) != 3 or grid[0][0] != 8 or tokens <= 0
            or tokens != math_product(grid[0]) // 4 or inputs['input_ids'].shape[0] != 1
            or inputs['input_ids'].shape[1] > 4096 or not torch.isfinite(inputs['pixel_values_videos']).all().item()):
        raise RuntimeError('Video tensor/token/grid count does not prove the fixed 16-frame input')
    record = dict(preparation, messages=conversation, rendered_chat=rendered,
                  rendered_chat_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                  processor_class=type(processor).__name__, video_processor_class=type(processor.video_processor).__name__,
                  processor_options=dict(do_sample_frames=False, video_size=contract.VIDEO_SIZE, device='cpu'),
                  original_video_metadata=dict(returned[0]), merged_pair_timestamps_seconds=processor._calculate_timestamps(list(contract.FRAME_INDICES), 24, 2),
                  tensors=records, input_ids=inputs['input_ids'].tolist(), input_token_count=int(inputs['input_ids'].shape[1]),
                  video_token_count=tokens, video_grid_thw=grid, video_tensor_present=True,
                  scope='16 sampled source frames with original PTS; not all 124 frames shown to the VLM')
    return processor, inputs, record


def math_product(values):
    result = 1
    for value in values: result *= value
    return result


def loading_info_json(value):
    """Preserve HF loading-report structure while serializing its set fields.

    Original nonempty missing/unexpected/mismatched/error fields are rejected
    before this conversion. Never stringify or erase an unsupported object.
    """
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('Loading report keys must be strings')
        return {key: loading_info_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [loading_info_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((loading_info_json(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('Unsupported loading-report type: '+type(value).__name__)


def infer(host, run_id, directory, identity):
    started = time.monotonic()
    frozen = contract.json_read(directory/'frozen_evidence.json')
    if (frozen['host'] != host or frozen['run_id'] != run_id or frozen['source'] != contract.source_record()
            or frozen['sources'] != contract.source_manifest() or frozen['model'] != contract.model_identity()):
        raise RuntimeError('VLM preflight evidence changed before worker execution')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'transformers', 'accelerate', 'av')}
    if versions['transformers'] != '5.14.1' or versions['accelerate'] != '1.12.0':
        raise RuntimeError('Unreviewed transformers/accelerate runtime API version')
    before = time.monotonic(); video, preparation = contract.extract_video()
    decode_seconds = time.monotonic() - before
    before = time.monotonic(); processor, inputs, prep = preprocess(video, preparation)
    preprocess_seconds = time.monotonic() - before
    prep.update(host=host, run_id=run_id, worker_proc_identity=identity,
                decoded_seconds=decode_seconds, preprocessing_seconds=preprocess_seconds)
    contract.base.atomic_json(directory/'preprocessing.json', prep)
    import torch
    import torch_npu  # Explicit opt-in path only; registers npu device/ops.
    from transformers import Qwen3VLForConditionalGeneration
    if not torch.npu.is_available() or torch.npu.device_count() != 2:
        raise RuntimeError('Exactly the authorized two visible NPUs are required')
    torch.npu.set_device(0)
    before = time.monotonic()
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(str(contract.MODEL),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map=contract.DEVICE_MAP.copy(),
        attn_implementation='eager', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('VLM checkpoint did not load completely: ' + repr(loading))
    actual_map = {key: str(value) for key, value in model.hf_device_map.items()}
    if actual_map != contract.DEVICE_MAP: raise RuntimeError('Actual VLM device map differs from the reviewed two-card map')
    parameter_bytes = 0
    for name, parameter in model.named_parameters():
        matching = [key for key in contract.DEVICE_MAP if name == key or name.startswith(key+'.')]
        expected = contract.DEVICE_MAP[max(matching, key=len)] if matching else None
        if expected is None or str(parameter.device) != expected or parameter.dtype != torch.bfloat16:
            raise RuntimeError('Unexpected VLM parameter placement/dtype: '+name)
        parameter_bytes += parameter.numel()*parameter.element_size()
    if parameter_bytes != 66714780128: raise RuntimeError('Materialized VLM parameter coverage differs from the audited complete model')
    model.eval()
    for device in (0, 1): torch.npu.synchronize(device)
    load_seconds = time.monotonic() - before
    load_record = dict(host=host, run_id=run_id, worker_proc_identity=identity, model_class=type(model).__name__,
                       device_map=actual_map, parameter_bytes=parameter_bytes, loading_info=loading_info_json(loading),
                       dtype='bfloat16', attention='eager', versions=versions, load_seconds=load_seconds)
    contract.base.atomic_json(directory/'model_load.json', load_record)
    # Accelerate routes tensors at each mapped module; generation starts with
    # visual/embed inputs on npu0, while final norm/lm_head reside on npu1.
    inputs = {key: value.to('npu:0') for key, value in inputs.items()}
    for device in (0, 1): torch.npu.synchronize(device)
    before = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512,
                                use_cache=True, return_dict_in_generate=True, output_scores=False)
    for device in (0, 1): torch.npu.synchronize(device)
    inference_seconds = time.monotonic() - before
    prompt_len = prep['input_token_count']
    ids = output.sequences[0, prompt_len:].detach().cpu().tolist()
    raw = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw_record = dict(host=host, run_id=run_id, source=preparation['source'], edit_prompt=contract.PROMPT,
        generated_ids=ids, raw_text=raw, generation=dict(do_sample=False, num_beams=1, max_new_tokens=512, use_cache=True),
        input_token_count=prompt_len, generated_token_count=len(ids), inference_seconds=inference_seconds)
    contract.base.atomic_json(directory/'model_output.json', raw_record)  # Retain raw reply even when parsing fails.
    if not 1 <= len(ids) <= 512 or len(ids) == 512:
        raise RuntimeError('Empty or token-ceiling-truncated VLM output; no preserve fallback')
    classification = contract.parse_classification(raw)
    if (contract.source_record() != frozen['source'] or contract.source_manifest() != frozen['sources']
            or contract.model_identity() != frozen['model'] or contract.profiles.require_host(host) != host):
        raise RuntimeError('VLM source/model/code/host changed during inference')
    result = dict(status='passed', case=contract.CASE, host=host, run_id=run_id, worker_proc_identity=identity,
        leased_physical_cards=list(range(8)), used_physical_cards=[0, 1],
        sample_id=contract.SAMPLE, source=preparation['source'], edit_prompt=contract.PROMPT, classification=classification,
        video_tensor_present=True, inference_executed=True, human_router_labels_sent_to_model=False,
        records={name: contract.record(directory/name) for name in ('preprocessing.json', 'model_load.json', 'model_output.json')},
        timing_seconds=dict(decode=decode_seconds, preprocessing=preprocess_seconds, model_loading=load_seconds,
                            inference=inference_seconds, worker_wall=time.monotonic()-started),
        scope='Real 16-frame VLM inference, not ground truth, not H3 edit-quality validation. uncertain remains uncertain.')
    contract.base.atomic_json(directory/'worker_result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_only:
        _, record = contract.extract_video()
        print(json.dumps(dict(status='cpu_preparation_only', **record), ensure_ascii=False))
        return 0
    host, run_id, directory, identity = inherited_run()
    try:
        infer(host, run_id, directory, identity)
    except BaseException as exc:
        contract.base.atomic_json(directory/'worker_failure.json', dict(status='failed', run_id=run_id, host=host,
                                  worker_proc_identity=identity, error=f'{type(exc).__name__}: {exc}',
                                  completed_at=contract.base.utc_now(), inference_success_not_assumed=True))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
