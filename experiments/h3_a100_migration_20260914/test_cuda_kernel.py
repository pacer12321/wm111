"""Small real CUDA check, original segmented SDPA vs new packed FA2 math."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT = Path('/cache/zhonghao/h3/a100_v1')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert torch.cuda.device_count() == 2
    records = []
    for gpu in range(2):
        torch.cuda.set_device(gpu)
        assert torch.cuda.get_device_capability(gpu) == (8, 0)
        for case in ('B', 'C', 'D'):
            path = ROOT / f'candidates/{case}/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py'
            module = load(f'cuda_test_{case}_{gpu}', path)
            for q_lengths, k_lengths in [([7], [13]), ([7, 11, 19], [17, 8, 31])]:
                torch.manual_seed(4101)
                q = torch.randn(sum(q_lengths), 28, 128, dtype=torch.bfloat16, device=f'cuda:{gpu}')
                k = torch.randn(sum(k_lengths), 28, 128, dtype=q.dtype, device=q.device)
                v = torch.randn_like(k)
                cu_q, cu_k, expected = [], [], []
                qi = ki = 0
                with sdpa_kernel(SDPBackend.MATH):
                    for qn, kn in zip(q_lengths, k_lengths):
                        expected.append(F.scaled_dot_product_attention(
                            q[qi:qi+qn].float().transpose(0, 1).unsqueeze(0),
                            k[ki:ki+kn].float().transpose(0, 1).unsqueeze(0),
                            v[ki:ki+kn].float().transpose(0, 1).unsqueeze(0),
                            scale=128**-0.5).squeeze(0).transpose(0, 1))
                        qi += qn
                        ki += kn
                        cu_q.append(qi)
                        cu_k.append(ki)
                actual = module._fusion_attention_tnd(q, k, v, cu_q, cu_k, 128**-0.5)
                reference = torch.cat(expected)
                torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
                records.append({'gpu': gpu, 'case': case, 'segments': len(q_lengths),
                                'max_abs_error': (actual.float() - reference).abs().max().item()})
    (ROOT / 'cuda_kernel_test.json').write_text(json.dumps({'passed': True, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'records': records}, indent=2))
    print('CUDA varlen regression passed on both A100s')


if __name__ == '__main__':
    main()
