#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/cache/zhonghao/h3/cuda_compat13/usr/local/cuda-13.0/compat:/cache/zhonghao/h3/env_cuda_v1/lib:${LD_LIBRARY_PATH:-}
/cache/zhonghao/h3/env_cuda_v1/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
find /cache/zhonghao/h3/dmd8_b_skip_20260917 -maxdepth 4 -type f \
  \( -name 'transformer_minimax_h3.py' -o -name 'modeling_vdn_h3.py' -o -name '__init__.py' \) \
  -print | head -n 100
