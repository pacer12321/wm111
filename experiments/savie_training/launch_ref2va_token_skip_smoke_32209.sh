#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/h3
REPO="$ROOT/train_repos/vdn-minimax-h3"
BASE="$ROOT/models/OpenVDN-vdn-minimax-h3/ref2va-base"  # build_ref2va_base.py output, never h3-base (FL2VA)
DMD8="$ROOT/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
OUT="$ROOT/train_runs/ref2va_dual_token_skip_smoke_dmd8"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$ROOT/cuda_compat13/usr/local/cuda-13.0/compat:$ROOT/env_cuda_v1/lib:${LD_LIBRARY_PATH:-}"

exec "$ROOT/env_cuda_v1/bin/torchrun" --standalone --nproc-per-node=4 \
  -m src.training.train_ref2va_dual \
  --base "$BASE" \
  --dmd8 "$DMD8" \
  --output "$OUT" \
  --max-steps 1 \
  --shard-size 2 \
  --no-cpu-offload
