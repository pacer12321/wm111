#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/h3
REPO="$ROOT/train_repos/vdn-minimax-h3"
BASE="$ROOT/models/OpenVDN-vdn-minimax-h3/ref2va-base"  # build_ref2va_base.py output, never h3-base (FL2VA)
DMD8="$ROOT/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
SAMPLES=/temp/zhonghao/savie_stream/samples
OUT="$ROOT/train_runs/savie_dmd8_skipoff_2k"

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
  --sample-dir "$SAMPLES" \
  --sample-count 2000 \
  --max-steps 1000 \
  --save-every 100 \
  --lr 1e-6 \
  --lora-rank 64 \
  --lora-alpha 64 \
  --shard-size 2 \
  --no-train-token-skip \
  --no-cpu-offload
