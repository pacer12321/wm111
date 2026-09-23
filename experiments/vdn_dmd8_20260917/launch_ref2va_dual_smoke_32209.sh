#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/h3
REPO="$ROOT/train_repos/vdn-minimax-h3"
BASE="$ROOT/models/OpenVDN-vdn-minimax-h3/ref2va-base"  # build_ref2va_base.py output, never h3-base (FL2VA)
DMD8="$ROOT/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
OUT="$ROOT/train_runs/ref2va_dual_smoke_dmd8"

mkdir -p "$OUT"

if [[ ! -s "$BASE/ref2va_base_receipt.json" ]]; then
  echo "Ref2VA base missing: run build_ref2va_base.py first (train_ref2va_dual verifies the receipt)" >&2
  exit 20
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$ROOT/cuda_compat13/usr/local/cuda-13.0/compat:$ROOT/env_cuda_v1/lib:${LD_LIBRARY_PATH:-}"

"$ROOT/env_cuda_v1/bin/torchrun" --standalone --nproc-per-node=4 \
  -m src.training.train_ref2va_dual \
  --base "$BASE" \
  --dmd8 "$DMD8" \
  --output "$OUT" \
  --max-steps 1 \
  --no-cpu-offload
