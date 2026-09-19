#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/h3
REPO="$ROOT/train_repos/vdn-minimax-h3"
BASE="$ROOT/models/OpenVDN-vdn-minimax-h3/h3-base"
DMD8="$ROOT/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
OUT="$ROOT/train_runs/ref2va_dual_smoke_dmd8"
DOWNLOAD_PID=1947334

mkdir -p "$OUT"

while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
  sleep 15
done

INDEX="$BASE/transformer/diffusion_pytorch_model.safetensors.index.json"
if [[ ! -s "$INDEX" ]]; then
  echo "h3-base download failed: missing $INDEX" >&2
  exit 20
fi
if [[ "$(find "$BASE/transformer" -maxdepth 1 -name 'diffusion_pytorch_model-*.safetensors' | wc -l)" -ne 14 ]]; then
  echo "h3-base download incomplete: expected 14 shards" >&2
  exit 21
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
