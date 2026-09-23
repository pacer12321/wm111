#!/usr/bin/env bash
# One-off: convert native MiniMax-H3/Ref2VA into the h3-base (diffusers) layout for training.
# CPU only; peak RAM about one 5 GB shard; writes ~66 GB. Refuses to overwrite.
set -euo pipefail

ROOT=/cache/zhonghao/h3
MODELS="$ROOT/models"
FL2VA="$MODELS/MiniMax-H3/FL2VA"
FL2VA_ARGS=()
# Optional: also prove h3-base == convert(FL2VA), i.e. the two bases differ only in weights.
if [[ -d "$FL2VA/transformer" ]]; then FL2VA_ARGS=(--fl2va "$FL2VA"); fi

cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
exec "$ROOT/env_cuda_v1/bin/python" -u build_ref2va_base.py \
  --ref2va "$MODELS/MiniMax-H3/Ref2VA" \
  --template "$MODELS/OpenVDN-vdn-minimax-h3/h3-base" \
  --output "$MODELS/OpenVDN-vdn-minimax-h3/ref2va-base" \
  "${FL2VA_ARGS[@]}" "$@"
