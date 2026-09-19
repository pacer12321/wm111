#!/usr/bin/env bash
set -euo pipefail
ROOT=/cache/zhonghao/h3
REPO="$ROOT/train_repos/vdn-minimax-h3"
# These must point to the newly encoded, validated data. Never fall back to the
# old 12-latent-frame cache or the incomplete tag-only preflight.
: "${SAVIE_RETRAIN_OUT:?new empty output directory required}"
: "${SAVIE_RETRAIN_SAMPLES:?new shared-clip sample directory required}"
: "${SAVIE_RETRAIN_MANIFEST:?frozen 2K manifest required}"
: "${SAVIE_READY_REPORT:?full real-data test receipts required}"
: "${SAVIE_AUDIO_POLICY:?explicit matching training/inference audio policy required}"
OUT="$SAVIE_RETRAIN_OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$ROOT/cuda_compat13/usr/local/cuda-13.0/compat:$ROOT/env_cuda_v1/lib:${LD_LIBRARY_PATH:-}"
"$ROOT/env_cuda_v1/bin/python" -u "$ROOT/retrain_readiness.py" \
  --report "$SAVIE_READY_REPORT" --manifest "$SAVIE_RETRAIN_MANIFEST" \
  --sample-dir "$SAVIE_RETRAIN_SAMPLES" --output "$OUT" --audio-policy "$SAVIE_AUDIO_POLICY"
exec "$ROOT/env_cuda_v1/bin/torchrun" --standalone --nproc-per-node=4 \
  -m src.training.train_ref2va_dual \
  --base "$ROOT/models/OpenVDN-vdn-minimax-h3/h3-base" \
  --dmd8 "$ROOT/models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250" \
  --output "$OUT" \
  --sample-dir "$SAVIE_RETRAIN_SAMPLES" --audio-input-policy "$SAVIE_AUDIO_POLICY" \
  --sample-count 2000 --max-steps 1000 --save-every 100 \
  --lr 1e-6 --lora-rank 64 --lora-alpha 64 --shard-size 2 \
  --no-train-token-skip --no-cpu-offload
