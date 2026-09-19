#!/usr/bin/env bash
set -euo pipefail

SRC_HOST="ma-user@dev-modelarts-cnnorth9.huaweicloud.com"
SSH_CMD="ssh -p 30674 -i /temp/shared/keys/KeyPair-liusonghua.pem -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ROOT="/cache/zhonghao/h3"

mkdir -p "$ROOT/dmd8_b_skip_20260917" "$ROOT/models" "$ROOT/data"

sync_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  rsync -a --partial --append-verify --info=progress2 -e "$SSH_CMD" "$SRC_HOST:$src/" "$dst/"
}

sync_dir /cache/zhonghao/h3/dmd8_b_skip_20260917/candidate_B_skip "$ROOT/dmd8_b_skip_20260917/candidate_B_skip"
sync_dir /cache/zhonghao/h3/dmd8_b_skip_20260917/loader "$ROOT/dmd8_b_skip_20260917/loader"
sync_dir /cache/zhonghao/h3/dmd8_b_skip_20260917/selector_dmd8_latent "$ROOT/dmd8_b_skip_20260917/selector_dmd8_latent"
sync_dir /cache/zhonghao/h3/dmd8_b_skip_20260917/results/B_DMD8_skip_refresh_1_5 "$ROOT/dmd8_b_skip_20260917/results/B_DMD8_skip_refresh_1_5"
rsync -a --partial --append-verify --info=progress2 -e "$SSH_CMD" \
  "$SRC_HOST:/cache/zhonghao/h3/dmd8_b_skip_20260917/launch_dmd8_b_skip.py" \
  "$ROOT/dmd8_b_skip_20260917/launch_dmd8_b_skip.py"
sync_dir /cache/zhonghao/h3/track_b_validation_20260915 "$ROOT/track_b_validation_20260915"
sync_dir /cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3 "$ROOT/models/OpenVDN-vdn-minimax-h3"
sync_dir /cache/zhonghao/h3/models/MiniMax-H3 "$ROOT/models/MiniMax-H3"
sync_dir /cache/zhonghao/h3/env_cuda_v1 "$ROOT/env_cuda_v1"
sync_dir /cache/zhonghao/h3/cuda_compat13 "$ROOT/cuda_compat13"
sync_dir /cache/zhonghao/h3/python312_runtime "$ROOT/python312_runtime"
sync_dir /cache/zhonghao/h3/data/shirt_red_couple_124 "$ROOT/data/shirt_red_couple_124"

date -Is > "$ROOT/MIGRATION_RUNTIME_COMPLETE"

