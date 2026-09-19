#!/usr/bin/env bash
set -euo pipefail

SRC_HOST="ma-user@dev-modelarts-cnnorth9.huaweicloud.com"
SSH_CMD="ssh -p 30674 -i /temp/shared/keys/KeyPair-liusonghua.pem -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
DEST="/cache/zhonghao/datasets/Ditto-1M"

mkdir -p "$DEST"
rsync -a --partial --append-verify --info=progress2 -e "$SSH_CMD" \
  "$SRC_HOST:/cache/zhonghao/datasets/Ditto-1M/" "$DEST/"
date -Is > "$DEST/MIGRATION_COMPLETE"
