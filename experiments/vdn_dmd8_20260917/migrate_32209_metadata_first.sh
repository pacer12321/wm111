#!/usr/bin/env bash
set -euo pipefail

src_host="ma-user@dev-modelarts-cnnorth9.huaweicloud.com"
ssh_cmd="ssh -p 30674 -i /temp/shared/keys/KeyPair-liusonghua.pem -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
src_root="/cache/zhonghao/datasets/Ditto-1M"
dst_root="/cache/zhonghao/datasets/Ditto-1M"

mkdir -p "$dst_root" /cache/zhonghao/migration_logs

for part in README.md manifests training_metadata tools logs; do
  rsync -a --partial --append-verify --info=progress2 -e "$ssh_cmd" \
    "$src_host:$src_root/$part" "$dst_root/"
done

date -Is > "$dst_root/METADATA_MIGRATION_COMPLETE"
