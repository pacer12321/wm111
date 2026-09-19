#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/datasets/Ditto-1M
ARCHIVES="$ROOT/archives"
STATE="$ROOT/manifests/download_state.json"
PID_FILE="$ROOT/manifests/download.pid"
mkdir -p "$ARCHIVES" "$(dirname "$STATE")"
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

STATE="$STATE" python3 -c 'import json, os, time; p=os.environ["STATE"]; json.dump({"phase":"downloading","pid":os.getpid(),"started_at":time.time()}, open(p,"w"), indent=2)'

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Ditto's large archives may otherwise be redirected to Xet/CAS, which
# currently returns 401 on this host.  Force the resumable HTTP path served
# by the configured mirror instead.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

hf download QingyanBai/Ditto-1M \
  --repo-type dataset \
  --include 'videos/source/*' \
  --include 'videos/local/*' \
  --include 'videos/global_freeform1/*' \
  --include 'videos/global_freeform2/*' \
  --include 'videos/global_style1/*' \
  --include 'videos/global_style2/*' \
  --local-dir "$ARCHIVES" \
  --max-workers 8

STATE="$STATE" python3 -c 'import json, os, time; p=os.environ["STATE"]; json.dump({"phase":"download_complete","finished_at":time.time()}, open(p,"w"), indent=2)'
