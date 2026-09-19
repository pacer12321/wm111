#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/datasets/Ditto-1M
DOWNLOAD_PID="$ROOT/manifests/download.pid"
STATE="$ROOT/manifests/download_state.json"
LOG="$ROOT/logs/finalize.log"

echo "Waiting for archive download" >> "$LOG"
while [[ -s "$DOWNLOAD_PID" ]]; do
  pid=$(<"$DOWNLOAD_PID")
  if kill -0 "$pid" 2>/dev/null; then
    sleep 60
  else
    rm -f "$DOWNLOAD_PID"
    break
  fi
done

if ! grep -q 'download_complete' "$STATE"; then
  echo "Download did not complete successfully" >> "$LOG"
  exit 1
fi

echo "Starting selected-member extraction" >> "$LOG"
DELETE_ARCHIVES_AFTER_VERIFY=0 bash "$ROOT/tools/extract_ditto_selected.sh" >> "$LOG" 2>&1

echo "Starting ffprobe QC and final split" >> "$LOG"
python3 "$ROOT/tools/qc_and_finalize_ditto20k.py" \
  "$ROOT/manifests/liveedit_candidates_25k.jsonl" \
  "$ROOT/liveedit_25k_candidates/videos" \
  --output-dir "$ROOT/liveedit_20k" \
  --target-size 20000 \
  --workers 24 >> "$LOG" 2>&1

date -Iseconds > "$ROOT/logs/finalize_complete.timestamp"
