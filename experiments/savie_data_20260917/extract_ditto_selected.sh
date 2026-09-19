#!/usr/bin/env bash
set -euo pipefail

ROOT=/cache/zhonghao/datasets/Ditto-1M
ARCHIVES="$ROOT/archives/videos"
MEMBERS="$ROOT/manifests/members_25k"
OUTPUT="$ROOT/liveedit_25k_candidates/videos"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUTPUT" "$LOG_DIR"

groups=(source global_freeform1 global_freeform2 global_style1 global_style2 local)
for group in "${groups[@]}"; do
  mapfile -t parts < <(find "$ARCHIVES/$group" -maxdepth 1 -type f -name '*.tar.gz.*' -print | sort)
  if [[ ${#parts[@]} -eq 0 ]]; then
    echo "Missing archive parts for $group" >&2
    exit 1
  fi
  member_file="$MEMBERS/$group.txt"
  if [[ ! -s "$member_file" ]]; then
    echo "Missing member list for $group" >&2
    exit 1
  fi

  cat "${parts[@]}" | tar -xzf - -C "$OUTPUT" -T "$member_file"

  expected=$(wc -l < "$member_file")
  found=0
  while IFS= read -r relpath; do
    [[ -f "$OUTPUT/$relpath" ]] && found=$((found + 1))
  done < "$member_file"
  printf '%s expected=%s found=%s\n' "$group" "$expected" "$found" | tee "$LOG_DIR/extract_${group}.status"
  if [[ "$found" -ne "$expected" ]]; then
    echo "Extraction verification failed for $group" >&2
    exit 1
  fi

  if [[ "${DELETE_ARCHIVES_AFTER_VERIFY:-0}" == "1" ]]; then
    rm -f -- "${parts[@]}"
  fi
done

date -Iseconds > "$LOG_DIR/extract_complete.timestamp"
