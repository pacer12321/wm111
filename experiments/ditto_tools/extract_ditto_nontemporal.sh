#!/usr/bin/env bash
set -euo pipefail

root=/cache/zhonghao/datasets/Ditto-1M
archives="$root/archives/videos"
members="$root/manifests/nontemporal_21882/members_nontemporal"
output="$root/liveedit_nontemporal_candidates/videos"
logs="$root/logs/nontemporal_extract"
mkdir -p "$output" "$logs"

groups=(source global_freeform1 global_freeform2 global_style1 global_style2 local)
for group in "${groups[@]}"; do
  status="$logs/${group}.complete"
  [[ -f "$status" ]] && continue
  mapfile -t parts < <(find "$archives/$group" -maxdepth 1 -type f -name '*.tar.gz.*' -print | sort)
  member_file="$members/$group.txt"
  [[ ${#parts[@]} -gt 0 ]] || { echo "missing archive parts: $group" >&2; exit 1; }
  [[ -s "$member_file" ]] || { echo "missing member list: $group" >&2; exit 1; }
  cat "${parts[@]}" | tar -xzf - -C "$output" -T "$member_file"
  expected=$(wc -l < "$member_file")
  found=0
  while IFS= read -r relpath; do
    [[ -f "$output/$relpath" ]] && found=$((found + 1))
  done < "$member_file"
  printf '%s expected=%s found=%s\n' "$group" "$expected" "$found" | tee "$status"
  [[ "$found" -eq "$expected" ]] || exit 1
done

date -Iseconds > "$logs/all.complete"
