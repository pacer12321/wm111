#!/usr/bin/env bash
set -euo pipefail

root=/cache/zhonghao/h3
review="$root/interleaved_sp_partial_latent_formal_20260917_v2/review"
mkdir -p "$review"

ffmpeg -hide_banner -loglevel error -y \
  -i "$root/data/shirt_red_couple_124/source.mp4" \
  -i "$root/bk_oracle_redshirt_B_formal_20260916_v1/timing_1/output.mp4" \
  -i "$root/interleaved_sp_partial_latent_formal_20260917_v2/formal_50step/output.mp4" \
  -filter_complex "[0:v]select='not(mod(n\,30))',scale=448:256,tile=5x1[s];[1:v]select='not(mod(n\,30))',scale=448:256,tile=5x1[b];[2:v]select='not(mod(n\,30))',scale=448:256,tile=5x1[p];[s][b][p]vstack=inputs=3[out]" \
  -map "[out]" -frames:v 1 "$review/source_B_interleaved_contact.png"

ffmpeg -hide_banner -loglevel error -y \
  -i "$root/data/shirt_red_couple_124/source.mp4" \
  -i "$root/bk_oracle_redshirt_B_formal_20260916_v1/timing_1/output.mp4" \
  -i "$root/interleaved_sp_partial_latent_formal_20260917_v2/formal_50step/output.mp4" \
  -filter_complex "[0:v]scale=448:256[s];[1:v]scale=448:256[b];[2:v]scale=448:256[p];[s][b][p]hstack=inputs=3[out]" \
  -map "[out]" -an -c:v libx264 -preset veryfast -crf 23 \
  "$review/source_B_interleaved_side_by_side.mp4"

ffmpeg -hide_banner -i \
  "$root/bk_oracle_redshirt_B_formal_20260916_v1/timing_1/output.mp4" \
  -i "$root/interleaved_sp_partial_latent_formal_20260917_v2/formal_50step/output.mp4" \
  -lavfi ssim -f null - 2> "$review/ssim.txt"

ffmpeg -hide_banner -i \
  "$root/bk_oracle_redshirt_B_formal_20260916_v1/timing_1/output.mp4" \
  -i "$root/interleaved_sp_partial_latent_formal_20260917_v2/formal_50step/output.mp4" \
  -lavfi psnr -f null - 2> "$review/psnr.txt"
