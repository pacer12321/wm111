#!/usr/bin/env bash
set -euo pipefail

base=/cache/zhonghao/h3/compare_redshirt_exp2_exp3
src=/cache/zhonghao/h3/data/shirt_red_couple_124/source.mp4
b="$base/experiment2_B.mp4"
e3=/cache/zhonghao/h3/ablation3_ss_hybrid_redshirt_20260915_v4/formal_50step/output.mp4
out="$base/source_exp2_exp3_compare.mp4"

ffmpeg -y \
  -i "$src" -i "$b" -i "$e3" \
  -filter_complex "[0:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Source':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v0];[1:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='2  B  S-S dense':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v1];[2:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='3  S-S local+linear':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v2];[v0][v1][v2]hstack=inputs=3[v]" \
  -map "[v]" -map 0:a? -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -c:a aac -shortest "$out"

ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of json "$out"
sha256sum "$out"
