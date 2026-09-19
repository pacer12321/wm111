#!/usr/bin/env bash
set -euo pipefail

dir=/cache/zhonghao/h3/quality_compare_exp4_vae_20260915_v1
font=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
ffmpeg_bin=/usr/bin/ffmpeg

filter="[0:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=${font}:text='Source':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v0];[1:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=${font}:text='B  no skip':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v1];[2:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=${font}:text='4  latent selector':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v2];[3:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile=${font}:text='4  VAE selector':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v3];[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]"

"${ffmpeg_bin}" -y \
  -i "${dir}/source.mp4" \
  -i "${dir}/experiment2_B.mp4" \
  -i "${dir}/experiment4_token_skip.mp4" \
  -i "${dir}/experiment4_vae_perceptual.mp4" \
  -filter_complex "${filter}" -map '[v]' -c:v libx264 -preset fast -crf 18 \
  -pix_fmt yuv420p -shortest "${dir}/source_B_exp4latent_exp4vae_compare.mp4"

"${ffmpeg_bin}" -y -i "${dir}/source_B_exp4latent_exp4vae_compare.mp4" \
  -vf "select='eq(n,0)+eq(n,24)+eq(n,48)+eq(n,72)+eq(n,96)+eq(n,123)',scale=1280:-2,tile=2x3" \
  -frames:v 1 -update 1 "${dir}/source_B_exp4latent_exp4vae_contact_sheet.jpg"

"${ffmpeg_bin}" -i "${dir}/experiment2_B.mp4" -i "${dir}/experiment4_vae_perceptual.mp4" \
  -lavfi '[0:v][1:v]ssim;[0:v][1:v]psnr' -f null /dev/null \
  >"${dir}/B_vs_VAE_metrics.log" 2>&1 || true

"${ffmpeg_bin}" -i "${dir}/experiment4_token_skip.mp4" -i "${dir}/experiment4_vae_perceptual.mp4" \
  -lavfi '[0:v][1:v]ssim;[0:v][1:v]psnr' -f null /dev/null \
  >"${dir}/latent_vs_VAE_metrics.log" 2>&1 || true

sha256sum "${dir}/source_B_exp4latent_exp4vae_compare.mp4" \
  "${dir}/source_B_exp4latent_exp4vae_contact_sheet.jpg"
grep -E 'SSIM|PSNR' "${dir}/B_vs_VAE_metrics.log" | tail -n 2
grep -E 'SSIM|PSNR' "${dir}/latent_vs_VAE_metrics.log" | tail -n 2
