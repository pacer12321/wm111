$ErrorActionPreference = 'Stop'
$root = 'C:\Users\DZH\OneDrive\Documents\ChatGPT\wm111'
$ffmpeg = Join-Path $root '.local_video_deps\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
$dir = Join-Path $root 'comparisons\redshirt_exp2_exp3'
$source = Join-Path $dir 'source.mp4'
$baseline = Join-Path $dir 'experiment2_B.mp4'
$latent = Join-Path $dir 'experiment4_token_skip.mp4'
$vae = Join-Path $dir 'experiment4_vae_perceptual.mp4'
$output = Join-Path $dir 'source_B_exp4latent_exp4vae_compare.mp4'
$sheet = Join-Path $dir 'source_B_exp4latent_exp4vae_contact_sheet.jpg'

$filter = "[0:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='Source':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v0];[1:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='B  no skip':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v1];[2:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='4  latent selector':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v2];[3:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='4  VAE selector':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v3];[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]"

& $ffmpeg -y -i $source -i $baseline -i $latent -i $vae -filter_complex $filter -map '[v]' -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -shortest $output
if ($LASTEXITCODE -ne 0) { throw "comparison render failed: $LASTEXITCODE" }

& $ffmpeg -y -i $output -vf "select='eq(n,0)+eq(n,24)+eq(n,48)+eq(n,72)+eq(n,96)+eq(n,123)',scale=1280:-2,tile=2x3" -frames:v 1 -update 1 $sheet
if ($LASTEXITCODE -ne 0) { throw "contact sheet render failed: $LASTEXITCODE" }

Get-FileHash -Algorithm SHA256 $output
Get-FileHash -Algorithm SHA256 $sheet
