$ErrorActionPreference = 'Stop'

$root = 'C:\Users\DZH\OneDrive\Documents\ChatGPT\wm111'
$deps = Join-Path $root '.local_video_deps'
$ffmpeg = Join-Path $deps 'imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'

$dir = Join-Path $root 'comparisons\redshirt_exp2_exp3'
$source = Join-Path $dir 'source.mp4'
$exp2 = Join-Path $dir 'experiment2_B.mp4'
$exp3 = Join-Path $dir 'experiment3_ss_local_linear.mp4'
$output = Join-Path $dir 'source_exp2_exp3_compare.mp4'

$filter = "[0:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='Source':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v0];[1:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='2  B  S-S dense':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v1];[2:v]fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:410:(ow-iw)/2:50:black,drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='3  S-S local+linear':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=10[v2];[v0][v1][v2]hstack=inputs=3[v]"

$ffargs = @(
    '-y', '-i', $source, '-i', $exp2, '-i', $exp3,
    '-filter_complex', $filter,
    '-map', '[v]', '-map', '0:a?',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-shortest', $output
)

& $ffmpeg @ffargs
if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed with exit code $LASTEXITCODE" }

Get-FileHash -Algorithm SHA256 $output
Get-Item $output | Select-Object FullName, Length, LastWriteTime
