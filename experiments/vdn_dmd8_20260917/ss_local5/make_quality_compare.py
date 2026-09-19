import subprocess
from pathlib import Path


BASE = Path("/cache/zhonghao/h3/dmd8_b_skip_20260917/results/B_DMD8_skip_refresh_1_5/formal_9step/output.mp4")
NEW = Path("/cache/zhonghao/h3/dmd8_b_skip_20260917/results/B_DMD8_skip_sslocal_pm5_refresh_1_5_v2/formal_9step/output.mp4")
OUT = NEW.parents[1]


def run(args):
    print("RUN", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, text=True, capture_output=True)


metric = run([
    "ffmpeg", "-hide_banner", "-i", str(BASE), "-i", str(NEW),
    "-lavfi",
    f"[0:v][1:v]ssim=stats_file={OUT / 'ssim.log'};"
    f"[0:v][1:v]psnr=stats_file={OUT / 'psnr.log'}",
    "-f", "null", "-",
])
print(metric.stderr[-4000:], flush=True)

run([
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", str(BASE), "-i", str(NEW),
    "-filter_complex",
    "[0:v]scale=640:-2,drawtext=text='Baseline B+latent skip':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55[a];"
    "[1:v]scale=640:-2,drawtext=text='B+latent skip+S-S local +/-5':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55[b];"
    "[a][b]hstack=inputs=2[v]",
    "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
    str(OUT / "baseline_vs_sslocal5.mp4"),
])

for sec, name in [(0.0, "start"), (1.0, "early"), (2.0, "middle"), (3.0, "late")]:
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(sec),
        "-i", str(OUT / "baseline_vs_sslocal5.mp4"), "-frames:v", "1",
        str(OUT / f"compare_{name}.png"),
    ])

print("DONE", OUT, flush=True)
