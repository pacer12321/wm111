"""Video-only MODEL input policy, separate from the preserved source audio track.

No automatic install. A serving launcher must opt in after train/infer parity
passes. Export must mux the untouched original source track, not decode this
placeholder as output audio.
"""
import torch


def install_fixed_silent_audio_inputs(branch_class):
    if getattr(branch_class, "_savie_fixed_silent_installed", False):
        return
    original = branch_class.forward_kwargs

    def forward_kwargs(self, **kwargs):
        if not bool(self.audio_update_mask.all()):
            raise ValueError("video-only input requires a video-only prepared reference; retain original audio at export")
        kwargs["audio_rows"] = torch.zeros_like(kwargs["audio_rows"])
        return original(self, **kwargs)

    branch_class.forward_kwargs = forward_kwargs
    branch_class._savie_fixed_silent_installed = True


def original_audio_mux_command(edited_video, original_source, destination,
                               duration_seconds, has_audio):
    """Copy, never synthesize/re-encode, the matching original audio interval.

Input files are never overwritten. The caller must inspect source streams and
validate the edited duration before invoking this command. No audio => video only.
"""
    from pathlib import Path
    if duration_seconds <= 0:
        raise ValueError("positive edited duration required")
    if Path(destination).resolve() in {Path(edited_video).resolve(), Path(original_source).resolve()}:
        raise ValueError("output must not overwrite original source or generated video")
    command = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-i", str(edited_video)]
    if has_audio:
        command += ["-i", str(original_source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        command += ["-map", "0:v:0", "-an"]
    return command + ["-c:v", "copy", "-t", str(duration_seconds), str(destination)]
