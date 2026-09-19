"""Compatibility entrypoint; uses the SAME shared-clip contract as every VAE producer.

Legacy --frames/--video-root calls intentionally fail rather than reintroducing
the 49-frame truncation and divergent Qwen preprocessing.
"""
from precompute_h3_video_latents import main

if __name__ == "__main__":
    main()
