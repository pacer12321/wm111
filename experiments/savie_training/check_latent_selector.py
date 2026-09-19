import torch

from src.training.ref2va_batch import (
    latent_selector_mask,
    patchify_video_latents,
    unpatchify_video_rows,
)


def main():
    generator = torch.Generator().manual_seed(4101)
    source = torch.randn((1, 24, 2, 8, 8), generator=generator)
    rows = patchify_video_latents(source, (1, 2, 2))
    restored = unpatchify_video_rows(rows, tuple(source.shape))
    assert torch.equal(source, restored), "patch/unpatch must be bit-exact"
    first_x0 = source + 0.01 * torch.randn(source.shape, generator=generator)
    active, scores, threshold = latent_selector_mask(source, first_x0)
    assert active.shape == scores.shape == (32,)
    assert 0 < int(active.sum()) <= active.numel()
    assert scores.min() < threshold < scores.max()
    print({
        "status": "LATENT_SELECTOR_UNIT_OK",
        "active_rows": int(active.sum()),
        "total_rows": active.numel(),
        "threshold": threshold,
    })


if __name__ == "__main__":
    main()
