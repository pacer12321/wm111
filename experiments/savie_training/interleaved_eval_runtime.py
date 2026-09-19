"""Inference-only odd/even sequence ownership; logical order at attention.

This adapter does not install an attention forward or change training code.
Consumers MUST select hidden states, RoPE and AdaLN using local_indices.
"""
import torch
from src.inference.utils.ulysses_runtime import UlyssesRuntime


class InterleavedEvalRuntime(UlyssesRuntime):
    def configure(self, sequence_length, num_heads):
        if self.world_size != 2 or self.branch_parallel:
            raise ValueError("This validated adapter requires two standard Ulysses ranks")
        super().configure(sequence_length, num_heads)
        if getattr(self, "_logical_indices", None) is None:
            self._logical_indices = torch.cat([
                torch.arange(r, sequence_length, 2, device=self.device)
                for r in range(2)
            ])
            self._inverse_indices = torch.argsort(self._logical_indices)
            self.local_indices = torch.arange(
                self.rank, sequence_length, 2, device=self.device
            )
            assert self.local_indices.numel() == self.splits[self.rank]

    def sequence_to_heads(self, tensor):
        rank_order = super().sequence_to_heads(tensor)
        return rank_order.index_select(0, self._inverse_indices)

    def heads_to_sequence(self, tensor):
        rank_order = tensor.index_select(0, self._logical_indices)
        return super().heads_to_sequence(rank_order)

    def gather_sequence(self, tensor):
        rank_order = super().gather_sequence(tensor)
        return rank_order.index_select(0, self._inverse_indices)

    def _local_video_frame_sums(self, local_x, layout):
        # Only target video participates in the SAViE linear branch.
        start = getattr(layout, "target_start", layout.video_start)
        end = getattr(layout, "target_end", layout.video_end)
        frame_rows = getattr(layout, "target_tokens_per_frame", layout.tokens_per_frame)
        sums = torch.zeros(layout.num_frames, local_x.shape[-1],
                           device=local_x.device, dtype=torch.float32)
        selected = (self.local_indices >= start) & (self.local_indices < end)
        frames = (self.local_indices[selected] - start) // frame_rows
        sums.index_add_(0, frames, local_x[selected].float())
        return sums


def test_collectives():
    """Run with torchrun --nproc-per-node=2; exact integer-valued CUDA tests."""
    import os
    import json
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    for length in (7, 128, 129, 257):
        rt = InterleavedEvalRuntime(rank, 2, rank, device, "nccl")
        rt.configure(length, 4)
        full = torch.arange(length * 4 * 3, device=device, dtype=torch.float32).reshape(length, 4, 3)
        local = full.index_select(0, rt.local_indices)
        heads = rt.sequence_to_heads(local)
        torch.testing.assert_close(heads, full[:, rank*2:(rank+1)*2], rtol=0, atol=0)
        restored = rt.heads_to_sequence(heads)
        torch.testing.assert_close(restored, local, rtol=0, atol=0)
        torch.testing.assert_close(rt.gather_sequence(local), full, rtol=0, atol=0)
        if rank == 0:
            print(json.dumps({"test":"interleaved_collectives", "sequence_length":length,
                              "exact_match":True}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    test_collectives()
