from __future__ import annotations

import torch

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    _active_query_openvdn_softmax_attention,
    _build_interleaved_tensor_map,
    _interleaved_local_frame_sums_counts,
    _interleaved_linear_forward_head_shard,
    _interleaved_openvdn_softmax_attention,
)
from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import (
    BidirectionalLinearBranch,
    OpenVDNLayout,
    local_frame_sums_counts,
    openvdn_softmax_attention,
)


def main() -> None:
    torch.manual_seed(7)
    packed_len = 28
    world = 2
    layout = OpenVDNLayout(
        used_len=26,
        video_start=6,
        num_frames=5,
        tokens_per_frame=4,
        frame_height=2,
        frame_width=2,
        text_start=0,
        text_len=2,
    )
    mapping = _build_interleaved_tensor_map(packed_len, world, torch.device("cpu"))
    q = torch.randn(packed_len, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_physical = q.index_select(0, mapping.physical_to_logical)
    k_physical = k.index_select(0, mapping.physical_to_logical)
    v_physical = v.index_select(0, mapping.physical_to_logical)

    dense_reference = openvdn_softmax_attention(q, k, v, layout, scale=0.5)
    dense_physical = _interleaved_openvdn_softmax_attention(
        q_physical,
        k_physical,
        v_physical,
        layout,
        scale=0.5,
        logical_to_physical=mapping.logical_to_physical,
        active_query_mask=None,
    )
    dense_restored = dense_physical.index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(dense_restored, dense_reference, rtol=1e-5, atol=1e-6)

    active_logical = torch.ones(packed_len, dtype=torch.bool)
    active_logical[9] = False
    active_logical[14:17] = False
    active_physical = active_logical.index_select(0, mapping.physical_to_logical)
    partial_reference = _active_query_openvdn_softmax_attention(
        q,
        k,
        v,
        layout,
        scale=0.5,
        active_query_mask=active_logical,
    )
    partial_physical = _interleaved_openvdn_softmax_attention(
        q_physical,
        k_physical,
        v_physical,
        layout,
        scale=0.5,
        logical_to_physical=mapping.logical_to_physical,
        active_query_mask=active_physical,
    )
    partial_restored = partial_physical.index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(partial_restored, partial_reference, rtol=1e-5, atol=1e-6)

    hidden = torch.randn(packed_len, 7)
    hidden_physical = hidden.index_select(0, mapping.physical_to_logical)
    reference_sums = torch.zeros(layout.num_frames - 2, hidden.shape[-1])
    reference_counts = torch.zeros(layout.num_frames - 2)
    mapped_sums = torch.zeros_like(reference_sums)
    mapped_counts = torch.zeros_like(reference_counts)
    for rank in range(world):
        start = rank * mapping.local_rows
        stop = start + mapping.local_rows
        sums, counts = local_frame_sums_counts(
            hidden[start:stop],
            layout,
            start,
        )
        reference_sums += sums
        reference_counts += counts
        sums, counts = _interleaved_local_frame_sums_counts(
            hidden_physical[start:stop],
            layout,
            mapping.physical_to_logical[start:stop],
        )
        mapped_sums += sums
        mapped_counts += counts
    torch.testing.assert_close(mapped_sums, reference_sums)
    torch.testing.assert_close(mapped_counts, reference_counts)

    branch = BidirectionalLinearBranch(hidden_size=8, num_heads=2, head_dim=4).eval()
    with torch.no_grad():
        for parameter in branch.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    qkv_linear = tuple(torch.randn(packed_len, 2, 4, dtype=torch.bfloat16) for _ in range(3))
    beta_logits = torch.randn(packed_len, 2, dtype=torch.bfloat16)
    frame_mean = torch.randn(layout.num_frames - 2, 8)
    with torch.no_grad():
        linear_reference = branch.forward_head_shard(
            qkv_linear,
            beta_logits,
            frame_mean,
            layout,
            head_start=0,
        )
    qkv_linear_physical = tuple(
        tensor.index_select(0, mapping.physical_to_logical) for tensor in qkv_linear
    )
    beta_physical = beta_logits.index_select(0, mapping.physical_to_logical)
    with torch.no_grad():
        linear_physical = _interleaved_linear_forward_head_shard(
            branch,
            qkv_linear_physical,
            beta_physical,
            frame_mean,
            layout,
            mapping.logical_to_physical,
            head_start=0,
        )
    linear_restored = linear_physical.index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(linear_restored, linear_reference, rtol=1e-4, atol=1e-4)

    boundary_rows = torch.tensor([0, 1, 5, 6, 25, 26, 27])
    restored_boundaries = mapping.physical_to_logical.index_select(
        0,
        mapping.logical_to_physical.index_select(0, boundary_rows),
    )
    torch.testing.assert_close(restored_boundaries, boundary_rows)
    print("PASS: dense, partial, linear, frame-reduce, and boundary mappings are equivalent")


if __name__ == "__main__":
    main()
