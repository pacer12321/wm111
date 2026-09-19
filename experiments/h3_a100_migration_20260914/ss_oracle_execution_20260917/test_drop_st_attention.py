from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    _build_interleaved_tensor_map,
    _interleaved_openvdn_softmax_attention,
)
from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import (
    OpenVDNLayout,
    openvdn_softmax_attention,
)


def _source_without_target_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_layout: OpenVDNLayout,
    source_layout: OpenVDNLayout,
    scale: float,
) -> torch.Tensor:
    """Reference B output with only the S-query -> T-key edges removed."""
    out = openvdn_softmax_attention(query, key, value, target_layout, scale=scale)
    source_queries = torch.arange(source_layout.video_start, source_layout.video_end)
    allowed_keys = torch.cat(
        (
            torch.arange(0, target_layout.video_start),
            torch.arange(target_layout.video_end, target_layout.used_len),
        )
    )
    source_out = F.scaled_dot_product_attention(
        query.index_select(0, source_queries).transpose(0, 1).unsqueeze(0),
        key.index_select(0, allowed_keys).transpose(0, 1).unsqueeze(0),
        value.index_select(0, allowed_keys).transpose(0, 1).unsqueeze(0),
        scale=scale,
    ).squeeze(0).transpose(0, 1)
    out.index_copy_(0, source_queries, source_out)
    return out


def _source_corresponding_target_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_layout: OpenVDNLayout,
    source_layout: OpenVDNLayout,
    scale: float,
) -> torch.Tensor:
    """Reference B output with S_i queries seeing only the matching T_i frame."""
    out = openvdn_softmax_attention(query, key, value, target_layout, scale=scale)
    non_target = torch.cat(
        (
            torch.arange(0, target_layout.video_start),
            torch.arange(target_layout.video_end, target_layout.used_len),
        )
    )
    for frame in range(source_layout.num_frames):
        source_start = source_layout.video_start + frame * source_layout.tokens_per_frame
        source_queries = torch.arange(
            source_start, source_start + source_layout.tokens_per_frame
        )
        target_start = target_layout.video_start + frame * target_layout.tokens_per_frame
        matching_target = torch.arange(
            target_start, target_start + target_layout.tokens_per_frame
        )
        allowed_keys = torch.cat((non_target, matching_target))
        source_out = F.scaled_dot_product_attention(
            query.index_select(0, source_queries).transpose(0, 1).unsqueeze(0),
            key.index_select(0, allowed_keys).transpose(0, 1).unsqueeze(0),
            value.index_select(0, allowed_keys).transpose(0, 1).unsqueeze(0),
            scale=scale,
        ).squeeze(0).transpose(0, 1)
        out.index_copy_(0, source_queries, source_out)
    return out


def main() -> None:
    torch.manual_seed(17)
    packed_len = 28
    world = 2
    source_layout = OpenVDNLayout(
        used_len=28,
        video_start=2,
        num_frames=3,
        tokens_per_frame=4,
        frame_height=2,
        frame_width=2,
        text_start=0,
        text_len=2,
    )
    target_layout = OpenVDNLayout(
        used_len=28,
        video_start=14,
        num_frames=3,
        tokens_per_frame=4,
        frame_height=2,
        frame_width=2,
        text_start=0,
        text_len=2,
    )
    mapping = _build_interleaved_tensor_map(packed_len, world, torch.device("cpu"))
    query = torch.randn(packed_len, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query_physical = query.index_select(0, mapping.physical_to_logical)
    key_physical = key.index_select(0, mapping.physical_to_logical)
    value_physical = value.index_select(0, mapping.physical_to_logical)

    reference = _source_without_target_reference(
        query, key, value, target_layout, source_layout, scale=0.5
    )
    physical = _interleaved_openvdn_softmax_attention(
        query_physical,
        key_physical,
        value_physical,
        target_layout,
        scale=0.5,
        logical_to_physical=mapping.logical_to_physical,
        active_query_mask=None,
        source_layout=source_layout,
        drop_source_to_target=True,
    )
    restored = physical.index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(restored, reference, rtol=1e-5, atol=1e-6)

    baseline = openvdn_softmax_attention(
        query, key, value, target_layout, scale=0.5
    )
    source_slice = slice(source_layout.video_start, source_layout.video_end)
    target_slice = slice(target_layout.video_start, target_layout.video_end)
    if torch.allclose(restored[source_slice], baseline[source_slice]):
        raise AssertionError("S->T ablation did not change source-query outputs")
    torch.testing.assert_close(
        restored[target_slice], baseline[target_slice], rtol=1e-5, atol=1e-6
    )

    active_logical = torch.ones(packed_len, dtype=torch.bool)
    active_logical[target_layout.video_start + 1] = False
    active_logical[target_layout.video_start + 6] = False
    active_physical = active_logical.index_select(0, mapping.physical_to_logical)
    partial = _interleaved_openvdn_softmax_attention(
        query_physical,
        key_physical,
        value_physical,
        target_layout,
        scale=0.5,
        logical_to_physical=mapping.logical_to_physical,
        active_query_mask=active_physical,
        source_layout=source_layout,
        drop_source_to_target=True,
    ).index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(
        partial[source_slice], reference[source_slice], rtol=1e-5, atol=1e-6
    )
    if torch.count_nonzero(partial[target_layout.video_start + 1]).item() != 0:
        raise AssertionError("inactive target query was unexpectedly computed")
    if torch.count_nonzero(partial[target_layout.video_start + 6]).item() != 0:
        raise AssertionError("inactive target query was unexpectedly computed")

    corresponding_reference = _source_corresponding_target_reference(
        query, key, value, target_layout, source_layout, scale=0.5
    )
    corresponding = _interleaved_openvdn_softmax_attention(
        query_physical,
        key_physical,
        value_physical,
        target_layout,
        scale=0.5,
        logical_to_physical=mapping.logical_to_physical,
        active_query_mask=None,
        source_layout=source_layout,
        source_to_target_corresponding=True,
    ).index_select(0, mapping.logical_to_physical)
    torch.testing.assert_close(
        corresponding, corresponding_reference, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        corresponding[target_slice], baseline[target_slice], rtol=1e-5, atol=1e-6
    )
    if torch.allclose(corresponding[source_slice], baseline[source_slice]):
        raise AssertionError("corresponding-frame S->T did not change source outputs")
    if torch.allclose(corresponding[source_slice], reference[source_slice]):
        raise AssertionError("corresponding-frame S->T unexpectedly equals full removal")
    print(
        "PASS: full-removal and corresponding-frame S->T routes match their "
        "logical-order references while keeping T-side routing unchanged"
    )


if __name__ == "__main__":
    main()
