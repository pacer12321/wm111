from __future__ import annotations

import torch

from analyze_svoo_isa_query_block_oracle import _shared_exact_hybrid_output
from analyze_svoo_isa_branch_oracle import (
    _attention_output,
    _block_means,
    _pad_branch_blocks,
)


def test_all_exact_matches_dense_branch_composition() -> None:
    torch.manual_seed(19)
    queries, keys, heads, dim = 7, 19, 3, 8
    q = torch.randn(queries, heads, dim)
    k = torch.randn(keys, heads, dim)
    v = torch.randn(keys, heads, dim)
    branch_positions = torch.arange(5, keys)
    non_positions = torch.arange(5)
    block_size = 4
    blocked_k, valid, counts = _pad_branch_blocks(k, branch_positions, block_size)
    blocked_v, _, _ = _pad_branch_blocks(v, branch_positions, block_size)
    centroid_k = _block_means(blocked_k, counts)
    centroid_v = _block_means(blocked_v, counts)
    scale = dim**-0.5
    per_query_scores = torch.einsum("qhd,bhd->qhb", q, centroid_k) * scale
    exact_indices = torch.arange(blocked_k.shape[0]).view(1, -1).expand(heads, -1)
    actual = _shared_exact_hybrid_output(
        q,
        k[non_positions],
        v[non_positions],
        blocked_k,
        blocked_v,
        valid,
        counts,
        centroid_v,
        per_query_scores,
        exact_indices,
        scale,
    )
    expected = _attention_output(q, k, v, scale)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
