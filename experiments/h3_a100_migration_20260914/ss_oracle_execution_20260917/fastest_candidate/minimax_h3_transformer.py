# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-token audio/video DiT for vLLM-Omni.

vLLM tensor parallel linears and the unified attention layer provide TP and
Ulysses/Ring sequence parallel execution without changing the checkpoint
layout.
"""

from __future__ import annotations

import math
import os
import json
from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from cache_dit import ForwardPattern
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode
from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import (
    BidirectionalLinearBranch,
    OpenVDNLayout,
    OutputGate,
    _activate,
    _delta_factor_apply,
    _frame_statistics,
    _fusion_attention_tnd,
    _gather_linear_state,
    _scan_states,
    _softmax_plan,
    grouped_anchor_softmax_attention,
    infer_openvdn_layout,
    local_frame_sums_counts,
    openvdn_softmax_attention,
    window_bounds,
)
from vllm_omni.diffusion.models.minimax_h3.interleaved_sequence_map import (
    InterleavedSequenceMap,
)
from vllm_omni.diffusion.models.minimax_h3.openvdn_source_hybrid import (
    source_target_hybrid_softmax_attention,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class _BlockCudaProfiler:
    """Low-overhead per-DiT-forward CUDA event profiler."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("ZHONGHAO_H3_BLOCK_PROFILE", "0") == "1"
        self.output_dir = Path(
            os.environ.get("ZHONGHAO_H3_BLOCK_PROFILE_DIR", "/tmp/h3_block_profile")
        )
        self.records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.metadata: dict[str, Any] = {}

    def begin(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self.records.clear()
        self.metadata = metadata

    @contextmanager
    def scope(self, name: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self.records.append((name, start, end))

    def flush(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        event_errors: list[dict[str, str]] = []
        for name, start, end in self.records:
            try:
                end.synchronize()
                elapsed = float(start.elapsed_time(end))
            except (RuntimeError, ValueError) as exc:
                event_errors.append({"name": name, "error": repr(exc)})
                continue
            totals[name] = totals.get(name, 0.0) + elapsed
            counts[name] = counts.get(name, 0) + 1
        block_total = totals.get("block_total", 0.0)
        percentages = {
            name: (100.0 * value / block_total if block_total else 0.0)
            for name, value in totals.items()
            if name != "block_total"
        }
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        payload = {
            **self.metadata,
            "rank": rank,
            "timings_ms": totals,
            "counts": counts,
            "percent_of_block_total": percentages,
            "event_errors": event_errors,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f"rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        logger.info(
            "H3_BLOCK_PROFILE step=%s mode=%s rank=%d block_total_ms=%.3f qkv_ms=%.3f "
            "softmax_ms=%.3f mlp_ms=%.3f",
            self.metadata.get("step"), self.metadata.get("mode"), rank, block_total,
            totals.get("qkv_projection", 0.0), totals.get("softmax_attention_core", 0.0),
            totals.get("mlp", 0.0),
        )
        self.records.clear()


_BLOCK_PROFILER = _BlockCudaProfiler()


def _profile_scope(name: str):
    return _BLOCK_PROFILER.scope(name)


@dataclass
class MiniMaxH3DiTArchConfig:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    openvdn_enabled: bool = False
    openvdn_num_frames: int = 102
    openvdn_frame_height: int = 24
    openvdn_frame_width: int = 42
    openvdn_groups_per_call: int = 4
    openvdn_interior_group_size: int = 0
    openvdn_group_impl: str = "varlen"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
        fields = cls.__dataclass_fields__
        values = {name: config[name] for name in fields if name in config}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        arch = cls(**values)
        if len(arch.patch_size) != 3:
            raise ValueError(f"patch_size must contain three values, got {arch.patch_size!r}")
        return arch


_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()
_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32

MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})

# AdaLN modality count: token tags carry -1 for padding and 0/1/2 for
# video/text/audio tokens (padding is clamped to 0 before the embedding
# lookup and masked out afterwards).
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass(frozen=True)
class _InterleavedTensorMap:
    """Device tensors for logical/original and physical/rank-concatenated rows."""

    logical_to_physical: torch.Tensor
    physical_to_logical: torch.Tensor
    world_size: int
    local_rows: int


def _build_interleaved_tensor_map(
    length: int,
    world_size: int,
    device: torch.device,
) -> _InterleavedTensorMap:
    specification = InterleavedSequenceMap(length=length, world_size=world_size)
    logical = torch.arange(length, device=device, dtype=torch.long)
    logical_to_physical = (
        logical.remainder(world_size) * specification.local_rows
        + torch.div(logical, world_size, rounding_mode="floor")
    )
    physical = torch.arange(length, device=device, dtype=torch.long)
    physical_to_logical = (
        physical.remainder(specification.local_rows) * world_size
        + torch.div(physical, specification.local_rows, rounding_mode="floor")
    )
    if not torch.equal(
        physical_to_logical.index_select(0, logical_to_physical),
        logical,
    ):
        raise RuntimeError("interleaved sequence map is not bijective")
    return _InterleavedTensorMap(
        logical_to_physical=logical_to_physical,
        physical_to_logical=physical_to_logical,
        world_size=world_size,
        local_rows=specification.local_rows,
    )


def _interleaved_local_frame_sums_counts(
    x_local: torch.Tensor,
    layout: OpenVDNLayout,
    local_logical_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame statistics for a non-contiguous, interleaved local row shard."""
    if x_local.ndim != 2 or local_logical_indices.shape != (x_local.shape[0],):
        raise ValueError("interleaved local rows and logical indices must align")
    frames = layout.num_frames - 2
    sums = torch.zeros(
        (frames, x_local.shape[-1]),
        device=x_local.device,
        dtype=torch.float32,
    )
    counts = torch.zeros(frames, device=x_local.device, dtype=torch.float32)
    interior_start = layout.video_start + layout.tokens_per_frame
    interior_end = layout.video_end - layout.tokens_per_frame
    selected = (local_logical_indices >= interior_start) & (
        local_logical_indices < interior_end
    )
    if selected.any():
        local_rows = torch.nonzero(selected, as_tuple=False).view(-1)
        logical_rows = local_logical_indices.index_select(0, local_rows)
        frame_ids = torch.div(
            logical_rows - interior_start,
            layout.tokens_per_frame,
            rounding_mode="floor",
        )
        sums.index_add_(0, frame_ids, x_local.index_select(0, local_rows).float())
        counts.index_add_(
            0,
            frame_ids,
            torch.ones_like(frame_ids, dtype=torch.float32),
        )
    return sums, counts


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> nn.RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply per-index affine modulation: x * (1 + scale[idx]) + shift[idx].
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply the per-index gated residual: x + gate[idx] * other.
    return (x + gate.index_select(0, indices) * other).to(dtype)


class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate the first rot_dim head dims; pass the rest through.

    x: [T, heads, head_dim]; freqs: [T, rot_dim]. In the unfused path, cos/sin
    are cast to the activation dtype before the elementwise math.
    """
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)  # [T, 1, rot_dim]
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=False,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32.

        The sinusoidal embedding stays fp32 throughout and concatenates cosine
        values before sine values.
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=_FP32_DTYPE, device=t.device) / half)
        args = t.to(_FP32_DTYPE)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden, _ = self.proj_in(t_freq)
        hidden = nn.functional.silu(hidden)
        out, _ = self.proj_out(hidden)
        return out


def _sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Segment-wise SDPA equivalent of the non-causal varlen FA call.

    Mirrors the generic attention layer's semantics: FA is the fast path,
    SDPA is the correctness fallback when the platform resolves another
    backend. Segments are delimited by ``cu_seqlens`` exactly like the
    varlen kernel, so attention never crosses packed-document boundaries.
    """
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = torch.nn.functional.scaled_dot_product_attention(
            seg_q,
            seg_k,
            seg_v,
            scale=softmax_scale,
        )
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        skip_sequence_parallel: bool = False,
        enable_openvdn: bool = False,
    ) -> None:
        super().__init__()
        self.total_num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        inner_dim = self.total_num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=arch.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            return_bias=True,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self._install_qkv_weight_loader(arch)
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            softmax_scale=self.softmax_scale,
            causal=False,
            skip_sequence_parallel=skip_sequence_parallel,
        )
        self.openvdn_enabled = bool(arch.openvdn_enabled and enable_openvdn)
        self.openvdn_linear_enabled = True
        self.openvdn_source_hybrid_enabled = (
            os.environ.get("ZHONGHAO_H3_ENABLE_SOURCE_HYBRID", "0") == "1"
        )
        self.drop_source_to_target = (
            os.environ.get("ZHONGHAO_H3_DROP_SOURCE_TO_TARGET", "0") == "1"
        )
        self.source_to_target_corresponding = (
            os.environ.get("ZHONGHAO_H3_SOURCE_TO_TARGET_CORRESPONDING", "0")
            == "1"
        )
        if self.drop_source_to_target and self.source_to_target_corresponding:
            raise ValueError(
                "S-to-T cannot be both fully dropped and corresponding-frame-only"
            )
        self.openvdn_groups_per_call = int(arch.openvdn_groups_per_call)
        self.openvdn_interior_group_size = int(arch.openvdn_interior_group_size)
        self.openvdn_group_impl = str(arch.openvdn_group_impl)
        self._block_index = -1
        self._equivalence_captured = False
        # Read-only source-query Q/K/V capture used by the S-S spatial-local
        # oracle.  It is deliberately independent from the attention route so
        # enabling it cannot change the fastest interleaved SpotEdit forward.
        self._ss_oracle_capture_context: dict[str, Any] | None = None
        self._ss_oracle_captured_steps: set[int] = set()
        if self.openvdn_enabled:
            if get_tensor_model_parallel_world_size() != 1 or arch.num_attention_heads != self.num_heads:
                raise ValueError("The initial OpenVDN NPU port supports tensor_parallel_size=1 only")
            self.softmax_gate = OutputGate(
                arch.hidden_size,
                self.num_heads,
                dtype=_BF16_DTYPE,
            )
            self.linear_attention = BidirectionalLinearBranch(
                arch.hidden_size,
                self.num_heads,
                self.head_dim,
            )
            self.to_out_linear = nn.Linear(
                inner_dim,
                arch.hidden_size,
                bias=False,
                dtype=_BF16_DTYPE,
            )

    def _install_qkv_weight_loader(self, arch: MiniMaxH3DiTArchConfig) -> None:
        base_loader = self.qkv_proj.weight.weight_loader

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # The grouped checkpoint layout is
            # [num_query_groups, q_per_group + k + v] before splitting.
            # MiniMax H3 uses MHA, so checkpoint rows are per-head [q, k, v],
            # while qkv_proj expects [q_all, k_all, v_all].
            reordered = _reorder_grouped_qkv_to_qkv(
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )
            base_loader(param, reordered)

        self.qkv_proj.weight.weight_loader = _weight_loader

    @torch.compiler.disable
    def _run_packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run packed attention as a small eager island.

        The scalar packed-layout metadata and the CuTe FlashAttention-4 DSL
        are intentionally opaque to Dynamo. Keeping this boundary narrow lets
        regional compile fuse projections, norms, RoPE, and the surrounding
        DiT block without repeatedly graph-breaking inside the FA4 compiler.
        """
        used = int(cu_seqlens[1].item())
        packed_total = int(cu_seqlens[-1].item())
        attn_mask = None
        if used < packed_total:
            attn_mask = torch.arange(packed_total, device=q.device)[None] < used
        metadata = AttentionMetadata(
            attn_mask=attn_mask,
            extra={
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
            },
        )
        return self.attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            metadata,
        ).squeeze(0)

    def _openvdn_softmax(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: OpenVDNLayout,
        source_layout: OpenVDNLayout | None = None,
        active_query_mask: torch.Tensor | None = None,
        interleaved_map: _InterleavedTensorMap | None = None,
    ) -> torch.Tensor:
        if interleaved_map is not None:
            if self.openvdn_source_hybrid_enabled or self.openvdn_interior_group_size:
                raise ValueError("interleaved SP currently supports the standard B OpenVDN route only")
            if (
                self.drop_source_to_target
                or self.source_to_target_corresponding
            ) and source_layout is None:
                raise ValueError("the selected S-to-T route requires the source layout")
            return _interleaved_openvdn_softmax_attention(
                q,
                k,
                v,
                layout,
                self.softmax_scale,
                interleaved_map.logical_to_physical,
                active_query_mask,
                source_layout=source_layout,
                drop_source_to_target=self.drop_source_to_target,
                source_to_target_corresponding=(
                    self.source_to_target_corresponding
                ),
                groups_per_call=self.openvdn_groups_per_call,
            )
        if active_query_mask is not None:
            if self.openvdn_source_hybrid_enabled or self.openvdn_interior_group_size:
                raise ValueError("partial-query SpotEdit supports the standard B OpenVDN route only")
            return _active_query_openvdn_softmax_attention(
                q,
                k,
                v,
                layout,
                self.softmax_scale,
                active_query_mask,
                groups_per_call=self.openvdn_groups_per_call,
            )
        if self.openvdn_source_hybrid_enabled:
            if source_layout is None:
                raise ValueError("source layout is required for the S-S hybrid ablation")
            return source_target_hybrid_softmax_attention(
                q,
                k,
                v,
                source_layout,
                layout,
                self.softmax_scale,
                groups_per_call=self.openvdn_groups_per_call,
            )
        if self.openvdn_interior_group_size:
            return grouped_anchor_softmax_attention(
                q, k, v, layout, self.softmax_scale,
                group_size=self.openvdn_interior_group_size,
                groups_per_call=self.openvdn_groups_per_call,
                implementation=self.openvdn_group_impl,
            )
        return openvdn_softmax_attention(
            q, k, v, layout, self.softmax_scale,
            groups_per_call=self.openvdn_groups_per_call,
        )

    @torch.compiler.disable
    def _run_openvdn_ulysses(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: OpenVDNLayout,
        source_layout: OpenVDNLayout | None,
        cu_seqlens: torch.Tensor,
        strategy: Any,
        active_query_mask_local: torch.Tensor | None,
        interleaved_map: _InterleavedTensorMap | None,
        local_logical_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Head-sharded hybrid attention; return local sequence rows.

        This uses the same all-to-all strategy as dense H3. The VDN scan is
        complete in time but only computes heads owned by this rank. Hidden
        states are never full-sequence all-gathered to duplicate the scan.
        """
        sp_group = get_sp_group()
        world, rank = sp_group.ulysses_world_size, sp_group.ulysses_rank
        if strategy.name != "ulysses" or sp_group.ring_world_size != 1:
            raise ValueError("OpenVDN supports only pure Ulysses with ring_degree=1")
        if get_tensor_model_parallel_world_size() != 1 or self.num_heads != self.total_num_heads:
            raise ValueError("OpenVDN requires tensor_parallel_size=1")
        if get_ulysses_mode(default="strict") != "strict":
            raise ValueError("OpenVDN currently requires equal-row strict Ulysses, not advanced_uaa")
        if self.num_heads % world or self.num_kv_heads != self.num_heads:
            raise ValueError("OpenVDN requires MHA heads divisible by Ulysses degree")
        if self.attention.scatter_idx != 2 or self.attention.gather_idx != 1:
            raise ValueError("OpenVDN requires standard sequence/head Ulysses axes")
        if getattr(self.attention, "_kv_cache_dtype", None) is not None:
            raise ValueError("OpenVDN's explicit kernel does not implement quantized KV caches")
        packed_len = int(cu_seqlens[-1].item())
        if packed_len != x.shape[0] * world:
            raise ValueError("OpenVDN local row shards do not reconstruct the global packed sequence")
        layout.validate(packed_len)

        with _profile_scope("ulysses_softmax_pre_attention"):
            q_head, k_head, v_head, _metadata, ctx = strategy.pre_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None,
            )
        if q_head.shape[1:] != (packed_len, self.num_heads // world, self.head_dim):
            raise ValueError("Unexpected OpenVDN post-Ulysses layout")
        active_query_mask_global = None
        if active_query_mask_local is not None:
            with _profile_scope("ulysses_active_mask_allgather"):
                local_u8 = active_query_mask_local.to(device=x.device, dtype=torch.uint8).contiguous()
                gathered = [torch.empty_like(local_u8) for _ in range(world)]
                dist.all_gather(gathered, local_u8, group=ctx.ulysses_pg)
                active_query_mask_global = torch.cat(gathered).to(torch.bool)
        with _profile_scope("softmax_attention_core"):
            softmax_head = self._openvdn_softmax(
                q_head[0], k_head[0], v_head[0], layout, source_layout,
                active_query_mask=active_query_mask_global,
                interleaved_map=interleaved_map,
            )
        with _profile_scope("ulysses_softmax_post_attention"):
            softmax_local = strategy.post_attention(softmax_head.unsqueeze(0), ctx).squeeze(0)
        del q_head, k_head, softmax_head
        if not self.openvdn_linear_enabled:
            return softmax_local, None

        # Raw Q/K need their own exchange because Softmax Q/K were normalized
        # and RoPE-rotated. Raw V is unchanged, so reuse v_head. The third
        # exchange carries scalar beta logits instead of redundantly moving V.
        with _profile_scope("linear_beta_projection"):
            beta_local = self.linear_attention.beta_proj(x).unsqueeze(-1)
        with _profile_scope("ulysses_linear_pre_attention"):
            raw_q, raw_k, beta_head, _metadata, linear_ctx = strategy.pre_attention(
                qkv_raw[0].unsqueeze(0), qkv_raw[1].unsqueeze(0), beta_local.unsqueeze(0), None,
            )
        with _profile_scope("ulysses_linear_frame_reduce"):
            if interleaved_map is None:
                sums, counts = local_frame_sums_counts(x, layout, rank * x.shape[0])
            else:
                if local_logical_indices is None:
                    raise ValueError("interleaved SP requires local logical row indices")
                sums, counts = _interleaved_local_frame_sums_counts(
                    x,
                    layout,
                    local_logical_indices,
                )
            dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=ctx.ulysses_pg)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=ctx.ulysses_pg)
        if not bool(torch.all(counts == layout.tokens_per_frame).item()):
            raise ValueError("OpenVDN sequence shards do not cover each target interior frame exactly once")
        with _profile_scope("linear_attention_core"):
            frame_mean = sums / counts[:, None]
            if interleaved_map is None:
                readout_head = self.linear_attention.forward_head_shard(
                    (raw_q[0], raw_k[0], v_head[0]),
                    beta_head[0, :, :, 0],
                    frame_mean,
                    layout,
                    head_start=rank * (self.num_heads // world),
                )
            else:
                readout_head = _interleaved_linear_forward_head_shard(
                    self.linear_attention,
                    (raw_q[0], raw_k[0], v_head[0]),
                    beta_head[0, :, :, 0],
                    frame_mean,
                    layout,
                    interleaved_map.logical_to_physical,
                    head_start=rank * (self.num_heads // world),
                )
        if self.openvdn_source_hybrid_enabled:
            if source_layout is None:
                raise ValueError("source layout is required for the S-S linear branch")
            source_sums, source_counts = local_frame_sums_counts(
                x, source_layout, rank * x.shape[0]
            )
            dist.all_reduce(source_sums, op=dist.ReduceOp.SUM, group=ctx.ulysses_pg)
            dist.all_reduce(source_counts, op=dist.ReduceOp.SUM, group=ctx.ulysses_pg)
            if not bool(torch.all(source_counts == source_layout.tokens_per_frame).item()):
                raise ValueError("sequence shards do not cover each source interior frame")
            readout_head = readout_head + self.linear_attention.forward_head_shard(
                (raw_q[0], raw_k[0], v_head[0]),
                beta_head[0, :, :, 0],
                source_sums / source_counts[:, None],
                source_layout,
                head_start=rank * (self.num_heads // world),
            )
        with _profile_scope("ulysses_linear_post_attention"):
            readout_local = strategy.post_attention(readout_head.unsqueeze(0), linear_ctx).squeeze(0)
        with _profile_scope("linear_output_gate"):
            readout_local = readout_local * self.linear_attention.output_gate(x).to(readout_local.dtype)
        return softmax_local, readout_local.reshape(x.shape[0], self.num_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        sp_seq_lens: list[int] | None = None,
        openvdn_layout: OpenVDNLayout | None = None,
        openvdn_source_layout: OpenVDNLayout | None = None,
        spotedit_active_mask: torch.Tensor | None = None,
        spotedit_refresh: bool = False,
        interleaved_map: _InterleavedTensorMap | None = None,
        local_logical_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        total = x.shape[0]
        with _profile_scope("qkv_projection"):
            qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        qkv_raw = (q, k, v)
        with _profile_scope("qk_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_freqs is not None:
            with _profile_scope("rope"):
                q = _apply_rope(q, rope_freqs)
                k = _apply_rope(k, rope_freqs)

        capture = self._ss_oracle_capture_context
        if capture is not None:
            step = int(capture["step"])
            if (
                self._block_index in capture["layers"]
                and step not in self._ss_oracle_captured_steps
            ):
                sp_group = get_sp_group()
                world = int(sp_group.ulysses_world_size)
                rank = int(sp_group.ulysses_rank)
                if local_logical_indices is None:
                    global_start = rank * total
                    logical_indices = torch.arange(
                        global_start,
                        global_start + total,
                        device=q.device,
                        dtype=torch.long,
                    )
                else:
                    logical_indices = local_logical_indices.to(q.device)
                if logical_indices.shape != (total,):
                    raise ValueError(
                        "S-S oracle logical row map must match the local shard: "
                        f"{tuple(logical_indices.shape)} != {(total,)}"
                    )

                query_indices = capture["query_indices"].to(q.device)
                query_mask = torch.isin(logical_indices, query_indices)
                key_mask = logical_indices < int(capture["used_len"])
                output_dir = Path(capture["output_dir"]) / f"step_{step:02d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                rank_path = output_dir / (
                    f"layer_{self._block_index:02d}_rank_{rank}.pt"
                )
                temporary = rank_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "schema": "h3_fastest_ss_source_qkv_v1",
                        "attention_mode": "b_vdn_interleaved_latent_partial_skip",
                        "query_stream": "source",
                        "layer": self._block_index,
                        "step": step,
                        "rank": rank,
                        "world_size": world,
                        "interleaved": interleaved_map is not None,
                        "spotedit_refresh": bool(spotedit_refresh),
                        "active_target_ratio": capture["active_target_ratio"],
                        "num_heads": q.shape[1],
                        "head_dim": q.shape[2],
                        "softmax_scale": self.softmax_scale,
                        "q_selected": q[query_mask].detach().cpu(),
                        "query_indices": logical_indices[query_mask].detach().cpu(),
                        "k_local": k[key_mask].detach().cpu(),
                        "v_local": v[key_mask].detach().cpu(),
                        "key_indices": logical_indices[key_mask].detach().cpu(),
                        "all_query_indices": capture["query_indices"].cpu(),
                        "all_query_coords": capture["query_coords"].cpu(),
                        "source_positions": capture["source_positions"].cpu(),
                        "source_coords": capture["source_coords"].cpu(),
                        "target_positions": capture["target_positions"].cpu(),
                        "target_coords": capture["target_coords"].cpu(),
                        "layout": capture["layout"],
                    },
                    temporary,
                )
                temporary.replace(rank_path)
                logger.info(
                    "FASTEST_SS_SOURCE_QKV_CAPTURED step=%d layer=%d rank=%d "
                    "queries=%d keys=%d path=%s",
                    step,
                    self._block_index,
                    rank,
                    int(query_mask.sum().item()),
                    int(key_mask.sum().item()),
                    rank_path,
                )
                self._ss_oracle_captured_steps.add(step)

        # The packed layout uses a second document for alignment padding.
        # Local/Ulysses backends unpad it, while Ring keeps aligned rows for
        # fixed-size P2P buffers.
        linear_readout_local = None
        if self.openvdn_enabled:
            if openvdn_layout is None:
                raise ValueError("openvdn_layout is required when OpenVDN attention is enabled")
            strategy = self.attention._get_active_parallel_strategy()
            active_query_mask = (
                spotedit_active_mask
                if spotedit_active_mask is not None and not spotedit_refresh
                else None
            )
            if strategy.name == "ulysses":
                out, linear_readout_local = self._run_openvdn_ulysses(
                    x, q, k, v, qkv_raw, openvdn_layout, openvdn_source_layout,
                    cu_seqlens, strategy, active_query_mask,
                    interleaved_map, local_logical_indices,
                )
            elif strategy.name == "none":
                if total != int(cu_seqlens[-1].item()):
                    raise ValueError("OpenVDN cannot apply a global layout to uncommunicated local rows")
                openvdn_layout.validate(total)
                out = self._openvdn_softmax(
                    q, k, v, openvdn_layout, openvdn_source_layout,
                    active_query_mask=active_query_mask,
                    interleaved_map=interleaved_map,
                )
            else:
                raise ValueError(f"Unsupported OpenVDN parallel strategy: {strategy.name}")
            with _profile_scope("softmax_output_gate"):
                out = out * self.softmax_gate(x).to(out.dtype)
        else:
            out = self._run_packed_attention(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        out = out.reshape(total, self.num_heads * self.head_dim)
        active_indices = None
        if spotedit_active_mask is not None and not spotedit_refresh:
            active_indices = torch.nonzero(spotedit_active_mask, as_tuple=False).view(-1)
            with _profile_scope("attention_out_projection"):
                projected, _ = self.out_proj(out.index_select(0, active_indices))
                sparse_projected = torch.zeros_like(x)
                sparse_projected.index_copy_(0, active_indices, projected)
                out = sparse_projected
        else:
            with _profile_scope("attention_out_projection"):
                out, _ = self.out_proj(out)
        if self.openvdn_enabled and self.openvdn_linear_enabled:
            if linear_readout_local is not None:
                # Non-target/anchor/padding rows were zero-filled before the
                # inverse all-to-all. The linear output projection has no bias.
                if active_indices is not None:
                    with _profile_scope("linear_out_projection"):
                        linear_active = linear_readout_local.index_select(0, active_indices).to(x.dtype)
                        projected_linear = self.to_out_linear(linear_active)
                        sparse_linear = torch.zeros_like(x)
                        sparse_linear.index_copy_(0, active_indices, projected_linear)
                        out = out + sparse_linear
                else:
                    with _profile_scope("linear_out_projection"):
                        out = out + self.to_out_linear(linear_readout_local.to(x.dtype))
            else:
                linear_readout = self.linear_attention(x, qkv_raw, openvdn_layout)
                linear_out = self.to_out_linear(linear_readout.to(x.dtype))
                video = slice(openvdn_layout.video_start, openvdn_layout.video_end)
                if torch.is_grad_enabled():
                    out = out.clone()
                out[video] += linear_out
                if self.openvdn_source_hybrid_enabled:
                    if openvdn_source_layout is None:
                        raise ValueError("source layout is required for the S-S linear branch")
                    source_readout = self.linear_attention(x, qkv_raw, openvdn_source_layout)
                    source_out = self.to_out_linear(source_readout.to(x.dtype))
                    source_video = slice(
                        openvdn_source_layout.video_start,
                        openvdn_source_layout.video_end,
                    )
                    out[source_video] += source_out
        capture_dir = os.environ.get("ZHONGHAO_H3_EQ_CAPTURE_DIR")
        capture_block = int(os.environ.get("ZHONGHAO_H3_EQ_CAPTURE_BLOCK", "0"))
        if (
            capture_dir
            and self._block_index == capture_block
            and not self._equivalence_captured
        ):
            try:
                rank = int(get_sp_group().ulysses_rank)
            except Exception:
                rank = 0
            capture_path = Path(capture_dir)
            capture_path.mkdir(parents=True, exist_ok=True)
            if local_logical_indices is None:
                start = rank * out.shape[0]
                captured_logical_indices = torch.arange(
                    start,
                    start + out.shape[0],
                    device=out.device,
                    dtype=torch.long,
                )
            else:
                captured_logical_indices = local_logical_indices
            temporary = capture_path / f"attention_block{capture_block}_rank{rank}.pt.tmp"
            final = capture_path / f"attention_block{capture_block}_rank{rank}.pt"
            torch.save(
                {
                    "schema": "h3_attention_equivalence_capture_v1",
                    "block": self._block_index,
                    "rank": rank,
                    "interleaved": interleaved_map is not None,
                    "logical_indices": captured_logical_indices.detach().cpu(),
                    "attention_output": out.detach().cpu(),
                },
                temporary,
            )
            temporary.replace(final)
            self._equivalence_captured = True
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size, arch.ffn_hidden_size],
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        self._install_fc1_weight_loader()
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )

    def _install_fc1_weight_loader(self) -> None:
        base_loader = self.fc1.weight.weight_loader

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape[0] % 2:
                raise ValueError(
                    "MiniMax H3 fc1 checkpoint rows must split evenly into "
                    f"gate/up matrices, got {tuple(loaded_weight.shape)}"
                )
            gate, up = loaded_weight.chunk(2, dim=0)
            base_loader(param, gate, 0)
            base_loader(param, up, 1)

        self.fc1.weight.weight_loader = _weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        gate, up = hidden.chunk(2, dim=-1)
        hidden = nn.functional.silu(gate) * up
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """t_emb: [M, t_dim] -> expand_ratio tensors of [M*modality_num, H]."""
        x = nn.functional.silu(t_emb)
        x, _ = self.linear(x.to(self.linear.weight.dtype))
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Standard pre-norm transformer block without AdaLN or RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        # Text refinement runs on replicated rows before ``sp_prepare``.
        # Applying Ulysses here would all-to-all an unsharded sequence while
        # retaining the original packed ``cu_seqlens`` metadata.
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            skip_sequence_parallel=True,
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_freqs=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(arch, quant_config) for _ in range(arch.token_refiner_num_layers)]
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, quant_config, enable_openvdn=True)
        self.mlp = MiniMaxH3MLP(arch, quant_config)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            quant_config,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        sp_seq_lens: list[int] | None = None,
        openvdn_layout: OpenVDNLayout | None = None,
        openvdn_source_layout: OpenVDNLayout | None = None,
        spotedit_active_mask: torch.Tensor | None = None,
        spotedit_refresh: bool = False,
        interleaved_map: _InterleavedTensorMap | None = None,
        local_logical_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [T, H]; t_emb: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        with _profile_scope("adaln_projection"):
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaln_proj(t_emb)

        if spotedit_active_mask is not None:
            spotedit_active_mask = spotedit_active_mask.to(device=x.device, dtype=torch.bool)
            if spotedit_active_mask.shape != (x.shape[0],):
                raise ValueError(
                    "spotedit_active_mask must cover this SP row shard: "
                    f"{tuple(spotedit_active_mask.shape)} != {(x.shape[0],)}"
                )

        residual = x
        with _profile_scope("norm1_and_modulation"):
            h = self.norm1(x)
            h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices, dtype=_BF16_DTYPE)
        h = self.attn(
            h,
            rope_freqs=rope_freqs,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sp_seq_lens=sp_seq_lens,
            openvdn_layout=openvdn_layout,
            openvdn_source_layout=openvdn_source_layout,
            spotedit_active_mask=spotedit_active_mask,
            spotedit_refresh=spotedit_refresh,
            interleaved_map=interleaved_map,
            local_logical_indices=local_logical_indices,
        )
        with _profile_scope("attention_residual_gate"):
            x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16_DTYPE)

        residual = x
        with _profile_scope("norm2_and_modulation"):
            h = self.norm2(x)
            h = _modulate_scale_shift(h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16_DTYPE)
        if spotedit_active_mask is not None and not spotedit_refresh:
            active_indices = torch.nonzero(spotedit_active_mask, as_tuple=False).view(-1)
            with _profile_scope("mlp"):
                h_active = self.mlp(h.index_select(0, active_indices))
                h_sparse = torch.zeros_like(h)
                h_sparse.index_copy_(0, active_indices, h_active)
                h = h_sparse
        else:
            with _profile_scope("mlp"):
                h = self.mlp(h)
        with _profile_scope("mlp_residual_gate"):
            output = _modulate_gate(residual, gate_mlp, h, combined_indices, dtype=_BF16_DTYPE)

        if spotedit_active_mask is not None:
            with _profile_scope("stable_cache_update"):
                stable_indices = torch.nonzero(~spotedit_active_mask, as_tuple=False).view(-1)
                if spotedit_refresh:
                    self._spotedit_stable_output = output.index_select(0, stable_indices).detach().clone()
                else:
                    cached = getattr(self, "_spotedit_stable_output", None)
                    if cached is None or cached.shape != (stable_indices.numel(), output.shape[-1]):
                        raise RuntimeError("SpotEdit stable block cache is missing or has the wrong shape")
                    output = output.clone()
                    output.index_copy_(0, stable_indices, cached.to(device=output.device, dtype=output.dtype))
        return output


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            quant_config,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [T, H] -> (video_logits [T, 96] fp32, audio_logits [T, 32] fp32).

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        """
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices, dtype=_BF16_DTYPE)
        # Preserve full precision through both final output projections.
        h = h.to(_FP32_DTYPE)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio


class MiniMaxH3SPPrepare(nn.Module):
    """Explicit boundary for sharding packed rows and their metadata together."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        combined_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden_states, rope_freqs, combined_indices


class MiniMaxH3SPGather(nn.Module):
    """Explicit boundary for restoring packed rows after the block stack."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class MiniMaxH3DiTModel(nn.Module):
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"blocks": ForwardPattern.Pattern_3},
        # H3 is CFG-distilled and performs one transformer forward per step.
        has_separate_cfg=False,
        check_forward_pattern=False,
    )
    _repeated_blocks = ["MiniMaxH3DiTBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        del module
        parts = name.split(".")
        return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
    _hsdp_ignored_modules = [
        "video_patch_proj",
        "audio_patch_proj",
        "time_embedder",
        "final_layer",
    ]
    _sp_plan = {
        "sp_prepare": {
            0: SequenceParallelInput(
                split_dim=0,
                expected_dims=2,
                split_output=True,
            ),
            1: SequenceParallelInput(
                split_dim=0,
                expected_dims=2,
                split_output=True,
            ),
            2: SequenceParallelInput(
                split_dim=0,
                expected_dims=1,
                split_output=True,
            ),
        },
        "sp_gather": SequenceParallelOutput(gather_dim=0, expected_dims=2),
    }
    packed_modules_mapping = {}

    def _validate_tp_config(self, *, arch: MiniMaxH3DiTArchConfig, tp_size: int) -> None:
        if tp_size < 1:
            raise ValueError(f"tensor_parallel_size must be positive, got {tp_size}")
        if arch.num_attention_heads % tp_size:
            raise ValueError(
                "num_attention_heads must be divisible by tensor_parallel_size: "
                f"{arch.num_attention_heads} % {tp_size} != 0"
            )
        if arch.ffn_hidden_size % tp_size:
            raise ValueError(
                f"ffn_hidden_size must be divisible by tensor_parallel_size: {arch.ffn_hidden_size} % {tp_size} != 0"
            )
        if arch.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if arch.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if arch.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if arch.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        tf_config = od_config.tf_model_config
        config_mapping = tf_config.to_dict() if hasattr(tf_config, "to_dict") else dict(tf_config)
        arch = MiniMaxH3DiTArchConfig.from_mapping(config_mapping)
        self.arch = arch
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self._validate_tp_config(
            arch=arch,
            tp_size=get_tensor_model_parallel_world_size(),
        )
        local_heads = arch.num_attention_heads // get_tensor_model_parallel_world_size()
        ulysses_degree = int(self.parallel_config.ulysses_degree)
        if local_heads % ulysses_degree:
            raise ValueError(
                "MiniMax H3 local attention heads must be divisible by "
                "ulysses_degree: "
                f"({arch.num_attention_heads} / "
                f"{get_tensor_model_parallel_world_size()}) % "
                f"{ulysses_degree} != 0"
            )

        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(arch, quant_config)
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch, quant_config)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch, quant_config) for _ in range(arch.num_layers)])
        for block_index, block in enumerate(self.blocks):
            block.attn._block_index = block_index
        self.sp_prepare = MiniMaxH3SPPrepare()
        self.sp_gather = MiniMaxH3SPGather()
        self.final_layer = MiniMaxH3FinalLayer(arch, quant_config)
        self._source_hybrid_enabled = (
            os.environ.get("ZHONGHAO_H3_ENABLE_SOURCE_HYBRID", "0") == "1"
        )
        self._drop_source_to_target = (
            os.environ.get("ZHONGHAO_H3_DROP_SOURCE_TO_TARGET", "0") == "1"
        )
        self._source_to_target_corresponding = (
            os.environ.get("ZHONGHAO_H3_SOURCE_TO_TARGET_CORRESPONDING", "0")
            == "1"
        )
        if self._drop_source_to_target and self._source_to_target_corresponding:
            raise ValueError(
                "S-to-T cannot be both fully dropped and corresponding-frame-only"
            )
        self._interleaved_sp_enabled = (
            os.environ.get("ZHONGHAO_H3_INTERLEAVED_SP", "0") == "1"
        )
        if self._interleaved_sp_enabled and self._source_hybrid_enabled:
            raise ValueError("interleaved SP currently supports standard B OpenVDN only")
        self._spotedit_payload_path = os.environ.get("ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD")
        self._spotedit_payload: dict[str, Any] | None = None
        self._spotedit_step = 0
        self._spotedit_last_target_t: float | None = None
        self._spotedit_input_cache: torch.Tensor | None = None
        self._spotedit_initial_steps = int(os.environ.get("ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS", "4"))
        reset_text = os.environ.get("ZHONGHAO_H3_SPOTEDIT_RESET_STEPS", "13,22,31")
        self._spotedit_reset_steps = {int(value) for value in reset_text.split(",") if value.strip()}
        self._mark_missing_params_required()

    def _load_spotedit_payload(self, target_count: int) -> dict[str, Any] | None:
        if not self._spotedit_payload_path:
            return None
        if self._spotedit_payload is None:
            path = Path(self._spotedit_payload_path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("schema") != "h3_fixed_selector_payload_v1":
                raise ValueError(f"unsupported SpotEdit payload schema: {payload.get('schema')!r}")
            active = payload["active_target_mask"].view(-1).to(torch.bool).contiguous()
            source = payload["source_clean_normalized_rows"].float().contiguous()
            if active.numel() != target_count or source.shape[0] != target_count:
                raise ValueError(
                    "SpotEdit payload target size mismatch: "
                    f"active={active.numel()} source={source.shape[0]} request={target_count}"
                )
            if source.shape[1] != self.arch.latents_dim * math.prod(self.arch.patch_size):
                raise ValueError(f"SpotEdit source row width mismatch: {source.shape[1]}")
            self._spotedit_payload = {**payload, "active_target_mask": active, "source_clean_normalized_rows": source}
            logger.info(
                "SPOTEDIT_PAYLOAD_LOADED path=%s target_tokens=%d active_ratio=%.6f "
                "attention_policy=partial_query_openvdn_full_kv_mlp_skip",
                path,
                target_count,
                float(active.float().mean()),
            )
        return self._spotedit_payload

    def _mark_missing_params_required(self) -> None:
        for _, param in self.named_parameters():
            param.missing_param_init = "error"

    def post_load_weights(self) -> None:
        for name, param in self.named_parameters():
            if name in MINIMAX_H3_FP32_PARAM_NAMES and param.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        for name, buffer in self.named_buffers():
            if name in MINIMAX_H3_FP32_BUFFER_NAMES and buffer.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {buffer.dtype}.")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load exact H3 checkpoint names with logical TP-aware loaders."""
        params = dict(self.named_parameters())
        params.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            param = params.get(name)
            if param is None:
                logger.warning("Skipping MiniMax H3 weight not present in model: %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        return loaded

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if isinstance(pos_info, dict):
            ids = pos_info.get("position_ids")
        else:
            ids = getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        if isinstance(psp, dict):
            value = psp.get(field)
        else:
            value = getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build packed multimodal embeddings for the TP=1/SP=1 inference path.

        Returns (decoder_input [S, H] bf16, t_emb [M, t_dim] fp32).
        """
        # Latent embedders stay fp32 in and out; their outputs are cast to the
        # bf16 sequence dtype only during indexed scattering.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(_FP32_DTYPE)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(_FP32_DTYPE)
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=_BF16_DTYPE)
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
        )

        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE)
        embeddings.index_add_(0, text_pos, text_embed.to(_BF16_DTYPE)[: text_pos.shape[0]])
        embeddings.index_add_(0, img_pos, video_embed.to(_BF16_DTYPE)[: img_pos.shape[0]])
        embeddings.index_add_(0, audio_pos, audio_embed.to(_BF16_DTYPE)[: audio_pos.shape[0]])

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward.

        Keyword names follow the checkpoint's serving contract.
        Returns `(video_logits, audio_logits)` from rows selected by
        `img_pos_for_infer_output_info` and `audio_pos_info`, with condition
        rows zeroed by update masks.
        """
        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required_kwarg(kwargs, "update_mask")
        token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_max = int(self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q"))

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device
        # Compute RoPE frequencies over the full packed sequence.
        rope_freqs = self.rope(img_position_ids).to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            seq_len=seq_len,
            device=device,
        )

        combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        block_rope = rope_freqs
        block_combined = combined_indices
        openvdn_layout = None
        openvdn_source_layout = None
        update = update_mask.view(-1).to(device=device, dtype=torch.bool)
        if self.arch.openvdn_enabled:
            if update.numel() != img_pos.numel():
                raise ValueError("OpenVDN update_mask must identify each source/target visual row")
            target_video_pos = img_pos.to(device).index_select(
                0,
                torch.nonzero(update, as_tuple=False).view(-1),
            )
            # The public five-block microbenchmark's 102x24x42 defaults are
            # not the real Ref2VA request layout. Infer frame/grid dimensions
            # from the selected target patch coordinates before SP sharding.
            openvdn_layout = infer_openvdn_layout(
                target_video_pos, text_pos, img_position_ids, cu_seqlens,
            )
            if (
                self._source_hybrid_enabled
                or self._drop_source_to_target
                or self._source_to_target_corresponding
            ):
                source_video_pos = img_pos.to(device).index_select(
                    0,
                    torch.nonzero(~update, as_tuple=False).view(-1),
                )
                openvdn_source_layout = infer_openvdn_layout(
                    source_video_pos, text_pos, img_position_ids, cu_seqlens,
                )
                if openvdn_source_layout.used_len != openvdn_layout.used_len:
                    raise ValueError("source and target layouts cover different packed requests")

        target_count = int(update.sum().item())
        spot_payload = self._load_spotedit_payload(target_count)
        spot_active_global = None
        spot_refresh = False
        spot_skip_step = False
        if spot_payload is not None:
            active_target = spot_payload["active_target_mask"].to(device=device)
            spot_active_global = torch.ones(seq_len, dtype=torch.bool, device=device)
            target_sequence_positions = img_pos.to(device).index_select(
                0, torch.nonzero(update, as_tuple=False).view(-1)
            )
            target_timestep_indices = inverse_indices.index_select(0, target_sequence_positions)
            target_t_values = unique_timesteps.to(device).index_select(0, target_timestep_indices)
            if not bool(torch.allclose(target_t_values, target_t_values[:1], rtol=0, atol=1e-7)):
                raise ValueError("SpotEdit target rows must share one diffusion timestep")
            current_target_t = float(target_t_values[0].detach().cpu())
            if (
                self._spotedit_last_target_t is not None
                and current_target_t < self._spotedit_last_target_t - 1e-7
            ):
                self._spotedit_step = 0
                self._spotedit_input_cache = None
                logger.info(
                    "SPOTEDIT_REQUEST_RESET previous_t=%.8f current_t=%.8f",
                    self._spotedit_last_target_t,
                    current_target_t,
                )
            self._spotedit_last_target_t = current_target_t
            spot_active_global.index_copy_(0, target_sequence_positions, active_target)
            source_sequence_positions = img_pos.to(device).index_select(
                0, torch.nonzero(~update, as_tuple=False).view(-1)
            )
            if not bool(torch.all(spot_active_global[source_sequence_positions]).item()):
                raise RuntimeError(
                    "SpotEdit must not mark source queries stable; S-S spatial-local "
                    "and target token-skip are defined on disjoint query sets"
                )
            spot_refresh = (
                self._spotedit_step < self._spotedit_initial_steps
                or self._spotedit_step in self._spotedit_reset_steps
            )
            spot_skip_step = not spot_refresh

        interleaved_map = None
        local_logical_indices = None
        if self._interleaved_sp_enabled:
            if spot_payload is None or openvdn_layout is None:
                raise ValueError("interleaved SP requires SpotEdit and OpenVDN")
            sp_group = get_sp_group()
            world = int(sp_group.ulysses_world_size)
            if world < 2 or int(sp_group.ring_world_size) != 1:
                raise ValueError("interleaved SP requires pure Ulysses with world_size >= 2")
            interleaved_map = _build_interleaved_tensor_map(seq_len, world, device)
            permutation = interleaved_map.physical_to_logical
            hidden = hidden.index_select(0, permutation)
            block_rope = block_rope.index_select(0, permutation)
            block_combined = block_combined.index_select(0, permutation)
            if spot_active_global is not None:
                spot_active_global = spot_active_global.index_select(0, permutation)

        hidden, block_rope, block_combined = self.sp_prepare(
            hidden,
            block_rope,
            block_combined,
        )
        spot_active_local = None
        if spot_active_global is not None:
            sp_group = get_sp_group()
            world = int(sp_group.ulysses_world_size)
            rank = int(sp_group.ulysses_rank)
            local_rows = int(hidden.shape[0])
            if local_rows * world != seq_len:
                raise ValueError(
                    "SpotEdit proxy requires strict equal-row Ulysses shards: "
                    f"local={local_rows} world={world} global={seq_len}"
                )
            start = rank * local_rows
            spot_active_local = spot_active_global[start : start + local_rows]
            if interleaved_map is not None:
                local_logical_indices = interleaved_map.physical_to_logical[
                    start : start + local_rows
                ]
            stable_local = ~spot_active_local
            if spot_refresh:
                self._spotedit_input_cache = hidden.index_select(
                    0, torch.nonzero(stable_local, as_tuple=False).view(-1)
                ).detach().clone()
            else:
                if self._spotedit_input_cache is None:
                    raise RuntimeError("SpotEdit input cache is missing before a skip step")
                stable_indices = torch.nonzero(stable_local, as_tuple=False).view(-1)
                if self._spotedit_input_cache.shape != (stable_indices.numel(), hidden.shape[-1]):
                    raise RuntimeError("SpotEdit input cache has the wrong shape")
                hidden = hidden.clone()
                hidden.index_copy_(
                    0,
                    stable_indices,
                    self._spotedit_input_cache.to(device=hidden.device, dtype=hidden.dtype),
                )
        profile_mode = "refresh" if spot_refresh else ("partial_skip" if spot_skip_step else "disabled")
        ss_capture_context = None
        ss_capture_dir = os.environ.get("ZHONGHAO_H3_SS_ORACLE_CAPTURE_DIR")
        ss_capture_steps = {
            int(value)
            for value in os.environ.get(
                "ZHONGHAO_H3_SS_ORACLE_CAPTURE_STEPS", "4"
            ).split(",")
            if value.strip()
        }
        ss_capture_layers = {
            int(value)
            for value in os.environ.get(
                "ZHONGHAO_H3_SS_ORACLE_CAPTURE_LAYERS", "8,24,41"
            ).split(",")
            if value.strip()
        }
        if ss_capture_dir and self._spotedit_step in ss_capture_steps:
            used_len = int(cu_seqlens[1].item())
            all_img_coords = img_position_ids.reshape(-1, 3)
            source_visual = torch.nonzero(~update, as_tuple=False).view(-1)
            target_visual = torch.nonzero(update, as_tuple=False).view(-1)
            source_positions = img_pos.to(device).index_select(0, source_visual)
            target_positions = img_pos.to(device).index_select(0, target_visual)
            source_positions = source_positions[source_positions < used_len]
            target_positions = target_positions[target_positions < used_len]
            source_coords = all_img_coords.index_select(0, source_positions)
            target_coords = all_img_coords.index_select(0, target_positions)
            source_times = torch.unique(source_coords[:, 0], sorted=True)
            source_ys = torch.unique(source_coords[:, 1], sorted=True)
            source_xs = torch.unique(source_coords[:, 2], sorted=True)
            frames = int(source_times.numel())
            height = int(source_ys.numel())
            width = int(source_xs.numel())
            per_frame = height * width
            if source_positions.numel() != frames * per_frame:
                raise ValueError(
                    "S-S oracle cannot form the source grid: "
                    f"positions={source_positions.numel()} grid={(frames, height, width)}"
                )
            if target_positions.numel() != source_positions.numel():
                raise ValueError("S-S oracle requires matching source and target grids")
            source_grid = source_positions.view(frames, height, width)
            ys = torch.linspace(
                0, height - 1, steps=4, device=device
            ).round().long().unique()
            xs = torch.linspace(
                0, width - 1, steps=4, device=device
            ).round().long().unique()
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            sampled = source_grid[:, yy.reshape(-1), xx.reshape(-1)].reshape(-1)
            ss_capture_context = {
                "output_dir": ss_capture_dir,
                "layers": ss_capture_layers,
                "step": int(self._spotedit_step),
                "used_len": used_len,
                "active_target_ratio": (
                    float(spot_payload["active_target_mask"].float().mean())
                    if spot_payload is not None else 1.0
                ),
                "query_indices": sampled.detach(),
                "query_coords": all_img_coords.index_select(0, sampled).detach(),
                "source_positions": source_positions.detach(),
                "source_coords": source_coords.detach(),
                "target_positions": target_positions.detach(),
                "target_coords": target_coords.detach(),
                "layout": {
                    "used_len": used_len,
                    "num_frames": frames,
                    "tokens_per_frame": per_frame,
                    "frame_height": height,
                    "frame_width": width,
                    "text_start": int(text_pos[0].item()),
                    "text_len": int(text_pos.numel()),
                    "ss_spatial_window": 5,
                    "ss_neighbors_per_source_frame": 25,
                },
            }
        _BLOCK_PROFILER.begin(
            step=int(self._spotedit_step),
            mode=profile_mode,
            active_target_ratio=(
                float(spot_payload["active_target_mask"].float().mean())
                if spot_payload is not None else 1.0
            ),
            local_sequence_rows=int(hidden.shape[0]),
            local_active_rows=(
                int(spot_active_local.sum().item())
                if spot_active_local is not None else int(hidden.shape[0])
            ),
            interleaved_sp=bool(interleaved_map is not None),
            num_blocks=len(self.blocks),
        )
        for block in self.blocks:
            block.attn._ss_oracle_capture_context = ss_capture_context
            with _profile_scope("block_total"):
                hidden = block(
                    hidden,
                    t_emb=t_emb,
                    combined_indices=block_combined,
                    rope_freqs=block_rope,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    openvdn_layout=openvdn_layout,
                    openvdn_source_layout=openvdn_source_layout,
                    spotedit_active_mask=spot_active_local,
                    spotedit_refresh=spot_refresh,
                    interleaved_map=interleaved_map,
                    local_logical_indices=local_logical_indices,
                )
            block.attn._ss_oracle_capture_context = None
        _BLOCK_PROFILER.flush()
        hidden = self.sp_gather(hidden)
        if interleaved_map is not None:
            hidden = hidden.index_select(0, interleaved_map.logical_to_physical)

        video_logits, audio_logits = self.final_layer(
            hidden,
            t_emb=t_emb,
            inverse_indices=inverse_indices,
        )

        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        if spot_payload is not None and spot_skip_step:
            active_target = spot_payload["active_target_mask"].to(device=device)
            stable_target = ~active_target
            target_visual_indices = torch.nonzero(update, as_tuple=False).view(-1)
            current_visual_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos.to(device))
            current_target_rows = current_visual_rows.index_select(0, target_visual_indices)
            source_rows = spot_payload["source_clean_normalized_rows"].to(
                device=device, dtype=current_target_rows.dtype
            )
            target_sequence_positions = img_pos.to(device).index_select(0, target_visual_indices)
            target_timestep_indices = inverse_indices.index_select(0, target_sequence_positions)
            target_timesteps = unique_timesteps.to(device).index_select(0, target_timestep_indices)
            if not bool(torch.allclose(target_timesteps, target_timesteps[:1], rtol=0, atol=1e-7)):
                raise ValueError("SpotEdit target rows must share one diffusion timestep")
            sigma = 1.0 - target_timesteps[0].to(dtype=current_target_rows.dtype)
            if not bool((sigma > 0).item()):
                raise ValueError("SpotEdit source velocity requires positive sigma")
            # SpotFusion-style gradual hand-off at the x0 level.  H3 exposes
            # velocity, so first reconstruct the model's current x0 estimate,
            # blend stable rows from cached/model state toward the aligned
            # source condition, then convert the fused x0 back to velocity.
            # H3's target timestep rises from 0 -> 1 over denoising; therefore
            # alpha starts near 1 (history/model estimate) and ends near 0
            # (source condition), matching alpha(t)=cos^2(pi*t/2).
            target_velocity = video_logits.index_select(0, target_visual_indices).to(
                current_target_rows.dtype
            )
            target_x0 = current_target_rows + sigma * target_velocity
            progress = target_timesteps[0].to(dtype=torch.float32).clamp(0.0, 1.0)
            fusion_alpha = torch.cos(progress * (math.pi / 2.0)).square().to(
                dtype=current_target_rows.dtype
            )
            fused_x0 = fusion_alpha * target_x0 + (1.0 - fusion_alpha) * source_rows
            fused_velocity = (fused_x0 - current_target_rows) / sigma
            stable_visual_indices = target_visual_indices.index_select(
                0, torch.nonzero(stable_target, as_tuple=False).view(-1)
            )
            stable_velocity = fused_velocity.index_select(
                0, torch.nonzero(stable_target, as_tuple=False).view(-1)
            )
            video_logits = video_logits.clone()
            video_logits.index_copy_(0, stable_visual_indices, stable_velocity.to(video_logits.dtype))
        if spot_payload is not None:
            try:
                rank = int(get_sp_group().ulysses_rank)
            except Exception:
                rank = 0
            if rank == 0:
                logger.info(
                    "SPOTEDIT_STEP step=%d mode=%s active_target_ratio=%.6f "
                    "attention=active_queries_full_kv mlp=active_only fusion_alpha=%.6f",
                    self._spotedit_step,
                    "FULL_REFRESH" if spot_refresh else "PARTIAL_QUERY_SKIP",
                    float(spot_payload["active_target_mask"].float().mean()),
                    float(fusion_alpha.detach().cpu()) if spot_skip_step else 1.0,
                )
            self._spotedit_step += 1
        return video_logits, audio_logits


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "_reorder_grouped_qkv_to_qkv",
]


def _interleaved_openvdn_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: OpenVDNLayout,
    scale: float,
    logical_to_physical: torch.Tensor,
    active_query_mask: torch.Tensor | None,
    source_layout: OpenVDNLayout | None = None,
    drop_source_to_target: bool = False,
    source_to_target_corresponding: bool = False,
    groups_per_call: int = 4,
) -> torch.Tensor:
    """OpenVDN softmax in physical order using logical frame-window plans."""
    if logical_to_physical.shape != (query.shape[0],):
        raise ValueError("logical_to_physical must cover the packed sequence")
    if active_query_mask is not None and active_query_mask.shape != (query.shape[0],):
        raise ValueError("active query mask must cover the physical packed sequence")

    dense_q_logical, _all_k, groups_logical = _softmax_plan(layout, query.device)
    valid_k_logical = torch.arange(layout.used_len, device=query.device, dtype=torch.long)
    valid_k = logical_to_physical.index_select(0, valid_k_logical)

    out = torch.zeros_like(query)

    # Standard B puts source rows in the global-query set, so source queries
    # normally attend the noisy target as well.  The S->T ablation splits only
    # those queries into a fixed-shape call whose K/V excludes the target-video
    # interval.  Every other query and every T->S/T->T route remains unchanged.
    if drop_source_to_target and source_to_target_corresponding:
        raise ValueError(
            "S-to-T cannot be both fully dropped and corresponding-frame-only"
        )
    if drop_source_to_target or source_to_target_corresponding:
        if source_layout is None:
            raise ValueError("source_layout is required for the selected S-to-T route")
        if source_to_target_corresponding and (
            source_layout.num_frames != layout.num_frames
            or source_layout.tokens_per_frame != layout.tokens_per_frame
        ):
            raise ValueError(
                "corresponding-frame S-to-T requires matching source/target grids"
            )
        source_query_mask = (
            (dense_q_logical >= source_layout.video_start)
            & (dense_q_logical < source_layout.video_end)
        )
        source_q_logical = dense_q_logical[source_query_mask]
        other_q_logical = dense_q_logical[~source_query_mask]
    else:
        source_q_logical = dense_q_logical[:0]
        other_q_logical = dense_q_logical

    other_q = logical_to_physical.index_select(0, other_q_logical)
    if active_query_mask is not None:
        other_q = other_q[active_query_mask.index_select(0, other_q)]
    if other_q.numel():
        dense_out = _fusion_attention_tnd(
            query.index_select(0, other_q),
            key.index_select(0, valid_k),
            value.index_select(0, valid_k),
            [other_q.numel()],
            [valid_k.numel()],
            scale,
        )
        out.index_copy_(0, other_q, dense_out)

    source_k_logical = torch.cat(
        (
            torch.arange(
                0, layout.video_start, device=query.device, dtype=torch.long
            ),
            torch.arange(
                layout.video_end,
                layout.used_len,
                device=query.device,
                dtype=torch.long,
            ),
        )
    )

    if drop_source_to_target and source_q_logical.numel():
        source_q = logical_to_physical.index_select(0, source_q_logical)
        if active_query_mask is not None:
            source_q = source_q[active_query_mask.index_select(0, source_q)]
        source_k = logical_to_physical.index_select(0, source_k_logical)
        source_out = _fusion_attention_tnd(
            query.index_select(0, source_q),
            key.index_select(0, source_k),
            value.index_select(0, source_k),
            [source_q.numel()],
            [source_k.numel()],
            scale,
        )
        out.index_copy_(0, source_q, source_out)

    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    if source_to_target_corresponding:
        assert source_layout is not None
        for frame in range(source_layout.num_frames):
            source_start = (
                source_layout.video_start
                + frame * source_layout.tokens_per_frame
            )
            source_frame_q_logical = torch.arange(
                source_start,
                source_start + source_layout.tokens_per_frame,
                device=query.device,
                dtype=torch.long,
            )
            target_start = layout.video_start + frame * layout.tokens_per_frame
            target_frame_k_logical = torch.arange(
                target_start,
                target_start + layout.tokens_per_frame,
                device=query.device,
                dtype=torch.long,
            )
            q_idx = logical_to_physical.index_select(
                0, source_frame_q_logical
            )
            k_idx = logical_to_physical.index_select(
                0, torch.cat((source_k_logical, target_frame_k_logical))
            )
            if active_query_mask is not None:
                q_idx = q_idx[active_query_mask.index_select(0, q_idx)]
            if q_idx.numel():
                groups.append((q_idx, k_idx))
    for q_logical, k_logical in groups_logical:
        q_idx = logical_to_physical.index_select(0, q_logical)
        k_idx = logical_to_physical.index_select(0, k_logical)
        if active_query_mask is not None:
            q_idx = q_idx[active_query_mask.index_select(0, q_idx)]
        if q_idx.numel():
            groups.append((q_idx, k_idx))

    for start in range(0, len(groups), groups_per_call):
        batch = groups[start : start + groups_per_call]
        q_indices = [item[0] for item in batch]
        k_indices = [item[1] for item in batch]
        q_pack = torch.cat([query.index_select(0, idx) for idx in q_indices])
        k_pack = torch.cat([key.index_select(0, idx) for idx in k_indices])
        v_pack = torch.cat([value.index_select(0, idx) for idx in k_indices])
        q_cu: list[int] = []
        kv_cu: list[int] = []
        q_total = kv_total = 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        group_out = _fusion_attention_tnd(q_pack, k_pack, v_pack, q_cu, kv_cu, scale)
        out.index_copy_(0, torch.cat(q_indices), group_out)
    return out


def _interleaved_linear_forward_head_shard(
    branch: BidirectionalLinearBranch,
    qkv_raw_physical: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    beta_logits_physical: torch.Tensor,
    frame_mean: torch.Tensor,
    layout: OpenVDNLayout,
    logical_to_physical: torch.Tensor,
    head_start: int,
) -> torch.Tensor:
    """VDN linear branch with logical frame order gathered from physical rows."""
    query_raw, key_raw, value_raw = qkv_raw_physical
    if query_raw.ndim != 3 or any(t.shape != query_raw.shape for t in qkv_raw_physical):
        raise ValueError("interleaved raw Q/K/V must have matching [rows, heads, dim] shapes")
    packed_len, heads, dim = query_raw.shape
    layout.validate(packed_len)
    if logical_to_physical.shape != (packed_len,):
        raise ValueError("logical_to_physical must cover the packed sequence")
    if dim != branch.head_dim or not 0 <= head_start < head_start + heads <= branch.num_heads:
        raise ValueError("invalid interleaved OpenVDN linear head shard")
    if beta_logits_physical.shape != (packed_len, heads):
        raise ValueError("interleaved beta logits must match packed rows and local heads")

    frames = layout.num_frames - 2
    per_frame = layout.tokens_per_frame
    interior_logical = torch.arange(
        layout.video_start + per_frame,
        layout.video_end - per_frame,
        device=query_raw.device,
        dtype=torch.long,
    )
    interior_physical = logical_to_physical.index_select(0, interior_logical)
    frame_size = (layout.frame_height, layout.frame_width)
    features = tuple(
        _activate(
            branch.short_conv.apply(
                name,
                raw.index_select(0, interior_physical),
                frames,
                frame_size,
                head_start,
            ),
            name != "v",
        )
        for name, raw in zip(("q", "k", "v"), qkv_raw_physical)
    )
    query, key_features, value_features = (
        tensor.view(frames, per_frame, heads, dim)
        .permute(0, 2, 1, 3)
        .contiguous()
        for tensor in features
    )
    beta = torch.sigmoid(beta_logits_physical.index_select(0, interior_physical))
    beta = beta.view(frames, per_frame, heads).permute(0, 2, 1).contiguous()
    A, B = _frame_statistics(key_features, value_features, beta)
    alpha = branch.alpha.forward_head_shard(frame_mean, head_start, heads)

    text_logical = torch.arange(
        layout.text_start,
        layout.text_start + layout.text_len,
        device=query_raw.device,
        dtype=torch.long,
    )
    text_physical = logical_to_physical.index_select(0, text_logical)
    text_key = _activate(key_raw.index_select(0, text_physical), True).transpose(0, 1).unsqueeze(0)
    text_value = _activate(value_raw.index_select(0, text_physical), False).transpose(0, 1).unsqueeze(0)
    text_beta = torch.sigmoid(beta_logits_physical.index_select(0, text_physical)).transpose(0, 1).unsqueeze(0)
    text_A, text_B = _frame_statistics(text_key, text_value, text_beta)
    ones = torch.ones((1, heads, dim), device=query_raw.device, dtype=torch.float32)
    _, text_injection = _delta_factor_apply("vdn_solve", ones, text_A, text_B, layout.text_len)
    text_state = branch.TEXT_STATE_SCALE * text_injection[0]
    prefix, suffix = _scan_states(
        alpha,
        A,
        B,
        text_state,
        delta_rule="vdn_solve",
        tokens_per_frame=per_frame,
    )
    bounds = [(lo - 1, hi - 1) for lo, hi in window_bounds(layout.num_frames)[1:-1]]
    linear_state = _gather_linear_state(prefix, suffix, alpha, bounds, text_state).to(query_raw.dtype)
    readout = torch.matmul(query, linear_state.transpose(-1, -2))
    readout = readout.permute(0, 2, 1, 3).reshape(frames * per_frame, heads, dim)
    out = torch.zeros_like(query_raw)
    out.index_copy_(0, interior_physical, branch.norm(readout))
    return out


def _active_query_openvdn_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: OpenVDNLayout,
    scale: float,
    active_query_mask: torch.Tensor,
    groups_per_call: int = 4,
) -> torch.Tensor:
    """Run OpenVDN with all K/V but only the selected query rows.

    The mask is global packed-sequence order. Non-target/source rows remain
    active; SpotEdit only clears stable target rows. This is actual partial
    query attention, not post-hoc masking of a fully-computed attention map.
    """
    if active_query_mask.ndim != 1 or active_query_mask.numel() < layout.used_len:
        raise ValueError(
            "active query mask must cover at least the used packed sequence: "
            f"{tuple(active_query_mask.shape)} vs used_len={layout.used_len}"
        )
    active_query_mask = active_query_mask[: layout.used_len].to(
        device=query.device, dtype=torch.bool
    )
    dense_q, _all_k, groups = _softmax_plan(layout, query.device)
    dense_q = dense_q[active_query_mask.index_select(0, dense_q)]
    out = torch.zeros_like(query)

    if dense_q.numel():
        dense_out = _fusion_attention_tnd(
            query.index_select(0, dense_q),
            key[: layout.used_len],
            value[: layout.used_len],
            [dense_q.numel()],
            [layout.used_len],
            scale,
        )
        out.index_copy_(0, dense_q, dense_out)

    active_groups = []
    for q_idx, k_idx in groups:
        q_idx = q_idx[active_query_mask.index_select(0, q_idx)]
        if q_idx.numel():
            active_groups.append((q_idx, k_idx))
    for start in range(0, len(active_groups), groups_per_call):
        batch = active_groups[start : start + groups_per_call]
        q_indices = [item[0] for item in batch]
        k_indices = [item[1] for item in batch]
        q_pack = torch.cat([query.index_select(0, idx) for idx in q_indices])
        k_pack = torch.cat([key.index_select(0, idx) for idx in k_indices])
        v_pack = torch.cat([value.index_select(0, idx) for idx in k_indices])
        q_cu, kv_cu = [], []
        q_total = kv_total = 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        group_out = _fusion_attention_tnd(q_pack, k_pack, v_pack, q_cu, kv_cu, scale)
        out.index_copy_(0, torch.cat(q_indices), group_out)
    return out
