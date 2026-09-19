"""OpenVDN hybrid attention port for MiniMax-H3 on Ascend NPU.

The softmax branch expresses the trained chunk-window mask as dense query/key
segments and executes them with npu_fusion_attention's TND varlen interface.
The linear branch preserves OpenVDN's released VDN-solve update rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    import torch_npu
except ImportError:
    torch_npu = None


@dataclass(frozen=True)
class OpenVDNLayout:
    used_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int
    text_len: int

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    def validate(self, packed_len: int) -> None:
        if not 0 < self.used_len <= packed_len:
            raise ValueError("OpenVDN used_len must fit the packed sequence")
        if self.num_frames < 3:
            raise ValueError("OpenVDN requires two anchor frames and at least one interior frame")
        if self.frame_height < 1 or self.frame_width < 1:
            raise ValueError("OpenVDN frame grid must be nonempty")
        if self.tokens_per_frame != self.frame_height * self.frame_width:
            raise ValueError("OpenVDN tokens_per_frame does not match its frame grid")
        if not 0 <= self.video_start < self.video_end <= self.used_len:
            raise ValueError("OpenVDN target video is outside the used sequence")
        if not 0 <= self.text_start <= self.text_start + self.text_len <= self.used_len:
            raise ValueError("OpenVDN text segment must fit the used sequence")
        if max(self.text_start, self.video_start) < min(self.text_start + self.text_len, self.video_end):
            raise ValueError("OpenVDN text and target video segments must not overlap")


def infer_openvdn_layout(
    target_video_pos: torch.Tensor,
    text_pos: torch.Tensor,
    img_position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> OpenVDNLayout:
    """Infer the actual target (t, h, w) patch grid, never a benchmark shape.

    Positions refer to the global packed sequence, including the reference
    prefix. Only one real document plus optional alignment padding is supported.
    CPU metadata validation avoids repeating device scalar synchronizations.
    """
    target = target_video_pos.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    text = text_pos.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    cu = cu_seqlens.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    if cu.numel() not in (2, 3) or int(cu[0]) != 0 or not bool(torch.all(cu[1:] >= cu[:-1])):
        raise ValueError("OpenVDN supports one packed document plus optional trailing padding")
    packed_len = int(cu[-1])
    coords = img_position_ids.detach().reshape(-1, 3)
    if coords.shape[0] != packed_len or target.numel() == 0:
        raise ValueError("OpenVDN needs valid global coordinates and target rows")
    for name, rows in (("target", target), ("text", text)):
        if not rows.numel():
            continue
        if int(rows[0]) < 0 or int(rows[-1]) >= int(cu[1]):
            raise ValueError(f"OpenVDN {name} rows are outside the real document")
        if rows.numel() > 1 and not bool(torch.all(rows[1:] - rows[:-1] == 1)):
            raise ValueError(f"OpenVDN requires ordered contiguous {name} rows")
    grid = coords.index_select(0, target.to(coords.device)).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("OpenVDN target coordinates must be finite")
    times, frame_counts = torch.unique_consecutive(grid[:, 0], return_counts=True)
    if times.numel() < 3 or not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("OpenVDN requires at least three ordered target frames")
    if not bool(torch.all(frame_counts == frame_counts[0])):
        raise ValueError("OpenVDN target frames must have equal patch counts")
    per_frame = int(frame_counts[0])
    spatial = grid[:per_frame, 1:]
    height_axis = torch.unique_consecutive(spatial[:, 0])
    width_axis = torch.unique(spatial[:, 1], sorted=True)
    height, width = height_axis.numel(), width_axis.numel()
    if height * width != per_frame or not bool(torch.all(height_axis[1:] > height_axis[:-1])):
        raise ValueError("OpenVDN target patches must form an ordered rectangular grid")
    hh, ww = torch.meshgrid(height_axis, width_axis, indexing="ij")
    expected_spatial = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
    if not torch.equal(spatial, expected_spatial):
        raise ValueError("OpenVDN target patches must be in frame/height/width row-major order")
    if not torch.equal(grid[:, 1:].view(times.numel(), per_frame, 2), spatial.expand(times.numel(), -1, -1)):
        raise ValueError("OpenVDN target spatial grid must be identical across frames")
    layout = OpenVDNLayout(
        used_len=int(cu[1]), video_start=int(target[0]), num_frames=times.numel(),
        tokens_per_frame=per_frame, frame_height=height, frame_width=width,
        text_start=int(text[0]) if text.numel() else 0, text_len=text.numel(),
    )
    layout.validate(packed_len)
    return layout


def local_frame_sums_counts(
    x_local: torch.Tensor,
    layout: OpenVDNLayout,
    row_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local sufficient statistics for interior-frame mean hidden states.

    The caller SUM-all-reduces both tensors over the sequence group. Only
    O(frames * hidden) statistics are communicated, not all hidden tokens.
    Works when a sequence shard cuts across a frame or contains no target rows.
    """
    if x_local.ndim != 2 or row_start < 0 or layout.num_frames < 3:
        raise ValueError("Invalid local rows or OpenVDN interior-frame layout")
    frames = layout.num_frames - 2
    sums = torch.zeros((frames, x_local.shape[-1]), device=x_local.device, dtype=torch.float32)
    counts = torch.zeros(frames, device=x_local.device, dtype=torch.float32)
    row_end = row_start + x_local.shape[0]
    for inner_frame in range(frames):
        start = layout.video_start + (inner_frame + 1) * layout.tokens_per_frame
        lo, hi = max(start, row_start), min(start + layout.tokens_per_frame, row_end)
        if lo < hi:
            sums[inner_frame] = x_local[lo - row_start : hi - row_start].sum(dim=0, dtype=torch.float32)
            counts[inner_frame] = hi - lo
    return sums, counts


def window_bounds(num_frames: int, radius: int = 1, chunk: int = 5) -> list[tuple[int, int]]:
    return [
        (((frame // chunk) - radius) * chunk, ((frame // chunk) + radius + 1) * chunk - 1)
        for frame in range(num_frames)
    ]


def _frame_indices(layout: OpenVDNLayout, lo: int, hi: int, device: torch.device) -> torch.Tensor:
    start = layout.video_start + lo * layout.tokens_per_frame
    stop = layout.video_start + hi * layout.tokens_per_frame
    return torch.arange(start, stop, device=device, dtype=torch.long)


def _global_indices(layout: OpenVDNLayout, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(0, layout.video_start, device=device, dtype=torch.long),
            torch.arange(layout.video_end, layout.used_len, device=device, dtype=torch.long),
        ]
    )


def _fusion_attention_tnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_cu: list[int],
    kv_cu: list[int],
    scale: float,
) -> torch.Tensor:
    if query.device.type != "npu" or torch_npu is None:
        outputs = []
        q_start = kv_start = 0
        for q_stop, kv_stop in zip(q_cu, kv_cu):
            q = query[q_start:q_stop].transpose(0, 1).unsqueeze(0)
            k = key[kv_start:kv_stop].transpose(0, 1).unsqueeze(0)
            v = value[kv_start:kv_stop].transpose(0, 1).unsqueeze(0)
            outputs.append(
                F.scaled_dot_product_attention(q, k, v, scale=scale)
                .squeeze(0)
                .transpose(0, 1)
            )
            q_start, kv_start = q_stop, kv_stop
        return torch.cat(outputs)
    return torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        head_num=query.shape[1],
        input_layout="TND",
        scale=scale,
        keep_prob=1.0,
        actual_seq_qlen=q_cu,
        actual_seq_kvlen=kv_cu,
        sparse_mode=0,
    )[0]


_SOFTMAX_PLAN_CACHE: dict[tuple, tuple] = {}
_GROUPED_PLAN_CACHE: dict[tuple, tuple] = {}


def _softmax_plan(layout: OpenVDNLayout, device: torch.device) -> tuple:
    key = (
        layout.used_len,
        layout.video_start,
        layout.num_frames,
        layout.tokens_per_frame,
        str(device),
    )
    cached = _SOFTMAX_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    global_idx = _global_indices(layout, device)
    first_anchor = _frame_indices(layout, 0, 1, device)
    last_anchor = _frame_indices(layout, layout.num_frames - 1, layout.num_frames, device)
    anchors = torch.cat((first_anchor, last_anchor))
    dense_q = torch.cat((global_idx, anchors)).sort().values
    all_k = torch.arange(layout.used_len, device=device, dtype=torch.long)
    bounds = window_bounds(layout.num_frames)

    grouped: dict[tuple[int, int], list[int]] = {}
    for frame in range(1, layout.num_frames - 1):
        lo = max(0, bounds[frame][0])
        hi = min(layout.num_frames - 1, bounds[frame][1])
        grouped.setdefault((lo, hi), []).append(frame)

    groups = []
    for (lo, hi), frames in grouped.items():
        q_parts = [_frame_indices(layout, frame, frame + 1, device) for frame in frames]
        q_idx = torch.cat(q_parts)
        window_idx = _frame_indices(layout, lo, hi + 1, device)
        pieces = [global_idx, window_idx]
        if lo > 0:
            pieces.append(first_anchor)
        if hi < layout.num_frames - 1:
            pieces.append(last_anchor)
        k_idx = torch.cat(pieces).sort().values.unique_consecutive()
        groups.append((q_idx, k_idx))

    cached = dense_q, all_k, tuple(groups)
    _SOFTMAX_PLAN_CACHE[key] = cached
    return cached


def openvdn_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: OpenVDNLayout,
    scale: float,
    groups_per_call: int = 4,
) -> torch.Tensor:
    """Exact OpenVDN c5/r1 + first/last row/column anchor softmax."""
    dense_q, _all_k, groups = _softmax_plan(layout, query.device)
    out = torch.zeros_like(query)

    dense_out = _fusion_attention_tnd(
        query.index_select(0, dense_q),
        key[: layout.used_len],
        value[: layout.used_len],
        [dense_q.numel()],
        [layout.used_len],
        scale,
    )
    out.index_copy_(0, dense_q, dense_out)

    for start in range(0, len(groups), groups_per_call):
        batch = groups[start : start + groups_per_call]
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
        del q_pack, k_pack, v_pack, group_out
    return out


def _grouped_anchor_plan(
    layout: OpenVDNLayout,
    device: torch.device,
    group_size: int,
) -> tuple:
    key = (
        layout.used_len,
        layout.video_start,
        layout.num_frames,
        layout.tokens_per_frame,
        group_size,
        str(device),
    )
    cached = _GROUPED_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    inner_frames = layout.num_frames - 2
    if inner_frames <= 0 or inner_frames % group_size:
        raise ValueError(
            f"{inner_frames} interior frames are not divisible by group_size={group_size}"
        )
    global_idx = _global_indices(layout, device)
    first_anchor = _frame_indices(layout, 0, 1, device)
    last_anchor = _frame_indices(layout, layout.num_frames - 1, layout.num_frames, device)
    anchors = torch.cat((first_anchor, last_anchor))
    dense_q = torch.cat((global_idx, anchors)).sort().values
    groups = []
    for lo in range(1, layout.num_frames - 1, group_size):
        hi = lo + group_size
        q_idx = _frame_indices(layout, lo, hi, device)
        k_idx = torch.cat((global_idx, anchors, q_idx)).sort().values
        groups.append((q_idx, k_idx))
    cached = dense_q, torch.arange(layout.used_len, device=device), tuple(groups)
    _GROUPED_PLAN_CACHE[key] = cached
    return cached


def grouped_anchor_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: OpenVDNLayout,
    scale: float,
    group_size: int,
    groups_per_call: int = 5,
    implementation: str = "varlen",
) -> torch.Tensor:
    """Full global/anchor rows plus independent groups of interior video frames."""
    dense_q, _all_k, groups = _grouped_anchor_plan(layout, query.device, group_size)
    out = torch.zeros_like(query)
    dense_out = _fusion_attention_tnd(
        query.index_select(0, dense_q),
        key[: layout.used_len],
        value[: layout.used_len],
        [dense_q.numel()],
        [layout.used_len],
        scale,
    )
    out.index_copy_(0, dense_q, dense_out)
    if implementation == "loop":
        for q_idx, k_idx in groups:
            group_out = _fusion_attention_tnd(
                query.index_select(0, q_idx),
                key.index_select(0, k_idx),
                value.index_select(0, k_idx),
                [q_idx.numel()],
                [k_idx.numel()],
                scale,
            )
            out.index_copy_(0, q_idx, group_out)
        return out

    if implementation == "batch":
        q_indices = [item[0] for item in groups]
        k_indices = [item[1] for item in groups]
        q_batch = torch.stack([query.index_select(0, idx) for idx in q_indices])
        k_batch = torch.stack([key.index_select(0, idx) for idx in k_indices])
        v_batch = torch.stack([value.index_select(0, idx) for idx in k_indices])
        if query.device.type == "npu" and torch_npu is not None:
            group_out = torch_npu.npu_fusion_attention(
                q_batch.contiguous(),
                k_batch.contiguous(),
                v_batch.contiguous(),
                head_num=query.shape[1],
                input_layout="BSND",
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0,
            )[0]
        else:
            group_out = F.scaled_dot_product_attention(
                q_batch.permute(0, 2, 1, 3),
                k_batch.permute(0, 2, 1, 3),
                v_batch.permute(0, 2, 1, 3),
                scale=scale,
            ).permute(0, 2, 1, 3)
        out.index_copy_(0, torch.cat(q_indices), group_out.flatten(0, 1))
        return out

    if implementation != "varlen":
        raise ValueError(f"unknown grouped attention implementation: {implementation}")

    for start in range(0, len(groups), groups_per_call):
        batch = groups[start : start + groups_per_call]
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


class OutputGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bottleneck: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.down = (
            None
            if bottleneck is None
            else nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype)
        )
        self.up = nn.Linear(
            bottleneck or hidden_size,
            num_heads * (head_dim or 1),
            bias=True,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.num_heads, self.head_dim or 1)


class FrameKDAAlpha(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.down = nn.Linear(hidden_size, head_dim, bias=False, dtype=torch.bfloat16)
        self.up = nn.Linear(head_dim, num_heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.bfloat16))
        self.dt_bias = nn.Parameter(torch.empty(num_heads * head_dim, dtype=torch.bfloat16))

    def forward(self, frame_mean_x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(frame_mean_x.float(), self.down.weight.float())
        delta = F.linear(delta, self.up.weight.float()) + self.dt_bias.float()
        scale = torch.exp(self.A_log.float())[:, None]
        delta = delta.view(-1, self.num_heads, self.head_dim)
        return torch.exp(-scale * F.softplus(delta))

    def forward_head_shard(self, frame_mean_x: torch.Tensor, head_start: int, num_heads: int) -> torch.Tensor:
        """Same alpha projection, restricted to this Ulysses rank's heads."""
        if not 0 <= head_start < head_start + num_heads <= self.num_heads:
            raise ValueError("Invalid OpenVDN alpha head shard")
        channels = slice(head_start * self.head_dim, (head_start + num_heads) * self.head_dim)
        delta = F.linear(frame_mean_x.float(), self.down.weight.float())
        delta = F.linear(delta, self.up.weight[channels].float()) + self.dt_bias[channels].float()
        scale = torch.exp(self.A_log[head_start : head_start + num_heads].float())[:, None]
        return torch.exp(-scale * F.softplus(delta.view(-1, num_heads, self.head_dim)))


class LinearAttentionSepConv(nn.Module):
    KERNEL = 5

    def __init__(self, channels: int) -> None:
        super().__init__()
        for name in ("k", "v"):
            setattr(
                self,
                f"{name}_sp",
                nn.Conv2d(
                    channels,
                    channels,
                    self.KERNEL,
                    padding=self.KERNEL // 2,
                    groups=channels,
                    bias=False,
                    dtype=torch.bfloat16,
                ),
            )
            setattr(
                self,
                f"{name}_tm",
                nn.Conv1d(
                    channels,
                    channels,
                    self.KERNEL,
                    padding=self.KERNEL // 2,
                    groups=channels,
                    bias=False,
                    dtype=torch.bfloat16,
                ),
            )

    def apply(
        self,
        proj: str,
        tokens: torch.Tensor,
        num_frames: int,
        frame_size: tuple[int, int],
        head_start: int = 0,
    ) -> torch.Tensor:
        if proj == "q":
            return tokens
        heads, head_dim = tokens.shape[-2:]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        spatial = getattr(self, f"{proj}_sp")
        channel_start = head_start * head_dim
        channel_stop = channel_start + channels
        if channel_start < 0 or channel_stop > spatial.weight.shape[0]:
            raise ValueError("Invalid OpenVDN short-convolution head shard")
        volume = F.conv2d(
            volume,
            spatial.weight[channel_start:channel_stop],
            padding=self.KERNEL // 2,
            groups=channels,
        )
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        weight = getattr(self, f"{proj}_tm").weight[channel_start:channel_stop].squeeze(1).to(x.dtype)
        pad = self.KERNEL // 2
        padded = F.pad(x, (0, 0, 0, 0, pad, pad))
        out = sum(
            padded[offset : offset + num_frames] * weight[:, offset].view(1, 1, -1)
            for offset in range(self.KERNEL)
        )
        return out.reshape(-1, heads, head_dim)


def _activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    out = F.silu(tokens)
    return F.normalize(out, dim=-1, eps=1e-6).to(out.dtype) if l2norm else out


class BranchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(2)
        ms = ms / x.shape[-1]
        return x * torch.rsqrt(ms + self.eps).to(x.dtype) * self.weight.to(x.dtype)


def _frame_statistics(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_bf16 = key.contiguous()
    key_fp32 = key_bf16.float()
    scaled = (key_fp32 * beta.unsqueeze(-1).float()).contiguous()
    value_beta = (value * beta.unsqueeze(-1).to(value.dtype)).contiguous()
    A = torch.matmul(scaled.transpose(-1, -2), key_fp32)
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.matmul(value_beta.transpose(-1, -2), key_bf16).float()
    return A, B


def _vdn_factor_apply(
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = A.float()
    eye = torch.eye(A.shape[-1], device=A.device, dtype=torch.float32).expand_as(A)
    chol = torch.linalg.cholesky(A + eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inv = linv.transpose(-1, -2) @ linv
    transition = alpha.unsqueeze(-1) * inv
    injection = B.float() @ inv
    return transition, injection


def _sana_scaled_factor_apply(
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    tokens_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenVDN's SANA-scaled ablation: first-order scaled delta update."""
    inv_tokens = 1.0 / tokens_per_chunk
    inv_sqrt_tokens = inv_tokens**0.5
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    transition = alpha.unsqueeze(-1) * (eye - inv_tokens * A)
    injection = inv_sqrt_tokens * B
    return transition, injection


def _delta_factor_apply(
    delta_rule: str,
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    tokens_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if delta_rule == "vdn_solve":
        return _vdn_factor_apply(alpha, A, B)
    if delta_rule == "sana_scaled":
        return _sana_scaled_factor_apply(alpha, A, B, tokens_per_chunk)
    raise ValueError(f"unsupported OpenVDN delta rule: {delta_rule}")


def _scan_states(
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    text_state: torch.Tensor | None,
    delta_rule: str = "vdn_solve",
    tokens_per_frame: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    transitions, injections = _delta_factor_apply(
        delta_rule,
        alpha,
        A,
        B,
        tokens_per_frame or 1,
    )
    frames = transitions.shape[0]
    start = torch.zeros_like(injections[0]) if text_state is None else text_state.float()
    prefix = torch.empty((frames, *start.shape), device=start.device, dtype=torch.float32)
    suffix = torch.empty_like(prefix)
    state = start
    for frame in range(frames):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        state = prefix[frame]
    state = start
    for frame in range(frames - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def _gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    text_state: torch.Tensor | None,
) -> torch.Tensor:
    frames = prefix.shape[0]
    device = prefix.device
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    before_idx = last_before.clamp(min=0)
    after_idx = first_after.clamp(max=frames - 1)
    has_before = last_before >= 0
    has_after = first_after < frames
    before = prefix[before_idx]
    after = suffix[after_idx]
    if text_state is not None:
        text_state = text_state.float()
        before = torch.where(has_before.view(-1, 1, 1, 1), before, text_state)
        after = torch.where(has_after.view(-1, 1, 1, 1), after, text_state)

    log_prefix = torch.cat(
        (torch.zeros_like(alpha[:1]), torch.log(alpha.clamp_min(1e-12)).cumsum(0))
    )
    frame_idx = torch.arange(frames, device=device)
    bridge_before = (last_before + 1).clamp(min=0)
    bridge_after = first_after.clamp(max=frames)
    decay_before = torch.exp(log_prefix[frame_idx + 1] - log_prefix[bridge_before])
    decay_after = torch.exp(log_prefix[bridge_after] - log_prefix[frame_idx])
    before = before * decay_before.unsqueeze(2)
    after = after * decay_after.unsqueeze(2)
    if text_state is not None:
        return before + after
    return before * has_before.view(-1, 1, 1, 1) + after * has_after.view(-1, 1, 1, 1)


class BidirectionalLinearBranch(nn.Module):
    TEXT_STATE_SCALE = 0.5

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.short_conv = LinearAttentionSepConv(num_heads * head_dim)
        self.alpha = FrameKDAAlpha(hidden_size, num_heads, head_dim)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False, dtype=torch.bfloat16)
        self.output_gate = OutputGate(
            hidden_size,
            num_heads,
            head_dim,
            bottleneck=head_dim,
            dtype=torch.bfloat16,
        )
        self.norm = BranchRMSNorm(head_dim, eps=1e-6)
        self.delta_rule = "vdn_solve"

    def _features(
        self,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_frames: int,
        frame_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            _activate(self.short_conv.apply(name, tensor, num_frames, frame_size), name != "v")
            for name, tensor in zip(("q", "k", "v"), qkv_raw)
        )

    def _text_state(
        self,
        text_x: torch.Tensor,
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        length = text_x.shape[0]
        key = _activate(text_qkv[1], True).view(1, length, self.num_heads, self.head_dim)
        value = _activate(text_qkv[2], False).view(1, length, self.num_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x)).view(1, length, self.num_heads).permute(0, 2, 1)
        A, B = _frame_statistics(key, value, beta)
        ones = torch.ones(1, self.num_heads, self.head_dim, device=A.device, dtype=torch.float32)
        _, injection = _delta_factor_apply(self.delta_rule, ones, A, B, length)
        return self.TEXT_STATE_SCALE * injection[0]

    def forward_head_shard(
        self,
        qkv_raw_global: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        beta_logits_global: torch.Tensor,
        frame_mean: torch.Tensor,
        layout: OpenVDNLayout,
        head_start: int = 0,
    ) -> torch.Tensor:
        """Global-sequence/local-head VDN-solve, before the local output gate.

        Inputs have already undergone sequence-to-head all-to-all. This rank
        evaluates only its own heads, including the full bidirectional scan;
        no full hidden sequence or duplicate all-head scan is materialized.
        The result keeps global packed rows, with source/text/anchors/padding
        zero. Inverse all-to-all must precede output_gate(local_x).
        """
        if self.delta_rule != "vdn_solve":
            raise ValueError("The initial OpenVDN Ulysses path requires the original vdn_solve rule")
        query_raw, key_raw, value_raw = qkv_raw_global
        if query_raw.ndim != 3 or any(t.shape != query_raw.shape for t in qkv_raw_global):
            raise ValueError("OpenVDN sharded raw Q/K/V must have matching [rows, heads, dim] shapes")
        packed_len, heads, dim = query_raw.shape
        layout.validate(packed_len)
        if dim != self.head_dim or not 0 <= head_start < head_start + heads <= self.num_heads:
            raise ValueError("Invalid OpenVDN linear head shard")
        if beta_logits_global.shape != (packed_len, heads):
            raise ValueError("OpenVDN sharded beta logits must match packed rows and local heads")
        frames, per_frame = layout.num_frames - 2, layout.tokens_per_frame
        if frame_mean.shape != (frames, self.alpha.down.in_features):
            raise ValueError("OpenVDN frame means must cover every interior frame and hidden channel")
        inner = slice(layout.video_start + per_frame, layout.video_end - per_frame)
        frame_size = (layout.frame_height, layout.frame_width)
        features = tuple(
            _activate(self.short_conv.apply(name, raw[inner], frames, frame_size, head_start), name != "v")
            for name, raw in zip(("q", "k", "v"), qkv_raw_global)
        )
        query, key, value = (
            t.view(frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()
            for t in features
        )
        beta = torch.sigmoid(beta_logits_global[inner]).view(frames, per_frame, heads)
        beta = beta.permute(0, 2, 1).contiguous()
        A, B = _frame_statistics(key, value, beta)
        alpha = self.alpha.forward_head_shard(frame_mean, head_start, heads)

        text_slice = slice(layout.text_start, layout.text_start + layout.text_len)
        text_key = _activate(key_raw[text_slice], True).transpose(0, 1).unsqueeze(0)
        text_value = _activate(value_raw[text_slice], False).transpose(0, 1).unsqueeze(0)
        text_beta = torch.sigmoid(beta_logits_global[text_slice]).transpose(0, 1).unsqueeze(0)
        text_A, text_B = _frame_statistics(text_key, text_value, text_beta)
        ones = torch.ones((1, heads, dim), device=query_raw.device, dtype=torch.float32)
        _, text_injection = _delta_factor_apply("vdn_solve", ones, text_A, text_B, layout.text_len)
        text_state = self.TEXT_STATE_SCALE * text_injection[0]
        prefix, suffix = _scan_states(
            alpha, A, B, text_state, delta_rule="vdn_solve", tokens_per_frame=per_frame,
        )
        bounds = [(lo - 1, hi - 1) for lo, hi in window_bounds(layout.num_frames)[1:-1]]
        linear_state = _gather_linear_state(prefix, suffix, alpha, bounds, text_state).to(query_raw.dtype)
        readout = torch.matmul(query, linear_state.transpose(-1, -2))
        readout = readout.permute(0, 2, 1, 3).reshape(frames * per_frame, heads, dim)
        out = torch.zeros_like(query_raw)
        out[inner] = self.norm(readout)
        return out

    def forward(
        self,
        x: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layout: OpenVDNLayout,
    ) -> torch.Tensor:
        frames, per_frame = layout.num_frames, layout.tokens_per_frame
        video = slice(layout.video_start, layout.video_end)
        inner = slice(per_frame, (frames - 1) * per_frame)
        x_video = x[video]
        xv = x_video[inner]
        qkv_video = tuple(t[video][inner] for t in qkv_raw)
        if self.delta_rule == "relu_linear":
            query, key, value = qkv_video
            query = F.relu(query).transpose(0, 1).contiguous()
            key = F.relu(key).transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()
            kv_state = torch.bmm(key.transpose(1, 2), value)
            readout = torch.bmm(query, kv_state).transpose(0, 1).contiguous()
            readout = self.norm(readout)
            readout = (readout * self.output_gate(xv)).reshape(xv.shape[0], -1)
            out = readout.new_zeros(frames * per_frame, readout.shape[-1])
            out[inner] = readout
            return out
        inner_frames = frames - 2
        frame_size = (layout.frame_height, layout.frame_width)
        query, key, value = self._features(qkv_video, inner_frames, frame_size)
        shape = (inner_frames, per_frame, self.num_heads, self.head_dim)
        query = query.view(shape).permute(0, 2, 1, 3).contiguous()
        key = key.view(shape).permute(0, 2, 1, 3).contiguous()
        value = value.view(shape).permute(0, 2, 1, 3).contiguous()
        beta = torch.sigmoid(self.beta_proj(xv)).view(inner_frames, per_frame, self.num_heads)
        beta = beta.permute(0, 2, 1).contiguous()
        A, B = _frame_statistics(key, value, beta)
        frame_mean = xv.view(inner_frames, per_frame, -1).mean(dim=1, dtype=torch.float32)
        alpha = self.alpha(frame_mean)

        text_slice = slice(layout.text_start, layout.text_start + layout.text_len)
        text_state = self._text_state(x[text_slice], tuple(t[text_slice] for t in qkv_raw))
        prefix, suffix = _scan_states(
            alpha,
            A,
            B,
            text_state,
            delta_rule=self.delta_rule,
            tokens_per_frame=per_frame,
        )
        bounds = [(lo - 1, hi - 1) for lo, hi in window_bounds(frames)[1:-1]]
        linear_state = _gather_linear_state(prefix, suffix, alpha, bounds, text_state).to(x.dtype)
        del prefix, suffix, A, B
        readout = torch.matmul(query, linear_state.transpose(-1, -2))
        readout = readout.permute(0, 2, 1, 3).reshape(inner_frames * per_frame, self.num_heads, self.head_dim)
        readout = self.norm(readout)
        readout = (readout * self.output_gate(xv)).reshape(inner_frames * per_frame, -1)
        out = readout.new_zeros(frames * per_frame, readout.shape[-1])
        out[inner] = readout
        return out


__all__ = [
    "BidirectionalLinearBranch",
    "OpenVDNLayout",
    "OutputGate",
    "infer_openvdn_layout",
    "local_frame_sums_counts",
    "grouped_anchor_softmax_attention",
    "openvdn_softmax_attention",
]
