"""Create an isolated B/VDN candidate with read-only Q/K/V capture.

The source is the already validated prepared-offload B candidate.  The patch
only records post-QK-norm/post-RoPE Q/K and V before the existing OpenVDN
Ulysses path; it deliberately leaves B's T-T local+linear computation intact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil


ROOT = Path("/cache/zhonghao/h3")
SOURCE = ROOT / "bd_prepared_20260915_v2/candidates/B"
TARGET = ROOT / "attention_oracle_b_multilayer_code_20260916_v1/candidate_B_capture"
REL = Path("vllm_omni/diffusion/models/minimax_h3")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one insertion point, found {count}")
    return text.replace(old, new, 1)


if TARGET.parent.exists():
    raise RuntimeError(f"isolated destination already exists: {TARGET.parent}")
TARGET.parent.mkdir(parents=True)
shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns("__pycache__", ".git"))

transformer = TARGET / REL / "minimax_h3_transformer.py"
text = transformer.read_text()
text = replace_once(text, "import math\n", "import math\nimport os\n", "os import")
text = replace_once(
    text,
    "from dataclasses import dataclass\n",
    "from dataclasses import dataclass\nfrom pathlib import Path\n",
    "Path import",
)

attention_state = '''        self.openvdn_enabled = bool(arch.openvdn_enabled and enable_openvdn)\n'''
text = replace_once(
    text,
    attention_state,
    '''        # Read-only diagnostic state.  This is intentionally placed before\n        # the existing B/VDN attention branch and does not alter its routing.\n        self._attention_capture_layer = -1\n        self._attention_capture_context: dict[str, Any] | None = None\n        self._attention_capture_done_steps: set[int] = set()\n        self.openvdn_enabled = bool(arch.openvdn_enabled and enable_openvdn)\n''',
    "attention capture state",
)

rope_point = '''        if rope_freqs is not None:\n            q = _apply_rope(q, rope_freqs)\n            k = _apply_rope(k, rope_freqs)\n\n        # The packed layout uses a second document for alignment padding.\n'''
capture_block = '''        if rope_freqs is not None:\n            q = _apply_rope(q, rope_freqs)\n            k = _apply_rope(k, rope_freqs)\n\n        capture = self._attention_capture_context\n        if capture is not None:\n            step = int(capture["step"])\n            if (\n                self._attention_capture_layer in capture["layers"]\n                and step not in self._attention_capture_done_steps\n            ):\n                sp_group = get_sp_group()\n                world = int(sp_group.ulysses_world_size)\n                rank = int(sp_group.ulysses_rank)\n                global_len = int(capture["global_len"])\n                if total * world != global_len:\n                    raise ValueError(\n                        "B attention capture requires equal contiguous SP shards: "\n                        f"local={total} world={world} global={global_len}"\n                    )\n                global_start = rank * total\n                global_stop = global_start + total\n                query_indices = capture["query_indices"].to(q.device)\n                local_query_mask = (\n                    (query_indices >= global_start) & (query_indices < global_stop)\n                )\n                local_query_global = query_indices[local_query_mask]\n                local_query_rows = local_query_global - global_start\n                used_stop = min(global_stop, int(capture["used_len"]))\n                local_key_count = max(0, used_stop - global_start)\n                local_key_global = torch.arange(\n                    global_start, used_stop, device=q.device, dtype=torch.long\n                )\n                output_dir = Path(capture["output_dir"]) / f"step_{step:02d}"\n                output_dir.mkdir(parents=True, exist_ok=True)\n                rank_path = output_dir / (\n                    f"layer_{self._attention_capture_layer:02d}_rank_{rank}.pt"\n                )\n                torch.save(\n                    {\n                        "schema": "h3_b_vdn_local_post_rope_qkv_v1",\n                        "attention_mode": "ref2va_b_vdn_tt_ts_dense",\n                        "layer": self._attention_capture_layer,\n                        "step": step,\n                        "rank": rank,\n                        "world_size": world,\n                        "num_heads": q.shape[1],\n                        "head_dim": q.shape[2],\n                        "softmax_scale": self.softmax_scale,\n                        "q_selected": q.index_select(0, local_query_rows).detach().cpu(),\n                        "query_indices": local_query_global.detach().cpu(),\n                        "k_local": k[:local_key_count].detach().cpu(),\n                        "v_local": v[:local_key_count].detach().cpu(),\n                        "key_indices": local_key_global.detach().cpu(),\n                        "all_query_indices": capture["query_indices"].cpu(),\n                        "all_query_coords": capture["query_coords"].cpu(),\n                        "source_positions": capture["source_positions"].cpu(),\n                        "source_coords": capture["source_coords"].cpu(),\n                        "target_positions": capture["target_positions"].cpu(),\n                        "target_coords": capture["target_coords"].cpu(),\n                        "layout": capture["layout"],\n                    },\n                    rank_path,\n                )\n                logger.info(\n                    "B_VDN_ATTENTION_QKV_CAPTURED step=%d layer=%d rank=%d "\n                    "queries=%d keys=%d path=%s",\n                    step,\n                    self._attention_capture_layer,\n                    rank,\n                    local_query_global.numel(),\n                    local_key_count,\n                    rank_path,\n                )\n                self._attention_capture_done_steps.add(step)\n\n        # The packed layout uses a second document for alignment padding.\n'''
text = replace_once(text, rope_point, capture_block, "attention QKV capture")

blocks_point = '''        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch, quant_config) for _ in range(arch.num_layers)])\n        self.sp_prepare = MiniMaxH3SPPrepare()\n'''
text = replace_once(
    text,
    blocks_point,
    '''        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch, quant_config) for _ in range(arch.num_layers)])\n        for layer_index, block in enumerate(self.blocks):\n            block.attn._attention_capture_layer = layer_index\n        self.sp_prepare = MiniMaxH3SPPrepare()\n''',
    "layer numbering",
)
text = replace_once(
    text,
    '''        self.final_layer = MiniMaxH3FinalLayer(arch, quant_config)\n        self._mark_missing_params_required()\n''',
    '''        self.final_layer = MiniMaxH3FinalLayer(arch, quant_config)\n        self._attention_capture_step = 0\n        self._mark_missing_params_required()\n''',
    "step counter",
)

loop_point = '''        hidden, block_rope, block_combined = self.sp_prepare(\n            hidden,\n            block_rope,\n            block_combined,\n        )\n        for block in self.blocks:\n            hidden = block(\n                hidden,\n                t_emb=t_emb,\n                combined_indices=block_combined,\n                rope_freqs=block_rope,\n                cu_seqlens=cu_seqlens,\n                max_seqlen=max_seqlen,\n                openvdn_layout=openvdn_layout,\n            )\n        hidden = self.sp_gather(hidden)\n'''
capture_loop = '''        hidden, block_rope, block_combined = self.sp_prepare(\n            hidden,\n            block_rope,\n            block_combined,\n        )\n        capture_dir = os.environ.get("ZHONGHAO_H3_ATTENTION_CAPTURE_DIR")\n        capture_steps = {\n            int(value)\n            for value in os.environ.get(\n                "ZHONGHAO_H3_ATTENTION_CAPTURE_STEPS", "0,48"\n            ).split(",")\n            if value.strip()\n        }\n        capture_layers = {\n            int(value)\n            for value in os.environ.get(\n                "ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS", "24"\n            ).split(",")\n            if value.strip()\n        }\n        capture_context = None\n        if capture_dir and self._attention_capture_step in capture_steps:\n            used_len = int(cu_seqlens[1].item())\n            all_img_coords = img_position_ids.reshape(-1, 3)\n            update = update_mask.view(-1).to(device=device, dtype=torch.bool)\n            source_visual = torch.nonzero(~update, as_tuple=False).view(-1)\n            target_visual = torch.nonzero(update, as_tuple=False).view(-1)\n            source_positions = img_pos.to(device).index_select(0, source_visual)\n            target_positions = img_pos.to(device).index_select(0, target_visual)\n            source_positions = source_positions[source_positions < used_len]\n            target_positions = target_positions[target_positions < used_len]\n            source_coords = all_img_coords.index_select(0, source_positions)\n            target_coords = all_img_coords.index_select(0, target_positions)\n            target_times = torch.unique(target_coords[:, 0], sorted=True)\n            target_ys = torch.unique(target_coords[:, 1], sorted=True)\n            target_xs = torch.unique(target_coords[:, 2], sorted=True)\n            frames = int(target_times.numel())\n            height = int(target_ys.numel())\n            width = int(target_xs.numel())\n            per_frame = height * width\n            if target_positions.numel() != frames * per_frame:\n                raise ValueError(\n                    "B attention capture cannot form target video grid: "\n                    f"positions={target_positions.numel()} grid={(frames, height, width)}"\n                )\n            source_times = torch.unique(source_coords[:, 0], sorted=True)\n            source_ys = torch.unique(source_coords[:, 1], sorted=True)\n            source_xs = torch.unique(source_coords[:, 2], sorted=True)\n            if (\n                source_positions.numel() != frames * per_frame\n                or source_times.numel() != frames\n                or source_ys.numel() != height\n                or source_xs.numel() != width\n            ):\n                raise ValueError(\n                    "B attention capture source/target grids differ: "\n                    f"source={source_positions.numel()} target={target_positions.numel()}"\n                )\n            target_grid = target_positions.view(frames, height, width)\n            ys = torch.linspace(0, height - 1, steps=4, device=device).round().long().unique()\n            xs = torch.linspace(0, width - 1, steps=4, device=device).round().long().unique()\n            yy, xx = torch.meshgrid(ys, xs, indexing="ij")\n            sampled = target_grid[:, yy.reshape(-1), xx.reshape(-1)].reshape(-1)\n            capture_context = {\n                "output_dir": capture_dir,\n                "layers": capture_layers,\n                "step": self._attention_capture_step,\n                "global_len": seq_len,\n                "used_len": used_len,\n                "query_indices": sampled.detach(),\n                "query_coords": all_img_coords.index_select(0, sampled).detach(),\n                "source_positions": source_positions.detach(),\n                "source_coords": source_coords.detach(),\n                "target_positions": target_positions.detach(),\n                "target_coords": target_coords.detach(),\n                "layout": {\n                    "used_len": used_len,\n                    "video_start": int(target_positions[0].item()),\n                    "num_frames": frames,\n                    "tokens_per_frame": per_frame,\n                    "frame_height": height,\n                    "frame_width": width,\n                    "text_start": int(text_pos[0].item()),\n                    "text_len": int(text_pos.numel()),\n                    "vdn_window_chunk": 5,\n                    "vdn_window_radius": 1,\n                },\n            }\n        for block in self.blocks:\n            block.attn._attention_capture_context = capture_context\n            hidden = block(\n                hidden,\n                t_emb=t_emb,\n                combined_indices=block_combined,\n                rope_freqs=block_rope,\n                cu_seqlens=cu_seqlens,\n                max_seqlen=max_seqlen,\n                openvdn_layout=openvdn_layout,\n            )\n            block.attn._attention_capture_context = None\n        hidden = self.sp_gather(hidden)\n'''
text = replace_once(text, loop_point, capture_loop, "model capture loop")
text = replace_once(
    text,
    '''        return video_logits, audio_logits\n\n\nEntryClass = MiniMaxH3DiTModel\n''',
    '''        self._attention_capture_step += 1\n        return video_logits, audio_logits\n\n\nEntryClass = MiniMaxH3DiTModel\n''',
    "step increment",
)

transformer.write_text(text)
print(f"TARGET={TARGET}")
print(f"PIPELINE_SHA256={sha(TARGET / REL / 'pipeline_minimax_h3.py')}")
print(f"TRANSFORMER_SHA256={sha(transformer)}")
