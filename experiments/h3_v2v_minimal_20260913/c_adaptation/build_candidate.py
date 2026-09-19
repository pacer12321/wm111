"""Mechanically render isolated C copies; never edits B or any remote path.

Run --check for an in-memory/compile check; --output creates a NEW direct child
directory of c_adaptation. Exact replacement gates fail if the B source drifts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "b_adaptation" / "patched"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"C patch anchor must occur exactly once: {old[:100]!r}")
    return text.replace(old, new, 1)


def render():
    files = {name: (BASE / name).read_text(encoding="utf-8") for name in
             ("openvdn_npu.py", "openvdn_checkpoint.py", "minimax_h3_transformer.py", "pipeline_minimax_h3.py")}
    before = dict(files)
    name = "minimax_h3_transformer.py"
    code = files[name]
    code = replace_once(code, "if TYPE_CHECKING:\n", "from .strict_source_attention import infer_strict_source_layout, strict_source_softmax_attention\nfrom .strict_source_layout import validate_arch\n\nif TYPE_CHECKING:\n")
    code = replace_once(code, '        "refiner_packed_seq_params",\n', '        "refiner_packed_seq_params",\n        "c_source_layout_metadata",\n')
    old = '''        if self.openvdn_interior_group_size:
            return grouped_anchor_softmax_attention(
                q, k, v, layout, self.softmax_scale,
                group_size=self.openvdn_interior_group_size,
                groups_per_call=self.openvdn_groups_per_call,
                implementation=self.openvdn_group_impl,
            )
        return openvdn_softmax_attention(
'''
    new = '''        if self.openvdn_interior_group_size:
            raise ValueError("C supports only the unchanged B c5/r1 mask")
        return strict_source_softmax_attention(
'''
    code = replace_once(code, old, new)
    old = '''        # Compute RoPE frequencies over the full packed sequence.
'''
    new = '''        # C metadata rejection precedes refiner and sequence collectives.
        validate_arch(self.arch)
        if int(cu_seqlens[-1]) != seq_len:
            raise ValueError("C packed metadata must cover exactly the full x sequence")
        openvdn_layout = infer_strict_source_layout(
            img_pos=img_pos, update_mask=update_mask, text_pos=text_pos, audio_pos=audio_pos,
            img_position_ids=img_position_ids, cu_seqlens=cu_seqlens,
            metadata=_required_kwarg(kwargs, "c_source_layout_metadata"),
        )
        openvdn_layout.validate(seq_len)
        # Compute RoPE frequencies over the full packed sequence.
'''
    code = replace_once(code, old, new)
    old = '''        openvdn_layout = None
        if self.arch.openvdn_enabled:
            update = update_mask.view(-1).to(device=device, dtype=torch.bool)
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
'''
    code = replace_once(code, old, "        # C layout was already validated before embedding.\n")
    files[name] = code

    name = "pipeline_minimax_h3.py"
    code = files[name]
    code = replace_once(code, "from .encoder import MiniMaxH3Qwen3VLEncoder\n", "from .encoder import MiniMaxH3Qwen3VLEncoder\nfrom .strict_source_layout import make_metadata, validate_arch, validate_raw_request\n")
    code = replace_once(code, '        raw_videos = multi_modal_data.get("video")\n', '''        raw_videos = multi_modal_data.get("video")
        validate_arch(self.transformer.arch)
        validate_raw_request(task, raw_videos, raw_image, multi_modal_data.get("audio"))
''')
    code = replace_once(code, "        initial_video, initial_audio = self._initial_noise(\n", '''        validate_arch(self.transformer.arch)
        c_source_metadata = make_metadata(
            task=task, ref_blocks=ref_blocks, visual_condition_shapes=visual_condition_shapes,
            target_shape=(latent_t, latent_h, latent_w),
            text_len=int(text_embeddings.shape[0]), audio_t=audio_t,
        )
        initial_video, initial_audio = self._initial_noise(
''')
    old = '''        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )
'''
    new = old + '''        # DenoiseBranch.forward_kwargs preserves static_kwargs each step.
        branch.static_kwargs["c_source_layout_metadata"] = c_source_metadata
        logger.info("C strict same-frame source metadata: %s", c_source_metadata)
'''
    code = replace_once(code, old, new)
    files[name] = code
    for name in ("strict_source_layout.py", "strict_source_attention.py"):
        files[name] = (HERE / name).read_text(encoding="utf-8")
    for name, code in files.items():
        compile(code, name, "exec")
    verify_frozen(before, files)
    return before, files


def _methods(code, classname):
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == classname)
    return {n.name: ast.dump(n, include_attributes=False) for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def verify_frozen(before, after):
    for name in ("openvdn_npu.py", "openvdn_checkpoint.py"):
        if before[name] != after[name]:
            raise AssertionError(f"C must keep {name} byte-text identical to B")
    name = "minimax_h3_transformer.py"
    for cls, allowed in (("MiniMaxH3Attention", {"_openvdn_softmax"}), ("MiniMaxH3DiTModel", {"forward"})):
        old, new = _methods(before[name], cls), _methods(after[name], cls)
        if old.keys() != new.keys() or any(old[key] != new[key] for key in old if key not in allowed):
            raise AssertionError(f"C changed protected {cls} math/weights/QKV/gates/SP methods")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check == (args.output is not None):
        parser.error("choose exactly one of --check or --output")
    before, files = render()
    sha = lambda data: hashlib.sha256(data.encode()).hexdigest()
    manifest = {"mode": "C_strict_visual_source_same_latent_frame_v1", "base_sha256": {n: sha(c) for n, c in before.items()},
                "candidate_sha256": {n: sha(c) for n, c in files.items()}, "runtime_verified": False}
    if args.output is not None:
        out = args.output.resolve()
        if out.parent != HERE or out == BASE or out.exists() or out.is_symlink():
            raise ValueError("C output must be a new direct child of c_adaptation; never overwrite")
        out.mkdir(exist_ok=False)
        for name, code in files.items():
            with (out / name).open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(code)
        with (out / "candidate_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
