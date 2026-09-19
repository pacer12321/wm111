"""Independent Boolean and frozen-QKV numerical CPU oracles (stdlib only)."""
from __future__ import annotations

import ast
import copy
import math
import random
import unittest
from dataclasses import replace
from types import SimpleNamespace

import build_candidate
from strict_source_layout import (MODE, expand_spans, infer_layout, make_metadata,
                                  strict_plan, validate_arch, validate_raw_request)


def fixture(frames=12, source_hw=(2, 4), target_hw=(2, 6), text=3, ra=1, ta=2,
            source_origin=3.125, target_origin=211.625, pad=7):
    shape_s, shape_t = (frames, *source_hw), (frames, *target_hw)
    meta = make_metadata(task="ref2va", ref_blocks=[dict(kind="video", latent_t=frames,
                        latent_h=source_hw[0], latent_w=source_hw[1], ref_audio_t=ra)],
                        visual_condition_shapes=[shape_s], target_shape=shape_t,
                        text_len=text, audio_t=ta)
    ps, pt = math.prod(source_hw) // 4, math.prod(target_hw) // 4
    ss = text + ra * 2
    se = ss + frames * ps
    ts = se + ta * 2
    used = ts + frames * pt
    coords = [[0.0, 0.0, 0.0] for _ in range(used + pad)]
    source, target = list(range(ss, se)), list(range(ts, used))
    for origin, start, shape in ((source_origin, ss, shape_s), (target_origin, ts, shape_t)):
        time = origin
        cursor = start
        for f in range(frames):
            for h in range(shape[1] // 2):
                for w in range(shape[2] // 2):
                    coords[cursor] = [time, h * 1.7 - 0.35, w * 0.8 + 0.25]
                    cursor += 1
            time += (1, 4, 4, 4, 4)[f % 5] * 5 / 3
    return dict(img_pos=source + target, update_mask=[False] * len(source) + [True] * len(target),
                text_pos=list(range(text)), audio_pos=list(range(text, ss)) + list(range(se, ts)),
                coords=coords, cu_seqlens=[0, used, used + pad], metadata=meta)


def b_oracle(layout, packed_len):
    """Dense Boolean B definition, independent of C span/group constructors."""
    out = [[False] * packed_len for _ in range(packed_len)]
    for q in range(layout.used_len):
        q_is_target = layout.video_start <= q < layout.video_end
        q_frame = (q - layout.video_start) // layout.tokens_per_frame
        for k in range(layout.used_len):
            k_is_target = layout.video_start <= k < layout.video_end
            if not q_is_target or q_frame in (0, layout.num_frames - 1) or not k_is_target:
                out[q][k] = True
            else:
                k_frame = (k - layout.video_start) // layout.tokens_per_frame
                out[q][k] = (k_frame in (0, layout.num_frames - 1)
                             or q_frame // 5 - 1 <= k_frame // 5 <= q_frame // 5 + 1)
    return out


def c_oracle(layout, packed_len):
    mask = b_oracle(layout, packed_len)
    for q in range(layout.video_start, layout.video_end):
        target_frame = (q - layout.video_start) // layout.tokens_per_frame
        for k in range(layout.source_start, layout.source_end):
            source_frame = (k - layout.source_start) // layout.source_tokens_per_frame
            if source_frame != target_frame:
                mask[q][k] = False
    return mask


def plan_mask(layout, packed_len):
    mask = [[False] * packed_len for _ in range(packed_len)]
    dense, groups = strict_plan(layout)
    for q in expand_spans(dense):
        for k in range(layout.used_len):
            mask[q][k] = True
    seen = set(expand_spans(dense))
    for group in groups:
        for q in range(*group.query_span):
            if q in seen:
                raise AssertionError("query written twice")
            seen.add(q)
            for k in expand_spans(group.key_spans):
                mask[q][k] = True
    if seen != set(range(layout.used_len)):
        raise AssertionError("query missing or padding consumed")
    return mask


def attend(q, k, v, mask, scale=0.7):
    result = []
    for qi, allowed in zip(q, mask):
        indices = [i for i, use in enumerate(allowed) if use]
        if not indices:
            result.append([0.0] * len(v[0]))
            continue
        scores = [sum(a * b for a, b in zip(qi, k[i])) * scale for i in indices]
        hi = max(scores)
        weights = [math.exp(s - hi) for s in scores]
        denom = sum(weights)
        result.append([sum(w * v[i][channel] for w, i in zip(weights, indices)) / denom
                       for channel in range(len(v[0]))])
    return result


def grouped_attend(q, k, v, layout):
    """Execute query/key span plan separately; no Boolean mask as input."""
    result = [[0.0] * len(v[0]) for _ in q]
    dense, groups = strict_plan(layout)
    spans = [(expand_spans(dense), list(range(layout.used_len)))]
    spans += [(list(range(*group.query_span)), expand_spans(group.key_spans)) for group in groups]
    for qs, ks in spans:
        if not qs:
            continue
        part = attend([q[i] for i in qs], [k[i] for i in ks], [v[i] for i in ks],
                      [[True] * len(ks) for _ in qs])
        for i, row in zip(qs, part):
            result[i] = row
    return result


class StrictCMetadataTests(unittest.TestCase):
    def test_different_time_origins_noninteger_nonuniform(self):
        data = fixture()
        layout = infer_layout(**data)
        self.assertNotEqual(layout.source_times, layout.target_times)
        self.assertAlmostEqual(layout.source_times[1] - layout.source_times[0], 5 / 3)
        self.assertAlmostEqual(layout.source_times[2] - layout.source_times[1], 20 / 3)
        self.assertEqual(plan_mask(layout, len(data["coords"])), c_oracle(layout, len(data["coords"])))

    def test_source_patch_area_may_differ(self):
        layout = infer_layout(**fixture(source_hw=(4, 6), target_hw=(2, 4)))
        self.assertEqual(layout.source_tokens_per_frame, 6)
        self.assertEqual(layout.tokens_per_frame, 2)
        _, groups = strict_plan(layout)
        for f, group in enumerate(groups):
            visible = [k for k in expand_spans(group.key_spans) if layout.source_start <= k < layout.source_end]
            self.assertEqual(visible, list(range(layout.source_start + 6 * f, layout.source_start + 6 * (f + 1))))

    def test_rejects_fs_mismatch(self):
        data = fixture()
        data["metadata"]["source_shape"] = (11, 2, 4)
        with self.assertRaisesRegex(ValueError, "Fs == target Ft"):
            infer_layout(**data)

    def test_rejects_multiref_even_plausible_grid(self):
        data = fixture()
        data["metadata"]["reference_count"] = 2
        with self.assertRaises(ValueError):
            infer_layout(**data)

    def test_rejects_missing_metadata_or_modes(self):
        for meta in (None, {}, {**fixture()["metadata"], "mode": "B"}, {**fixture()["metadata"], "reference_kind": "image"}):
            data = fixture()
            data["metadata"] = meta
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                infer_layout(**data)

    def test_rejects_bad_patch_and_actual_vae_shape(self):
        data = fixture()
        data["metadata"]["patch_size"] = (2, 2, 2)
        with self.assertRaises(ValueError):
            infer_layout(**data)
        with self.assertRaisesRegex(ValueError, "actual VAE"):
            make_metadata(task="ref2va", ref_blocks=[dict(kind="video", latent_t=12, latent_h=2, latent_w=4, ref_audio_t=1)],
                          visual_condition_shapes=[(11, 2, 4)], target_shape=(12, 2, 4), text_len=1, audio_t=2)

    def test_source_cannot_include_text_or_audio(self):
        for row in (0, 3, 29):
            data = fixture()
            data["img_pos"][0] = row
            with self.subTest(row=row), self.assertRaises(ValueError):
                infer_layout(**data)

    def test_grid_rejects_nonfinite_reorder_gap_and_time_reset(self):
        for mutation in ("nan", "spatial", "time_reset", "duplicate", "partial"):
            data = fixture(source_hw=(4, 4))
            start = data["img_pos"][0]
            if mutation == "nan":
                data["coords"][start][0] = float("nan")
            elif mutation == "spatial":
                data["coords"][start], data["coords"][start + 1] = data["coords"][start + 1], data["coords"][start]
            elif mutation == "time_reset":
                for row in range(start + 4, start + 8):
                    data["coords"][row][0] = data["coords"][start][0] - 1
            elif mutation == "duplicate":
                data["coords"][start + 1] = list(data["coords"][start])
            else:
                del data["img_pos"][0]
                del data["update_mask"][0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                infer_layout(**data)

    def test_rejects_tail_padding_as_source_and_internal_padding(self):
        data = fixture()
        data["img_pos"][0] = data["cu_seqlens"][1]
        with self.assertRaises(ValueError):
            infer_layout(**data)
        data = fixture()
        data["cu_seqlens"] = [0, 3, data["cu_seqlens"][1], len(data["coords"])]
        with self.assertRaises(ValueError):
            infer_layout(**data)

    def test_raw_request_fail_before_encoding(self):
        validate_raw_request("ref2va", "source.mp4")
        validate_raw_request("ref2va", ["source.mp4"])
        for args in (("ref2va", []), ("ref2va", ["a", "b"]), ("ref2va", None),
                     ("t2va", "source.mp4"), ("ref2va", "source.mp4", "image"),
                     ("ref2va", "source.mp4", None, "separate_audio")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_raw_request(*args)

    def test_arch_no_silent_fallback(self):
        good = dict(openvdn_enabled=True, patch_size=(1, 2, 2), openvdn_interior_group_size=0, openvdn_groups_per_call=4)
        validate_arch(SimpleNamespace(**good))
        for name, value in (("openvdn_enabled", False), ("patch_size", (2, 2, 2)), ("openvdn_interior_group_size", 5), ("openvdn_groups_per_call", 0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_arch(SimpleNamespace(**{**good, name: value}))

    def test_corrupted_mask_and_audio_positions_reject(self):
        for name in ("update_mask", "audio_pos", "text_pos"):
            data = fixture()
            data[name][0] = 1 if name == "update_mask" else 99
            with self.subTest(name=name), self.assertRaises(ValueError):
                infer_layout(**data)


class StrictCMaskTests(unittest.TestCase):
    def test_full_mask_exactly_b_minus_wrong_source_edges(self):
        for frames in (3, 7, 12, 17, 37):
            data = fixture(frames=frames)
            layout = infer_layout(**data)
            n = len(data["coords"])
            actual, baseline = plan_mask(layout, n), b_oracle(layout, n)
            self.assertEqual(actual, c_oracle(layout, n))
            removed = sum(int(b and not c) for br, cr in zip(baseline, actual) for b, c in zip(br, cr))
            self.assertEqual(removed, frames * layout.tokens_per_frame * (frames - 1) * layout.source_tokens_per_frame)

    def test_anchors_text_audio_source_queries_unchanged(self):
        data = fixture(frames=17)
        layout = infer_layout(**data)
        b, c = b_oracle(layout, len(data["coords"])), plan_mask(layout, len(data["coords"]))
        for q in range(layout.used_len):
            for k in range(layout.used_len):
                if not (layout.video_start <= q < layout.video_end and layout.source_start <= k < layout.source_end):
                    self.assertEqual(c[q][k], b[q][k])
        for frame in (0, layout.num_frames - 1):
            q = layout.video_start + frame * layout.tokens_per_frame
            self.assertTrue(all(c[q][layout.video_start:layout.video_end]))
            self.assertEqual(sum(c[q][layout.source_start:layout.source_end]), layout.source_tokens_per_frame)

    def test_padding_has_no_edges(self):
        data = fixture(pad=11)
        layout = infer_layout(**data)
        c = plan_mask(layout, len(data["coords"]))
        self.assertFalse(any(any(row) for row in c[layout.used_len:]))
        self.assertFalse(any(any(row[layout.used_len:]) for row in c))

    def test_group_frame_boundary_never_unions_source(self):
        layout = infer_layout(**fixture(frames=12))
        _, groups = strict_plan(layout)
        # Frames 1 and 2 have the same B target window, but disjoint source keys.
        self.assertNotEqual(groups[1].key_spans, groups[2].key_spans)
        self.assertEqual(len(groups), layout.num_frames)

    def test_cache_b_c_b_and_same_target_different_source_start(self):
        one = infer_layout(**fixture(ra=1, ta=3))
        two = infer_layout(**fixture(ra=2, ta=2))
        self.assertEqual(one.video_start, two.video_start)
        self.assertEqual(one.used_len, two.used_len)
        strict_plan.cache_clear()
        b_before = b_oracle(one, one.used_len)
        c_one = strict_plan(one)
        c_two = strict_plan(two)
        self.assertIs(c_one, strict_plan(one))
        self.assertNotEqual(c_one, c_two)
        self.assertEqual(b_before, b_oracle(one, one.used_len))
        with self.assertRaises(ValueError):
            strict_plan(replace(one, mode="B"))
        with self.assertRaises(ValueError):
            strict_plan(SimpleNamespace(**vars(one)))

    def test_same_shape_different_source_area_cache(self):
        one = infer_layout(**fixture(source_hw=(2, 4)))
        two = infer_layout(**fixture(source_hw=(4, 2)))
        self.assertNotEqual(one, two)
        strict_plan.cache_clear()
        strict_plan(one)
        strict_plan(two)
        self.assertEqual(strict_plan.cache_info().misses, 2)

    def test_frozen_qkv_numeric_grouped_dense_match(self):
        data = fixture(frames=12)
        layout = infer_layout(**data)
        n = len(data["coords"])
        rng = random.Random(101)
        q, k, v = ([[rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)] for _ in range(3))
        before = copy.deepcopy((q, k, v))
        dense = attend(q, k, v, c_oracle(layout, n))
        grouped = grouped_attend(q, k, v, layout)
        self.assertLess(max(abs(a - b) for aa, bb in zip(dense, grouped) for a, b in zip(aa, bb)), 1e-12)
        self.assertEqual(before, (q, k, v))

    def test_extreme_wrong_source_value_cannot_leak(self):
        data = fixture(frames=7)
        layout = infer_layout(**data)
        n = len(data["coords"])
        q = k = [[0.0] for _ in range(n)]
        v = [[0.0] for _ in range(n)]
        wrong_key = layout.source_start + layout.source_tokens_per_frame
        v[wrong_key] = [1e20]
        b, c = attend(q, k, v, b_oracle(layout, n)), grouped_attend(q, k, v, layout)
        for frame in (0, 2, layout.num_frames - 1):
            target_query = layout.video_start + frame * layout.tokens_per_frame
            self.assertGreater(b[target_query][0], 0)
            self.assertEqual(c[target_query][0], 0.0)
        self.assertGreater(c[layout.video_start + layout.tokens_per_frame][0], 0)
        self.assertEqual(c[0], b[0])

    def test_emulated_sp_splits_frames_and_empty_target_rank(self):
        data = fixture(frames=12, source_hw=(4, 6), target_hw=(2, 6), pad=13)
        layout = infer_layout(**data)
        n = len(data["coords"])
        c = plan_mask(layout, n)
        # CPU emulation of row partitioning/reassembly, NOT an HCCL test.
        cuts = sorted({0, layout.source_start + 1, layout.source_end - 1,
                       layout.video_start + 1, layout.video_end - 1, n})
        shards = [c[lo:hi] for lo, hi in zip(cuts, cuts[1:])]
        self.assertEqual([row for shard in shards for row in shard], c_oracle(layout, n))
        self.assertTrue(any(hi <= layout.video_start for lo, hi in zip(cuts, cuts[1:])))
        for lo, hi in zip(cuts, cuts[1:]):
            for global_q, row in enumerate(c[lo:hi], lo):
                for global_k, visible in enumerate(row):
                    if visible and layout.video_start <= global_q < layout.video_end and layout.source_start <= global_k < layout.source_end:
                        self.assertEqual((global_q - layout.video_start) // layout.tokens_per_frame,
                                         (global_k - layout.source_start) // layout.source_tokens_per_frame)

    def test_empty_text_no_reference_audio(self):
        data = fixture(text=0, ra=0)
        layout = infer_layout(**data)
        self.assertEqual(plan_mask(layout, len(data["coords"])), c_oracle(layout, len(data["coords"])))


class IntegrationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before, cls.after = build_candidate.render()

    def test_linear_rope_gates_qkv_sp_and_loader_frozen(self):
        build_candidate.verify_frozen(self.before, self.after)
        name = "minimax_h3_transformer.py"
        before, after = ast.parse(self.before[name]), ast.parse(self.after[name])
        old_classes = {n.name: n for n in before.body if isinstance(n, ast.ClassDef)}
        new_classes = {n.name: n for n in after.body if isinstance(n, ast.ClassDef)}
        for classname in old_classes:
            if classname not in {"MiniMaxH3Attention", "MiniMaxH3DiTModel"}:
                self.assertEqual(ast.dump(old_classes[classname]), ast.dump(new_classes[classname]))

    def test_metadata_whitelist_and_static_forward_kwargs(self):
        code = self.after["minimax_h3_transformer.py"]
        self.assertIn('"c_source_layout_metadata",', code)
        pipe = self.after["pipeline_minimax_h3.py"]
        self.assertIn('branch.static_kwargs["c_source_layout_metadata"] = c_source_metadata', pipe)
        upstream = build_candidate.HERE / "upstream/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py"
        loop = upstream.read_text(encoding="utf-8")
        self.assertIn("**self.static_kwargs,", loop)
        self.assertIn("v_video, v_audio = model(**fk)", loop)

    def test_preflight_before_rope_embed_and_sp_collectives(self):
        code = self.after["minimax_h3_transformer.py"]
        start = code.index("def forward(self, **kwargs: Any)")
        forward = code[start:]
        check = forward.index("openvdn_layout = infer_strict_source_layout(")
        self.assertLess(check, forward.index("self.rope(img_position_ids)"))
        self.assertLess(check, forward.index("self._embed("))
        self.assertLess(check, forward.index("self.sp_prepare("))
        pipe = self.after["pipeline_minimax_h3.py"]
        check = pipe.index("validate_raw_request(task, raw_videos")
        self.assertLess(check, pipe.index("prepared_videos = self._prepare_reference_videos(", check))

    def test_bad_source_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            build_candidate.replace_once("different upstream code", "old source", "new")


if __name__ == "__main__":
    unittest.main(verbosity=2)
