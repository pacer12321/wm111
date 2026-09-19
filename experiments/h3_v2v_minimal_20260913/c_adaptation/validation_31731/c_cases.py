"""Independent, stdlib-only C fixtures/oracle. Never imports the C plan."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

WORLD, HEADS, DIM, HIDDEN = 8, 56, 128, 64
CASES = ("prefix_compact", "prefix_wide_source")
MODE = "C_strict_visual_source_same_latent_frame_v1"
REJECTIONS = ("source_suffix", "multiref", "fs_mismatch", "prefix_is_not_source",
              "wrong_mode", "nonfinite_source", "source_frame_reorder")


def fixture(case):
    if case == "prefix_compact":
        frames, sh, sw, th, tw, text, ra, ta = 17, 4, 4, 4, 6, 5, 2, 3
    elif case == "prefix_wide_source":
        frames, sh, sw, th, tw, text, ra, ta = 12, 4, 6, 2, 6, 7, 3, 2
    else:
        raise ValueError("Unknown fixed C fixture")
    ps, pt = sh * sw // 4, th * tw // 4
    ss = text + 2 * ra
    se = ss + frames * ps
    ts = se + 2 * ta
    used = ts + frames * pt
    packed = math.ceil((used + 11) / WORLD) * WORLD
    coords = [[0.0, 0.0, 0.0] for _ in range(packed)]
    for start, h, w, origin in ((ss, sh, sw, 3.125), (ts, th, tw, 211.625)):
        row, time = start, origin
        for f in range(frames):
            for y in range(h // 2):
                for x in range(w // 2):
                    coords[row] = [time, y * 1.7 - 0.35, x * 0.8 + 0.25]
                    row += 1
            time += (1, 4, 4, 4, 4)[f % 5] * 5 / 3
    meta = dict(mode=MODE, reference_count=1, reference_kind="video", patch_size=(1, 2, 2),
                source_shape=(frames, sh, sw), target_shape=(frames, th, tw),
                text_len=text, source_audio_t=ra, target_audio_t=ta)
    return dict(img_pos=list(range(ss, se)) + list(range(ts, used)),
                update_mask=[False] * (se - ss) + [True] * (used - ts),
                text_pos=list(range(text)), audio_pos=list(range(text, ss)) + list(range(se, ts)),
                coords=coords, cu_seqlens=[0, used, packed], metadata=meta)


def invalid_fixtures(case):
    good = fixture(case)
    result = {}
    for name in REJECTIONS:
        data = copy.deepcopy(good)
        meta = data["metadata"]
        source = [i for i, u in zip(data["img_pos"], data["update_mask"]) if not u]
        target = [i for i, u in zip(data["img_pos"], data["update_mask"]) if u]
        if name == "source_suffix":
            # A complete but unsupported target-then-source arrangement;
            # original C inference must reject, not silently call B.
            # Move the FULL token partition, including target audio, so the
            # rejection is not explained by visual/audio overlap or gaps.
            used = data["cu_seqlens"][1]
            rows = list(range(source[0])) + list(range(source[-1] + 1, target[0])) + target + source
            if sorted(rows) != list(range(used)):
                raise AssertionError("Suffix fixture permutation is incomplete")
            remap = {old: n for n, old in enumerate(rows)}
            old_coords = copy.deepcopy(data["coords"])
            for old, new in remap.items():
                data["coords"][new] = old_coords[old]
            data["img_pos"] = [remap[x] for x in source + target]
            data["audio_pos"] = [remap[x] for x in data["audio_pos"]]
            data["text_pos"] = [remap[x] for x in data["text_pos"]]
        elif name == "multiref":
            meta["reference_count"] = 2
        elif name == "fs_mismatch":
            meta["source_shape"] = (meta["source_shape"][0] - 1, *meta["source_shape"][1:])
        elif name == "prefix_is_not_source":
            data["img_pos"][0] = data["text_pos"][0]
        elif name == "wrong_mode":
            meta["mode"] = "B"
        elif name == "nonfinite_source":
            data["coords"][source[0]][0] = float("nan")
        elif name == "source_frame_reorder":
            ps = meta["source_shape"][1] * meta["source_shape"][2] // 4
            for row in source[ps:2 * ps]:
                data["coords"][row][0] = data["coords"][source[0]][0] - 1
        result[name] = data
    return result


def geometry(case):
    """Expected block geometry derived only from the fixture, not C inference."""
    data = fixture(case)
    meta = data["metadata"]
    frames, sh, sw = meta["source_shape"]
    _, th, tw = meta["target_shape"]
    ss = meta["text_len"] + 2 * meta["source_audio_t"]
    se = ss + frames * sh * sw // 4
    ts = se + 2 * meta["target_audio_t"]
    return SimpleNamespace(used_len=data["cu_seqlens"][1], source_start=ss, source_end=se,
        video_start=ts, video_end=data["cu_seqlens"][1], num_frames=frames,
        source_tokens_per_frame=sh * sw // 4, tokens_per_frame=th * tw // 4,
        source_frame_height=sh // 2, source_frame_width=sw // 2,
        frame_height=th // 2, frame_width=tw // 2, text_start=0, text_len=meta["text_len"])


def oracle_masks(layout, packed):
    """Direct pairwise B definition followed by exactly the C edge deletion.

    No strict_plan, group spans, device cache or runtime attention is consulted.
    """
    baseline = [[False] * packed for _ in range(packed)]
    strict = [[False] * packed for _ in range(packed)]
    for q in range(layout.used_len):
        qt = layout.video_start <= q < layout.video_end
        qf = (q - layout.video_start) // layout.tokens_per_frame
        for k in range(layout.used_len):
            kt = layout.video_start <= k < layout.video_end
            kf = (k - layout.video_start) // layout.tokens_per_frame
            keep = (not qt or not kt or qf in (0, layout.num_frames - 1)
                    or kf in (0, layout.num_frames - 1) or abs(qf // 5 - kf // 5) <= 1)
            baseline[q][k] = keep
            ks = layout.source_start <= k < layout.source_end
            sf = (k - layout.source_start) // layout.source_tokens_per_frame
            strict[q][k] = keep and not (qt and ks and qf != sf)
    return baseline, strict


def audit_masks(layout, packed):
    b, c = oracle_masks(layout, packed)
    removed = 0
    for q in range(packed):
        for k in range(packed):
            qt = layout.video_start <= q < layout.video_end
            ks = layout.source_start <= k < layout.source_end
            wrong = (qt and ks and (q - layout.video_start) // layout.tokens_per_frame
                     != (k - layout.source_start) // layout.source_tokens_per_frame)
            if c[q][k] != (b[q][k] and not wrong):
                raise AssertionError("Oracle changed a non-C edge")
            removed += b[q][k] and not c[q][k]
    expected = layout.num_frames * layout.tokens_per_frame * (layout.num_frames - 1) * layout.source_tokens_per_frame
    if removed != expected:
        raise AssertionError("Wrong source edge count")
    rows = packed // WORLD
    cuts = range(rows, packed, rows)
    source_cuts = [cut for cut in cuts if layout.source_start < cut < layout.source_end
                   and (cut - layout.source_start) % layout.source_tokens_per_frame]
    target_cuts = [cut for cut in cuts if layout.video_start < cut < layout.video_end
                   and (cut - layout.video_start) % layout.tokens_per_frame]
    empty_target_ranks = [r for r in range(WORLD) if (r + 1) * rows <= layout.video_start or r * rows >= layout.video_end]
    if not source_cuts or not target_cuts or not empty_target_ranks:
        raise AssertionError("Fixture lacks intended SP8 boundary coverage")
    return dict(removed_edges=removed, source_frame_split_boundaries=source_cuts,
                target_frame_split_boundaries=target_cuts, ranks_without_target=empty_target_ranks,
                anchors_strict_source=True, nonvisual_audio_text_source_queries_unchanged=True)
