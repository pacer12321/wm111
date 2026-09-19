"""Independent, stdlib-only D fixtures/oracle. Never imports candidate mask builders."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

WORLD, HEADS, DIM, HIDDEN = 8, 56, 128, 64
CASES = ("prefix_compact", "prefix_wide_source")
MODE = "D_source_hybrid_same_frame_v1"
REJECTIONS = ("source_suffix", "multiref", "fs_mismatch", "prefix_is_not_source",
              "wrong_mode", "nonfinite_source", "source_frame_reorder")


def fixture(case):
    if case == "prefix_compact":
        frames, sh, sw, th, tw, text, ra, ta = 17, 4, 4, 4, 6, 5, 2, 3
    elif case == "prefix_wide_source":
        frames, sh, sw, th, tw, text, ra, ta = 21, 4, 6, 2, 6, 7, 3, 2
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
    """Independent pairwise C and D definitions; no candidate spans/kernel."""
    old_c = [[False] * packed for _ in range(packed)]
    new_d = [[False] * packed for _ in range(packed)]
    for q in range(layout.used_len):
        qs = layout.source_start <= q < layout.source_end
        qt = layout.video_start <= q < layout.video_end
        sfq = (q - layout.source_start) // layout.source_tokens_per_frame
        tfq = (q - layout.video_start) // layout.tokens_per_frame
        for k in range(layout.used_len):
            ks = layout.source_start <= k < layout.source_end
            kt = layout.video_start <= k < layout.video_end
            sfk = (k - layout.source_start) // layout.source_tokens_per_frame
            tfk = (k - layout.video_start) // layout.tokens_per_frame
            keep_c = (not qt or not kt or tfq in (0, layout.num_frames - 1)
                      or tfk in (0, layout.num_frames - 1) or abs(tfq // 5 - tfk // 5) <= 1)
            keep_c = keep_c and not (qt and ks and tfq != sfk)
            old_c[q][k] = keep_c
            if qs:
                new_d[q][k] = (not kt and (not ks or sfq in (0, layout.num_frames - 1)
                                  or sfk in (0, layout.num_frames - 1) or abs(sfq // 5 - sfk // 5) <= 1))
            else:
                new_d[q][k] = keep_c
    return old_c, new_d


def audit_masks(layout, packed):
    c, d = oracle_masks(layout, packed)
    source_removed = 0
    for q in range(packed):
        source_query = layout.source_start <= q < layout.source_end
        for k in range(packed):
            if not source_query and c[q][k] != d[q][k]:
                raise AssertionError("D changed a target or auxiliary query edge")
            if source_query and layout.video_start <= k < layout.video_end and d[q][k]:
                raise AssertionError("D source directly reads target visual rows")
            if d[q][k] and not c[q][k]:
                raise AssertionError("D unexpectedly added a Softmax edge")
            source_removed += c[q][k] and not d[q][k]
    rows = packed // WORLD
    cuts = range(rows, packed, rows)
    source_cuts = [cut for cut in cuts if layout.source_start < cut < layout.source_end
                   and (cut - layout.source_start) % layout.source_tokens_per_frame]
    target_cuts = [cut for cut in cuts if layout.video_start < cut < layout.video_end
                   and (cut - layout.video_start) % layout.tokens_per_frame]
    empty_target = [r for r in range(WORLD) if (r + 1) * rows <= layout.video_start or r * rows >= layout.video_end]
    if not source_cuts or not target_cuts or not empty_target or source_removed <= 0:
        raise AssertionError("Fixture lacks intended source/target SP8 boundary coverage")
    return dict(source_removed_edges=source_removed, source_frame_split_boundaries=source_cuts,
                target_frame_split_boundaries=target_cuts, ranks_without_target=empty_target,
                target_masks_equal_C=True, auxiliary_queries_unchanged=True,
                source_own_anchors=True, source_no_direct_target_visual=True,
                source_independence_claimed=False)


