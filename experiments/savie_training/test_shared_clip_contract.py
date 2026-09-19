import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from shared_clip_contract import (aligned_h3_length, audio_latent_count, clip_directory, digest,
    file_sha, load_prepared_clip, make_spec, merge_encoded_pair, payload_metadata,
    row_identity, validate_payload)


class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class SharedClipTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(sample_id="abc123", source_relpath="s/a.mp4",
            target_relpath="t/a.mp4", instruction="Change the shirt to red.")
        self.probe = dict(width=1280, height=720, frames=101, fps="20", duration="5.05")

    def spec(self):
        spec = make_spec(self.row, self.probe, self.probe, 512, 512, "center-crop")
        spec.update(source_clip_sha256="source", target_clip_sha256="target")
        spec["clip_contract_sha256"] = digest(spec)
        return spec

    def test_native_five_seconds_is_not_legacy_12_or_forced_37(self):
        s = self.spec()
        self.assertEqual((s["frame_count"], s["latent_frames"]), (107, 32))
        self.assertEqual(s["temporal_padding_frames"], 0)
        self.assertLess(Fraction(s["frame_count"], s["fps"]), Fraction(self.probe["duration"]))

    def test_audio_rows_follow_serving_duration_not_four_row_placeholder(self):
        self.assertEqual(audio_latent_count(107, 24), 178)
        self.assertEqual(audio_latent_count(124, 24), 207)
        with self.assertRaises(ValueError):
            audio_latent_count(0, 24)

    def test_exact_vae_boundaries(self):
        for n, expected in [(39,(39,12)), (49,(39,12)), (73,(73,22)),
            (107,(107,32)), (121,(107,32)), (124,(124,37))]:
            self.assertEqual(aligned_h3_length(n), expected)
        with self.assertRaises(ValueError):
            aligned_h3_length(21)

    def test_mismatched_duration_rejected(self):
        p = dict(self.probe, duration="4.8")
        with self.assertRaisesRegex(ValueError,"duration"):
            make_spec(self.row, self.probe, p, 512,512,"center-crop")

    def test_mismatched_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError,"geometry"):
            make_spec(self.row,self.probe,dict(self.probe,width=720),512,512,"resize")

    def test_spatial_policy_is_explicit(self):
        with self.assertRaises(ValueError):
            make_spec(self.row,self.probe,self.probe,512,512,None)
        with self.assertRaises(ValueError):
            make_spec(self.row,self.probe,self.probe,511,512,"resize")

    def test_cache_identity_cannot_be_overwritten_by_prompt(self):
        m = payload_metadata(self.spec())
        video = dict(m, source_video_latents=ShapeOnly((24,32,32,32)),
            target_video_latents=ShapeOnly((24,32,32,32)))
        prompt = dict(m, prompt_embeds="hidden")
        self.assertEqual(merge_encoded_pair(video,prompt)["prompt_embeds"], "hidden")
        for key in m:
            bad = dict(prompt); bad[key] = "different"
            with self.assertRaises(ValueError, msg=key):
                merge_encoded_pair(video,bad)

    def test_reject_old_cache_and_silently_short_latents(self):
        with self.assertRaises(ValueError):
            merge_encoded_pair({}, {})
        m=payload_metadata(self.spec())
        v=dict(m,source_video_latents=ShapeOnly((24,12,32,32)),
            target_video_latents=ShapeOnly((24,12,32,32)))
        with self.assertRaisesRegex(ValueError,"shape"):
            merge_encoded_pair(v,m)

    def test_committed_clip_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            folder=clip_directory(root,self.row["sample_id"]);folder.mkdir()
            (folder/"source.mp4").write_bytes(b"source")
            (folder/"target.mp4").write_bytes(b"target")
            s=self.spec();s.pop("clip_contract_sha256")
            for role in ("source","target"):
                s[f"{role}_clip_sha256"]=file_sha(folder/f"{role}.mp4")
            s["clip_contract_sha256"]=digest(s)
            (folder/"clip.json").write_text(json.dumps(s))
            with self.assertRaises(FileNotFoundError):
                load_prepared_clip(root,self.row)
            (folder/"ready.done").write_text(s["clip_contract_sha256"])
            self.assertEqual(load_prepared_clip(root,self.row)[1],s)
            with self.assertRaises(ValueError):
                load_prepared_clip(root,dict(self.row,instruction="blue"))
            (folder/"source.mp4").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"content"):
                load_prepared_clip(root,self.row)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            clip_directory("/tmp", "../elsewhere")


if __name__ == "__main__":
    unittest.main()
