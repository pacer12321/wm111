"""Diagnostic fixtures only: check actual native videos through both input readers.

Does not freeze/reselect the 2K dataset or silently certify editing quality.
Uses the old 512-square geometry for this isolated regression fixture only.
"""
import argparse
import json
from pathlib import Path
import random
import tempfile
import numpy as np

from shared_clip_contract import prepare_pair, load_prepared_clip
from vllm_omni.diffusion.models.minimax_h3.reference_video import (
    load_video_frames, sample_reference_video_frames,
    MINIMAX_H3_FPS, MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in args.inventory.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r["metadata_eligible"]
          and r["source_probe"]["frames"]==r["target_probe"]["frames"]]
    random.Random(4101).shuffle(rows)
    selected=[]
    for orientation in ("landscape","portrait"):
        selected.append(next(r for r in rows if
            ((r["source_probe"]["width"] > r["source_probe"]["height"])
             if orientation=="landscape" else
             (r["source_probe"]["height"] > r["source_probe"]["width"]))))
    manifest=args.output/"diagnostic_only_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in selected)+"\n")
    records=[]
    for row in selected:
        source=Path(row["source_probe"]["path"])
        video_root=source.parents[len(Path(row["source_relpath"]).parts)-1]
        prepare_pair(args.output/"clips",video_root,row,512,512,"center-crop")
        folder,spec=load_prepared_clip(args.output/"clips",row)
        frames=load_video_frames(str(folder/"source.mp4"))
        assert frames.shape==(spec["frame_count"],512,512,3),frames.shape
        with tempfile.TemporaryDirectory(prefix="qwen-common-clip-") as tmp:
            sampled=sample_reference_video_frames(str(folder/"source.mp4"),workdir=tmp)
        indices=[];cursor=0.
        while round(cursor)<len(frames):
            indices.append(round(cursor));cursor+=MINIMAX_H3_FPS/MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
        assert len(indices)==len(sampled["frames"])
        deltas=[np.abs(frames[i].astype(np.int16)-q.astype(np.int16))
                for i,q in zip(indices,sampled["frames"])]
        max_abs=max(int(d.max()) for d in deltas)
        # Distinct FFmpeg-backed decoders can differ by tiny RGB rounding.
        assert max_abs<=2,(row["sample_id"],max_abs)
        record=dict(sample_id=row["sample_id"],frame_count=spec["frame_count"],
            latent_frames_expected=spec["latent_frames"],qwen_frame_indices=indices,
            max_reader_pixel_difference=max_abs,spatial_shape=[512,512],
            original_duration=spec["source_probe"]["duration"],
            prepared_duration=spec["frame_count"]/spec["fps"],
            temporal_padding_frames=0,clip_contract_sha256=spec["clip_contract_sha256"])
        records.append(record);print(json.dumps(record),flush=True)
    receipt=dict(passed=True,test="real_shared_clip_readers",records=records,
        scope="input readers only; not full encoders, visual QC, or final train/infer forward")
    (args.output/"readers_receipt.json").write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
