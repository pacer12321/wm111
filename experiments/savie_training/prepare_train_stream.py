"""Prepare the existing approved 2K selection, without expanding its scope."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
from shared_clip_contract import prepare_pair, file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--original-manifest",type=Path,required=True)
    p.add_argument("--video-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in args.original_manifest.read_text().splitlines() if s.strip()]
    assert len(rows)==2000 and len({r['sample_id'] for r in rows})==2000
    manifest=args.output/'train_2k.jsonl'
    if manifest.exists():
        assert manifest.read_bytes()==args.original_manifest.read_bytes(),'manifest drift'
    else:
        manifest.write_bytes(args.original_manifest.read_bytes())
    (args.output/'selection.json').write_text(json.dumps(dict(
        original_manifest=str(args.original_manifest),sha256=file_sha(manifest),count=len(rows),
        selection_changed=False,forced_length_buckets=False,temporal_padding=False,
        geometry='existing 512x512 center crop, now identical for both encoders',
        note='Existing approved selection; no new claim of broader category coverage'),indent=2))
    def one(index,row):
        if shutil.disk_usage(args.output).free < 20*2**30:
            raise RuntimeError('shared storage free space below 20GiB; do not fill disk')
        started=time.monotonic()
        spec=prepare_pair(args.output/'clips',args.video_root,row,512,512,'center-crop')
        return dict(index=index,sample_id=row['sample_id'],status='prepared',
            latent_frames=spec['latent_frames'],seconds=time.monotonic()-started)
    failed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(one,i,r):i for i,r in enumerate(rows)}
        for job in as_completed(jobs):
            try: result=job.result()
            except Exception as exc:
                result=dict(index=jobs[job],status='rejected',error=str(exc));failed.append(result)
            print(json.dumps(result),flush=True)
    (args.output/'preparation_summary.json').write_text(json.dumps(dict(failed=failed,total=len(rows)),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
