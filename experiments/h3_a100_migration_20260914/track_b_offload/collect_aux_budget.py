"""Read-only auxiliary component header audit for CPU/GPU memory budgeting."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("aux_budget_headers.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    remote = r'''
import hashlib, json, math, re, struct
from pathlib import Path
root=Path('/cache/zhonghao/h3/models/MiniMax-H3/Ref2VA')
result={'header_bytes_read':0,'files':[],'components':{},'text_encoder_config':json.loads((root/'text_encoder/config.json').read_text())}
for component in ('text_encoder','video_vae','audio_vae'):
    records=[]
    for path in sorted((root/component).rglob('*.safetensors')):
        with path.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            if not 2<=size<=16*1024*1024: raise ValueError('invalid header length')
            raw=stream.read(size)
        if len(raw)!=size: raise ValueError('truncated header')
        result['header_bytes_read']+=8+size
        header=json.loads(raw)
        result['files'].append({'component':component,'path':str(path),'file_size':path.stat().st_size,'header_sha256':hashlib.sha256(raw).hexdigest()})
        for name,info in header.items():
            if name=='__metadata__':continue
            records.append({'name':name,'shape':info['shape'],'dtype':info['dtype'],'numel':math.prod(info['shape']),'file':str(path)})
    result['components'][component]=records
print(json.dumps(result,separators=(',',':')))
'''
    command = ["ssh", "-F", "C:/Users/DZH/.ssh/config", "-p", "30674", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=8", "dev-modelarts-cnnorth9.huaweicloud.com",
               "/cache/zhonghao/h3/env_cuda_v1/bin/python -c '" + remote.replace("'", "'\"'\"'") + "'"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    record = json.loads(result.stdout)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(dict(files=len(record["files"]), header_bytes_read=record["header_bytes_read"],
                         counts={k:len(v) for k,v in record["components"].items()})))


if __name__ == "__main__":
    main()
