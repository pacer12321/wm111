"""Preserve only this failed owned attempt, ready for corrected controller."""
import fcntl
import json
from pathlib import Path
import socket
import hashlib

ROOT=Path('/cache/zhonghao/h3')
RUN=ROOT/'bd_prepared_20260915_v2'
KIT=ROOT/'track_b_validation_20260915'
RUNTIME=('h3_prepared_integration.py','prepared_shard_hook.py','torch_streaming_shards.py',
         'streaming_shards.py','manifest_builder.py','meta_model_metadata.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname()=='os-node-created-mgf6h'
    lock=(ROOT/'a100_v1/queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((RUN/'queue_status.json').read_text())
    assert status['pid']==1730318 and status['status']=='failed'
    assert 'D 2-step HTTP request failed' in status['error']
    assert not Path('/proc/1730318').exists() and not Path('/proc/1730319').exists()
    target=RUN/'history/inference_tensor_v2'
    assert target.resolve().is_relative_to(RUN.resolve()) and not target.exists()
    target.mkdir(parents=True)
    for name in ('results/D','smoke','smoke_dispatch.log','queue_status.json',
                 'continuation.json','continuation.log'):
        source=RUN/name
        assert source.resolve().is_relative_to(RUN.resolve())
        if source.exists(): source.rename(target/name.replace('/','_'))
    previous=json.loads((RUN/'storage_hashes.json').read_text())
    current={name:sha(KIT/name) for name in RUNTIME}
    changed=sorted(name for name in RUNTIME if previous[name]!=current[name])
    assert changed==['h3_prepared_integration.py','prepared_shard_hook.py','torch_streaming_shards.py']
    (target/'runtime_change.json').write_text(json.dumps({'before':previous,'after':current,'changed':changed},indent=2))
    (RUN/'storage_hashes.json').write_text(json.dumps(current,indent=2))
    print(json.dumps({'archived':str(target),'deleted_files':0}))


if __name__=='__main__':main()
