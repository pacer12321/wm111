"""Read only safetensors headers; never load tensors or allocate GPU memory."""
import json
import math
from pathlib import Path
import struct

root = Path('/cache/zhonghao/h3/models')
groups = {
    'h3_dit': sorted((root / 'MiniMax-H3/Ref2VA/transformer').glob('*.safetensors')),
    'vdn_linear': [root / 'OpenVDN-vdn-minimax-h3/stage-b-step-2000/linear_branch/model.safetensors'],
    'vdn_lora': [root / 'OpenVDN-vdn-minimax-h3/stage-b-step-2000/adapters/default/adapter_model.safetensors'],
}
for group, paths in groups.items():
    count = 0
    tensors = 0
    dtypes = {}
    for path in paths:
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            assert size < 100_000_000
            header = json.loads(stream.read(size))
        for name, value in header.items():
            if name == '__metadata__':
                continue
            elements = math.prod(value['shape'])
            count += elements
            tensors += 1
            dtypes[value['dtype']] = dtypes.get(value['dtype'], 0) + elements
    print(json.dumps(dict(group=group, files=len(paths), tensors=tensors,
                          stored_elements=count, billions=count / 1e9, dtypes=dtypes)), flush=True)
