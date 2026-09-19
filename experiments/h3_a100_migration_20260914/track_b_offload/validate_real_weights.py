"""CPU-only exhaustive tensor-value gate; not DLO/GPU or model-order validation."""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open
from streaming_shards import RangeReader, TensorRef, TensorPlan, LoRA
from torch_streaming_shards import build_torch_block_shard
from test_torch_streaming_shards import original_production_merge


def original_reorder():
    path = Path(os.environ['TRACK_B_PRODUCTION_MERGE_SOURCE']).with_name('minimax_h3_transformer.py')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'a3dc6a0189fdd8f3a9e2aaeddf9544df1503a2f74af287b74b7e975dd7ce717c'
    tree = ast.parse(raw.decode())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_reorder_grouped_qkv_to_qkv']
    assert len(nodes) == 1
    ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),str(path),'exec'),ns)
    return ns[nodes[0].name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('real_manifest.json'))
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    readers = {}
    for f in manifest['file_provenance']:
        path = Path(f['path'])
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8),'little')
            assert size == f['header_size']
            assert hashlib.sha256(stream.read(size)).hexdigest() == f['header_sha256']
        assert path.stat().st_size == f['file_size']
        readers[str(path)] = RangeReader(path)
    def ref(item):
        return TensorRef(readers[item['source_file']],item['source_key'])
    merge, reorder = original_production_merge(), original_reorder()
    begin = time.time()
    result = dict(status='running',scope='all real tensors, two tensor-wise shards; not runtime block ordering or GPU',
                  torch_version=torch.__version__,threads=4,bitwise_required=True,verified=0,lora_pairs=0,qkv=0)
    (args.output/'status.json').write_text(json.dumps(result,indent=2))
    with (args.output/'tensor_results.jsonl').open('x') as log:
        try:
            for index,item in enumerate(manifest['plans']):
                start = time.time()
                with safe_open(item['source_file'],framework='pt',device='cpu') as file:
                    raw = file.get_tensor(item['source_key'])
                    expected = (reorder(raw,num_query_groups=item['qkv_heads'],heads_per_group=1,
                        head_dim=item['head_dim']) if item.get('qkv_heads') else raw.clone())
                    del raw
                for pair in item['lora_pairs']:
                    with safe_open(pair['a']['source_file'],framework='pt',device='cpu') as f:
                        a,b = f.get_tensor(pair['a']['source_key']),f.get_tensor(pair['b']['source_key'])
                        merge(expected[pair['start_row']:pair['end_row']],a,b,name=item['model_name'])
                        del a,b
                plan = TensorPlan(item['model_name'],ref(item),item.get('qkv_heads'),item.get('head_dim'),
                    tuple(LoRA(x['start_row'],x['end_row'],ref(x['a']),ref(x['b'])) for x in item['lora_pairs']))
                count = expected.numel()
                size = (count+1)//2
                expected_hash = hashlib.sha256()
                actual_hash = hashlib.sha256()
                for rank in (0,1):
                    shards,_ = build_torch_block_shard([plan],2,rank)
                    actual = shards[item['dtype']][:max(0,min(size,count-rank*size))]
                    wanted = expected.flatten()[rank*size:rank*size+actual.numel()]
                    for lo in range(0,actual.numel(),1<<18):
                        x,y = actual[lo:lo+(1<<18)],wanted[lo:lo+(1<<18)]
                        assert bool(torch.isfinite(y).all()),item['model_name']
                        assert torch.equal(x.view(torch.uint8),y.view(torch.uint8)), 'bitwise mismatch: '+item['model_name']
                        expected_hash.update(memoryview(y.view(torch.uint8).numpy()))
                        actual_hash.update(memoryview(x.view(torch.uint8).numpy()))
                    if rank*size+size > count:
                        assert torch.count_nonzero(shards[item['dtype']][actual.numel():]) == 0
                    del shards,actual,wanted,x,y
                assert expected_hash.hexdigest()==actual_hash.hexdigest()
                row = dict(name=item['model_name'],shape=item['shape'],dtype=item['dtype'],
                           sha256=actual_hash.hexdigest(),bitwise_equal=True,max_abs_diff=0,
                           seconds=time.time()-start)
                log.write(json.dumps(row)+'\n'); log.flush()
                result.update(verified=index+1,elapsed_seconds=time.time()-begin,
                              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                result['lora_pairs'] += len(item['lora_pairs'])
                result['qkv'] += int(bool(item.get('qkv_heads')))
                (args.output/'status.json').write_text(json.dumps(result,indent=2))
                if (index+1)%26==0:
                    print(json.dumps(result),flush=True)
                del expected,plan
                for reader in readers.values():
                    reader.payload_reads.clear()
                gc.collect()
            assert (result['verified'],result['lora_pairs'],result['qkv'])==(1335,208,52)
            result['status']='passed'
        except BaseException as exc:
            result.update(status='failed',error=repr(exc))
            raise
        finally:
            result['elapsed_seconds']=time.time()-begin
            (args.output/'status.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
