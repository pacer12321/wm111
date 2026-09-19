"""All real main-block weights via native prepared DLO hook, both CUDA slots.

Correctness only, one block resident at a time. Not pipeline or timing evidence.
Run two ranks with torchrun after fresh resource admission; no foreign cleanup.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from manifest_builder import tensor_plans_for_main_block
from streaming_shards import RangeReader
from torch_streaming_shards import build_torch_block_shard
from prepared_shard_hook import bundle_from_streaming, prepare_all_meta_blocks, make_prepared_hook_class


def module_for_plans(plans):
    root = torch.nn.Module()
    for p in plans:
        path = p.name.split('.')[2:]
        parent = root
        for name in path[:-1]:
            if name not in parent._modules:
                parent.add_module(name,torch.nn.Module())
            parent = parent._modules[name]
        dtype = {'BF16':torch.bfloat16,'F32':torch.float32}[p.source.info['dtype']]
        parent.register_parameter(path[-1],torch.nn.Parameter(torch.empty(p.source.info['shape'],device='meta',dtype=dtype),requires_grad=False))
    return root


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==2 and rank in (0,1)
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    # Bound our allocator to 8 GiB on each 80 GiB A100.
    torch.cuda.set_per_process_memory_fraction(0.1,rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size()==2
    manifest=json.loads(args.manifest.read_text())
    assert manifest.get('model_metadata_validated')
    refs={r['name']:r for r in map(json.loads,args.references.read_text().splitlines())}
    assert len(refs)==1335 and all(x['bitwise_equal'] for x in refs.values())
    args.output.mkdir(exist_ok=True)
    log=(args.output/f'rank{rank}.jsonl').open('x')
    readers={}
    def reader(path):
        if path not in readers: readers[path]=RangeReader(path)
        return readers[path]
    start=time.time(); result={'status':'running','rank':rank,'verified_blocks':0,'verified_tensor_slot_pairs':0,
                              'scope':'native prepared hook CUDA AllGather; both slots; not pipeline/forward/timing'}
    try:
        for i in range(50):
            plans=tensor_plans_for_main_block(manifest,i,reader)
            shards,metadata=build_torch_block_shard(plans,2,rank,pin_memory=True)
            bundle=bundle_from_streaming(f'blocks.{i}',2,rank,shards,metadata)
            # This isolated 1-block group is index0; names inside it stay exact.
            bundle.block_name='blocks.0'
            block=module_for_plans(plans)
            prepare_all_meta_blocks([block],[bundle],require_pinned=True)
            hook=make_prepared_hook_class()(next_block=block,device=torch.device('cuda',rank),
                dp_group=dist.group.WORLD,dp_size=2,rank=rank,copy_stream=torch.cuda.Stream(),
                comm_stream=torch.cuda.Stream(),pin_memory=True,prepared=bundle)
            hook.initialize_hook(block)
            # Native standalone hook allocates output buffers only. Backend
            # normally supplies these two rank-local AllGather input buffers.
            hook.gpu_shard_buffers = [
                {dtype: torch.empty_like(shard, device=torch.device('cuda', rank))
                 for dtype, shard in hook.cpu_shards.items()} for _ in range(2)
            ]
            for slot in (0,1):
                hook.prefetch_layer(slot,non_blocking=False)
                torch.cuda.synchronize()
                for name,weight in block.named_parameters():
                    sha=hashlib.sha256()
                    flat=weight.detach().reshape(-1)
                    for lo in range(0,flat.numel(),1<<18):
                        chunk=flat[lo:lo+(1<<18)].cpu()
                        sha.update(memoryview(chunk.view(torch.uint8).numpy()))
                    key=f'blocks.{i}.{name}'
                    assert sha.hexdigest()==refs[key]['sha256'],f'GPU weight mismatch {key} rank{rank} slot{slot}'
                    result['verified_tensor_slot_pairs']+=1
                log.write(json.dumps(dict(block=i,slot=slot,rank=rank,tensors=26,bitwise_equal=True))+'\n');log.flush()
                hook.offload_layer()
            result.update(verified_blocks=i+1,elapsed_seconds=time.time()-start,
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(rank))
            (args.output/f'rank{rank}_status.json').write_text(json.dumps(result,indent=2))
            if (i+1)%5==0: print(json.dumps(result),flush=True)
            del hook,block,bundle,shards,metadata,weight,flat,chunk
            for r in readers.values(): r.payload_reads.clear()
            torch.cuda.empty_cache()
        assert result['verified_tensor_slot_pairs']==2600
        dist.barrier()
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['elapsed_seconds']=time.time()-start
        (args.output/f'rank{rank}_status.json').write_text(json.dumps(result,indent=2))
        log.close(); dist.destroy_process_group()
        print(json.dumps(result),flush=True)

if __name__=='__main__': main()
