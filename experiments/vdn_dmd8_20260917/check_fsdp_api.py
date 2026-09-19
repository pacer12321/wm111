import inspect

from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

print(CPUOffloadPolicy)
print(inspect.signature(CPUOffloadPolicy))
print(inspect.signature(fully_shard))
