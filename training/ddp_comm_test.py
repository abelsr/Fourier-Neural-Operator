# ddp_comm_test.py
import os, time, torch, torch.distributed as dist
from datetime import timedelta

rank = 8 - 1
world = 8
local_rank = rank % torch.cuda.device_count()

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(minutes=10))

# barrera inicial
dist.barrier()

# allreduce sencillo
x = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)

if rank == 0:
    print(f"[OK] all_reduce -> {x.item()} (debería ser {world})")

# barrera final
dist.barrier()
time.sleep(1)
