# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# torch NCCL 128 MB all-reduce probe + version line. env RANK, RANK0_IP (rendezvous on port 29511)
import os, time, torch, torch.distributed as dist
r = int(os.environ["RANK"])
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['RANK0_IP']}:29511", rank=r, world_size=2)
torch.cuda.set_device(0)
x = torch.ones(32 * 1024 * 1024, device="cuda")  # 128 MB fp32
t = time.time(); dist.all_reduce(x); torch.cuda.synchronize()
print(f"rank{r} torch {torch.__version__} nccl {torch.cuda.nccl.version()} first allreduce ok val={x[0].item()} {time.time()-t:.2f}s", flush=True)
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize(); res = []
for rep in range(3):
    t = time.time()
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.time() - t) / 20; res.append(dt)
for dt in res:
    print(f"rank{r} allreduce 128MB avg {dt*1000:.2f} ms  busbw ~{(128e6*2*(1/2))/dt/1e9*8:.0f} Gb/s ({128e6/dt/1e9:.1f} GB/s busbw)", flush=True)
dist.destroy_process_group()
