# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# lambda probe: torch NCCL all-reduce under CUDA-graph replay, bf16, p50 over 20 replays of 100 captured ops, max over ranks.
# env RANK, RANK0_IP (rendezvous on port 29512). Run via tp.sh.
import os, torch, torch.distributed as dist
r = int(os.environ["RANK"]); torch.cuda.set_device(0)
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['RANK0_IP']}:29512", rank=r, world_size=2, device_id=torch.device("cuda:0"))
N = 100
sizes = [int(s) for s in os.environ.get("SIZES", "8192,12288,14336,16384,65536,98304,131072,196608,229376,262144,524288").split(",")]
for nb in sizes:
    x = torch.ones(nb // 2, dtype=torch.bfloat16, device="cuda"); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(N): dist.all_reduce(x)
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); dist.barrier(); ts = []
    for _ in range(5):  # each sample = 20 back-to-back replays (2,000 ops), so cross-host launch skew is amortised
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        g.replay(); a.record()
        for _ in range(20): g.replay()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / (20 * N))
    t = torch.tensor([sorted(ts)[2], min(ts)], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    # eager (no graph) single-op latency for reference
    torch.cuda.synchronize(); dist.barrier(); es = []
    for _ in range(50):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); dist.all_reduce(x); b.record(); torch.cuda.synchronize(); es.append(a.elapsed_time(b) * 1e3)
    e = torch.tensor([sorted(es)[25]], device="cuda"); dist.all_reduce(e, op=dist.ReduceOp.MAX)
    if r == 0: print(f"LAM {nb:>7} B  graph p50(max-rank) {t[0].item():7.2f} us  min {t[1].item():7.2f} us  | eager p50 {e.item():7.2f} us", flush=True)
    del g
dist.destroy_process_group()
