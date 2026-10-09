# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# lambda probe on SGLang's engine path: PyNcclCommunicator (ncclAllReduce on the current stream) under CUDA-graph replay.
# env RANK, RANK0_IP (rendezvous on port 29513). Run via tp.sh, or via mpiwrap.sh under mpirun.
import os, torch, torch.distributed as dist
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
r = int(os.environ["RANK"]); torch.cuda.set_device(0)
dist.init_process_group("gloo", init_method=f"tcp://{os.environ['RANK0_IP']}:29513", rank=r, world_size=2)
c = PyNcclCommunicator(group=dist.group.WORLD, device=0); c.disabled = False
N = 100
sizes = [int(s) for s in os.environ.get("SIZES", "8192,12288,14336,16384,65536,98304,131072,196608,229376,262144,524288").split(",")]
for nb in sizes:
    x = torch.ones(nb // 2, dtype=torch.bfloat16, device="cuda"); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(20): c.all_reduce(x)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(N): c.all_reduce(x)
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); dist.barrier(); ts = []
    for _ in range(5):  # each sample = 20 back-to-back replays (2,000 ops), so cross-host launch skew is amortised
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        g.replay(); a.record()
        for _ in range(20): g.replay()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / (20 * N))
    t = torch.tensor([sorted(ts)[2], min(ts)]); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if r == 0: print(f"LAMPY {nb:>7} B  graph p50(max-rank) {t[0].item():7.2f} us  min {t[1].item():7.2f} us", flush=True)
    del g
dist.destroy_process_group()
