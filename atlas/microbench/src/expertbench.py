# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# CPU vs zero-copy GPU vs HBM expert FFN (SwiGLU) at batch 1/4/16/64. bf16.
import time, sys, torch
H, I, NE = 7168, 2048, 48          # DeepSeek/Kimi-class routed expert; 48 experts x 88 MB = 4.2 GB pool
BATCHES = [1, 4, 16, 64]
torch.manual_seed(0)
dev = torch.device("cuda")
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), "cpu threads", torch.get_num_threads(), flush=True)

class CAI:  # expose a pinned host tensor to torch as a CUDA tensor (zero-copy over C2C/UVA)
    def __init__(self, t):
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": "<f2" if False else "<V2",
                                         "data": (t.data_ptr(), False), "version": 3, "strides": None}

def zc_view(t):
    # torch reads typestr; bf16 has no numpy typestr, so view as int16 then reinterpret
    i16 = t.view(torch.int16)
    obj = type("O", (), {})()
    obj.__cuda_array_interface__ = {"shape": tuple(i16.shape), "typestr": "<i2", "data": (i16.data_ptr(), False), "version": 3, "strides": None}
    return torch.as_tensor(obj, device=dev).view(torch.bfloat16)

def ffn(x, w13, w2):
    h = x @ w13.t()
    g, u = h.chunk(2, dim=-1)
    return (torch.nn.functional.silu(g) * u) @ w2.t()

bytes_per_exp = (2 * I * H + H * I) * 2
pin13 = [torch.randn(2 * I, H, dtype=torch.bfloat16).pin_memory() for _ in range(NE)]
pin2 = [torch.randn(H, I, dtype=torch.bfloat16).pin_memory() for _ in range(NE)]
modes = {}
modes["gpu_hbm"] = ([p.to(dev) for p in pin13], [p.to(dev) for p in pin2], dev)
modes["gpu_zerocopy_c2c"] = ([zc_view(p) for p in pin13], [zc_view(p) for p in pin2], dev)
modes["cpu_bf16"] = (pin13, pin2, torch.device("cpu"))
# correctness: zero-copy == hbm
x = torch.randn(4, H, dtype=torch.bfloat16, device=dev)
a = ffn(x, modes["gpu_hbm"][0][0], modes["gpu_hbm"][1][0]); b = ffn(x, modes["gpu_zerocopy_c2c"][0][0], modes["gpu_zerocopy_c2c"][1][0])
print("zerocopy_matches_hbm", torch.allclose(a, b), flush=True)

for name, (w13s, w2s, d) in modes.items():
    for B in BATCHES:
        x = torch.randn(B, H, dtype=torch.bfloat16, device=d)
        iters = NE * (2 if d.type == "cuda" else 1)
        for e in range(4): ffn(x, w13s[e], w2s[e])
        if d.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for k in range(iters): y = ffn(x, w13s[k % NE], w2s[k % NE])
        if d.type == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        print(f"RESULT expert_{name}_B{B} ms_per_expert {dt*1e3:.3f} eff_GBps {bytes_per_exp/dt/1e9:.1f} tok_expert_per_s {B/dt:.0f}", flush=True)
