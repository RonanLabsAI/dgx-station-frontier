#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""kvp_microbench.py --out DIR [--mock] [--quick]
results/kv-offload.md, test 1 (phase "mb"). Runs inside the vLLM af7f9488 image on ONE GPU (container sees it as cuda:0),
host allocations bound to Grace LPDDR (numactl --membind=0). No model, no network. Writes DIR/microbench.json and prints one line.

  copy      cudaMemcpy-class copies with pinned host memory (torch pin_memory -> cudaHostAlloc): H2D, D2H, both directions at once
            (two streams), plus pageable H2D. 4 GiB buffers (bidirectional 2 x 2 GiB), 10 reps (pageable 1 GiB x 3); GB/s = 1e9 bytes/s.
  zread     a GPU kernel (torch.sum over float32) reading pinned host memory in place (zero-copy: the host buffer aliased as a CUDA
            tensor through __cuda_array_interface__, the method of the 10-06 expertbench), and the same read from device memory.
  kvread    the "crutch" case: ONE decode step of Qwen3.8-27B's attention when its KV cache must be read from where it lives.
            Geometry from config.json: 16 full-attention layers, 24 query heads, 4 KV heads, head_dim 256, bf16 KV
            (65,536 bytes per token). For L in 32768 / 131072 / 245760 tokens: 16 per-layer K and V buffers in pinned host memory
            (read in place over the link) vs the same buffers in device memory; time of the 16 attention calls for one query token
            (q.K^T, softmax, .V as two batched matmuls: each KV byte read once), median of 5. Host and device outputs are compared
            (identical and max abs difference; the aliasing proof). This is the attention read only:
            weights, linear-attention layers and sampling are not included (the vLLM arm measures real decode separately).
--mock writes a schema-identical file with made-up numbers and no torch (offline tests only; mock=true is recorded in the file).
"""
import argparse
import json
import os
import statistics
import sys
import time

GEOM = {"layers": 16, "q_heads": 24, "kv_heads": 4, "head_dim": 256, "dtype_bytes": 2}
LENGTHS = (32768, 131072, 245760)
GIB = 2 ** 30


def kv_bytes_per_token(g=GEOM):
    return g["layers"] * g["kv_heads"] * g["head_dim"] * 2 * g["dtype_bytes"]


def gbps(nbytes, seconds):
    return round(nbytes / seconds / 1e9, 2) if seconds > 0 else None


def summarize(samples_s, nbytes):
    """-> dict with median / best GB/s (pure; tested)."""
    s = sorted(samples_s)
    return {"n": len(s), "median_s": round(statistics.median(s), 6), "best_s": round(s[0], 6),
            "median_gbps": gbps(nbytes, statistics.median(s)), "best_gbps": gbps(nbytes, s[0]), "bytes": nbytes}


def mock(out):
    """Schema-identical made-up numbers; the link speed follows the output dir name (.../gb300 = C2C-like, else PCIe-like)."""
    f = 0.152 if os.path.basename(out.rstrip("/")) == "gb300" else 1.0     # time factor
    kb = kv_bytes_per_token()
    r = {"mock": True, "gpu": "MOCK-" + os.path.basename(out.rstrip("/")), "torch": "-", "kv_bytes_per_token": kb, "geometry": GEOM,
         "copy": {"h2d_pinned": summarize([0.08 * f] * 10, 4 * GIB), "d2h_pinned": summarize([0.076 * f] * 10, 4 * GIB),
                  "bidir_pinned": summarize([0.1 * f] * 10, 4 * GIB), "h2d_pageable": summarize([0.2 * f] * 3, GIB)},
         "zread": {"host_pinned": summarize([0.085 * f] * 5, 4 * GIB), "device": summarize([0.0007] * 5, 4 * GIB)},
         "kvread": {}}
    for L in LENGTHS:
        hs, ds = L * kb / 52e9 * (f if f < 1 else 1), L * kb / 5e12
        r["kvread"][str(L)] = {"host_ms": round(hs * 1e3, 3), "device_ms": round(ds * 1e3, 3), "bytes": L * kb,
                               "host_gbps": gbps(L * kb, hs), "device_gbps": gbps(L * kb, ds), "identical": True, "max_abs_diff": 0.0, "method": "mock"}
    return r


def real(quick=False):
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda:0")
    torch.cuda.init()
    r = {"mock": False, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
         "kv_bytes_per_token": kv_bytes_per_token(), "geometry": GEOM}
    reps = 3 if quick else 10
    size = (256 * 2 ** 20) if quick else 4 * GIB

    def timed(fn, n, stream=None):
        out = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return out

    h = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    h.fill_(7)
    d = torch.empty(size, dtype=torch.uint8, device=dev)
    timed(lambda: d.copy_(h, non_blocking=True), 2)     # warm-up
    r["copy"] = {"h2d_pinned": summarize(timed(lambda: d.copy_(h, non_blocking=True), reps), size),
                 "d2h_pinned": summarize(timed(lambda: h.copy_(d, non_blocking=True), reps), size)}
    half = size // 2
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def bidir():
        with torch.cuda.stream(s1):
            d[:half].copy_(h[:half], non_blocking=True)
        with torch.cuda.stream(s2):
            h[half:].copy_(d[half:], non_blocking=True)
    r["copy"]["bidir_pinned"] = summarize(timed(bidir, reps), size)
    psize = min(size, GIB)
    hp = torch.empty(psize, dtype=torch.uint8)
    hp.fill_(3)
    r["copy"]["h2d_pageable"] = summarize(timed(lambda: d[:psize].copy_(hp), 3), psize)
    del hp

    class CAI:
        def __init__(self, ptr, n, typestr):
            self.__cuda_array_interface__ = {"shape": (n,), "typestr": typestr, "data": (ptr, False), "version": 3, "strides": None}

    def alias(t, typestr, n):   # a pinned host tensor as a CUDA tensor over the same bytes (UVA; verified by the identity checks)
        return torch.as_tensor(CAI(t.data_ptr(), n, typestr), device=dev)

    hf = alias(h, "<f4", size // 4)
    df = d.view(torch.float32)
    d.copy_(h)
    torch.cuda.synchronize()
    if not torch.equal(hf.sum().view(1), df.sum().view(1)):
        raise SystemExit("zero-copy alias read differs from the device copy: aliasing not proven")
    zr = max(2, reps // 2)
    r["zread"] = {"host_pinned": summarize(timed(lambda: hf.sum(), zr), size), "device": summarize(timed(lambda: df.sum(), zr), size)}
    del hf, df, d, h
    torch.cuda.empty_cache()

    g = GEOM
    F = torch.nn.functional
    r["kvread"] = {}
    for L in (LENGTHS[:1] if quick else LENGTHS):
        n = g["kv_heads"] * L * g["head_dim"]
        q = torch.randn(1, g["q_heads"], 1, g["head_dim"], device=dev, dtype=torch.bfloat16)
        dk = [torch.randn(1, g["kv_heads"], L, g["head_dim"], device=dev, dtype=torch.bfloat16) for _ in range(g["layers"])]
        dv = [torch.randn_like(dk[0]) for _ in range(g["layers"])]
        hk, hv, ak, av = [], [], [], []
        for i in range(g["layers"]):
            for src, keep, al in ((dk[i], hk, ak), (dv[i], hv, av)):
                t = torch.empty(n, dtype=torch.int16, pin_memory=True)
                t.copy_(src.view(-1).view(torch.int16))
                keep.append(t)
                al.append(alias(t, "<i2", n).view(torch.bfloat16).view(1, g["kv_heads"], L, g["head_dim"]))
        method = "bmm"   # q.K^T then softmax then .V per layer: every K and V byte is read once, no GQA expansion copies

        def step(K, V):
            outs = []
            qq = q.view(g["kv_heads"], g["q_heads"] // g["kv_heads"], g["head_dim"])
            for i in range(g["layers"]):
                s = torch.matmul(qq, K[i][0].transpose(1, 2)).float() / (g["head_dim"] ** 0.5)
                outs.append(torch.matmul(torch.softmax(s, -1).to(V[i].dtype), V[i][0]))
            return outs
        od = step(dk, dv)
        oh = step(ak, av)
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(od, oh))
        diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(od, oh))
        th = timed(lambda: step(ak, av), 5)
        td = timed(lambda: step(dk, dv), 5)
        nb = L * kv_bytes_per_token()
        r["kvread"][str(L)] = {"host_ms": round(statistics.median(th) * 1e3, 3), "device_ms": round(statistics.median(td) * 1e3, 3),
                               "bytes": nb, "host_gbps": gbps(nb, statistics.median(th)), "device_gbps": gbps(nb, statistics.median(td)),
                               "identical": bool(same), "max_abs_diff": diff, "method": method}
        del dk, dv, hk, hv, ak, av, od, oh
        torch.cuda.empty_cache()
    return r


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    r = mock(a.out) if a.mock else real(a.quick)
    r["wall_s"] = round(time.time() - t0, 1)
    json.dump(r, open(os.path.join(a.out, "microbench.json"), "w"), indent=1, sort_keys=True)
    kv = r["kvread"]
    print("microbench %s: h2d %s GB/s, d2h %s, bidir %s, zero-copy read %s; kvread host/device ms %s; identical %s" % (
        r["gpu"], r["copy"]["h2d_pinned"]["median_gbps"], r["copy"]["d2h_pinned"]["median_gbps"], r["copy"]["bidir_pinned"]["median_gbps"],
        r["zread"]["host_pinned"]["median_gbps"], {k: (v["host_ms"], v["device_ms"]) for k, v in kv.items()},
        all(v["identical"] for v in kv.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
