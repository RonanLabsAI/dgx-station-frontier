# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Check the 3-bit repack in sitecustomize.py: pack random int3 codes with compressed-tensors' own pack_to_int32
(canonical 10 values per int32), repack to the bit-dense layout, unpack with humming's common_unpack_weight semantics,
and compare with the original codes, for both Kimi K3 expert shapes.

usage (inside the vLLM image, which ships torch and compressed-tensors; no GPU needed):
  docker run --rm -v $PWD/recipes/kimi-k3/vllm/hook:/opt/k3hook:ro --entrypoint python3 <vllm image> /opt/k3hook/test_k3hook.py
expected: two lines ending in "exact True"
"""
import importlib.util
import os

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["K3_REPACK3"] = "0"          # load the module without installing the import hook
spec = importlib.util.spec_from_file_location("k3hook", os.path.join(HERE, "sitecustomize.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

torch.manual_seed(0)
ok = True
for N, K in ((3072, 3584), (3584, 3072)):
    q = torch.randint(-4, 4, (N, K), dtype=torch.int8)
    packed = pack_to_int32(q, 3)                       # canonical: 10 values per int32
    dense = h.repack_10_to_dense(packed)
    d = (dense.to(torch.int64) & 0xFFFFFFFF).reshape(N, K // 32, 3)
    vals = torch.empty(N, K // 32, 32, dtype=torch.int64)
    for i in range(32):
        idx = 3 * i
        wi, off = idx // 32, idx % 32
        v = (d[..., wi] >> off) & 7
        if off + 3 > 32:
            p1 = 32 - off
            v = ((d[..., wi] >> off) & ((1 << p1) - 1)) | ((d[..., wi + 1] & ((1 << (3 - p1)) - 1)) << p1)
        vals[..., i] = v
    rec = vals.reshape(N, K) - 4
    exact = bool(torch.equal(rec.to(torch.int8), q))
    ok &= exact
    print(N, K, tuple(packed.shape), "->", tuple(dense.shape), "exact", exact)
raise SystemExit(0 if ok else 1)
