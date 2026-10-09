# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""One-layer numerics check of the sidecar's b12x call against a plain PyTorch reference (run on a 6000).

  python b12x_layer_check.py --model /model --layer 5 --experts 0,1,2,3 [--mode a16|a8] [--layout w31|w13]

Reference: FP4 E2M1 (low nibble first) x E8M0 per-32 scale dequant, gate = x w1^T, up = x w3^T,
silu(min(gate, L)) * clamp(up, -L, L), then w2, weighted sum over the token's routes.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.environ.get("E3_HOOK_DIR", "/e3/hook"))
sys.path.insert(0, "/e3/sidecar")
from peer_server_v4 import tensor_reader  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402

H, I, K, LIMIT = 7168, 3072, 6, 10.0
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def deq(w, s):
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    v = torch.stack([E2M1.to(w.device)[lo], E2M1.to(w.device)[hi]], -1).reshape(w.shape[0], -1)
    sc = torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)
    return v * sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--experts", default=",".join(str(i) for i in range(36)))
    p.add_argument("--mode", default="a16")
    p.add_argument("--layout", default="w31")
    p.add_argument("--tokens", type=int, default=16)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    tensor = tensor_reader(a.model)
    ids = [int(x) for x in a.experts.split(",")]
    n = len(ids)
    raw = {}
    for e in ids:
        pre = f"layers.{a.layer}.ffn.experts.{e}"
        raw[e] = {k: tensor(f"{pre}.{k}.weight").to(dev) for k in ("w1", "w2", "w3")}
        raw[e].update({k + "s": tensor(f"{pre}.{k}.scale").to(dev) for k in ("w1", "w2", "w3")})
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(a.tokens, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    topk = torch.full((a.tokens, K), -1, dtype=torch.int32, device=dev)
    topk[:, 0] = torch.arange(a.tokens, device=dev, dtype=torch.int32) % n
    topk[:, 1] = (torch.arange(a.tokens, device=dev, dtype=torch.int32) + 1) % n
    w = torch.zeros(a.tokens, K, device=dev)
    w[:, 0], w[:, 1] = 0.7, 0.3
    ref = torch.zeros(a.tokens, H, device=dev)
    for t in range(a.tokens):
        for k in range(2):
            r = raw[ids[int(topk[t, k])]]
            gate = x[t].float() @ deq(r["w1"], r["w1s"]).T
            up = x[t].float() @ deq(r["w3"], r["w3s"]).T
            h = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
            ref[t] += w[t, k] * (h.to(torch.bfloat16).float() @ deq(r["w2"], r["w2s"]).T)
    first, second = ("w1", "w3") if a.layout == "w31" else ("w3", "w1")
    W13 = torch.stack([torch.cat([raw[e][first], raw[e][second]]) for e in ids])
    S13 = torch.stack([torch.cat([raw[e][first + "s"], raw[e][second + "s"]]) for e in ids])
    W2 = torch.stack([raw[e]["w2"] for e in ids])
    S2 = torch.stack([raw[e]["w2s"] for e in ids])
    mode = fm.ActivationMode.A16 if a.mode == "a16" else fm.ActivationMode.A8
    plan = fm.plan_weights(
        source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout(a.layout)),
        activation=fm.ActivationSpec(mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=LIMIT,
                                     swiglu_alpha=None, swiglu_beta=None),
        geometry=fm.MoEGeometry(num_experts=n, hidden_size=H, intermediate_size=I))
    ones = torch.ones(n, device=dev)
    one = torch.ones((), device=dev)
    ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
        w13=W13, w2=W2, w13_block_scales=S13.clamp(max=247).view(torch.float8_e8m0fnu),
        w2_block_scales=S2.clamp(max=247).view(torch.float8_e8m0fnu), w13_global_scales=ones, w2_global_scales=ones,
        input_scale=one, intermediate_scale=one, immutable_input_scales=True))
    decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
        max_tokens=64, top_k=K, warmup_token_counts=(1, 2, 4, 8, 16, 32, 64), route_num_experts=0))
    out = torch.zeros(64, H, dtype=torch.bfloat16, device=dev)
    xb = torch.cat([x, torch.zeros(64 - a.tokens, H, dtype=x.dtype, device=dev)])
    tb = torch.cat([topk, torch.full((64 - a.tokens, K), -1, dtype=topk.dtype, device=dev)])
    tb[a.tokens:, 0] = 0
    wb = torch.cat([w, torch.zeros(64 - a.tokens, K, device=dev)])
    from b12x.preparation import PreparationSession, PreparedCall
    from peer_server_v4 import request_for_capacity
    scratch = {}

    def call(st, m):
        specs = st.scratch.scratch_specs()
        scratch[m] = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=sp.device) for sp in specs)
        b = st.bind(scratch=scratch[m], a=xb[:m], experts=ex, topk_ids=tb[:m], topk_weights=wb[:m], output=out[:m],
                    input_scales_static=True)
        return PreparedCall(run=b.run, output=out[:m], owners=(b,))
    calls = {m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}
    from dataclasses import replace
    src = xb.clone()

    def race(st, m):
        return replace(call(st, m), produce=lambda: xb[:m].copy_(src[:m]))
    PreparationSession(device=dev).prepare((request_for_capacity(
        decl, name="chk", calls=calls, benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
    from b12x.preparation.types import require_prepared
    m = a.tokens
    variant = decl.variants[m]
    st = require_prepared(variant, variant.component_id)
    sc = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=sp.device) for sp in st.scratch.scratch_specs())
    b = fm.bind(variant, scratch=sc, a=xb[:m], experts=ex, topk_ids=tb[:m], topk_weights=wb[:m], output=out[:m],
                input_scales_static=True)
    fm.run(binding=b)
    torch.cuda.synchronize()
    o = out[:m].float()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    print(f"mode={a.mode} layout={a.layout} layer={a.layer} experts={ids}: cos={cos:.6f} "
          f"|ref|={ref.norm():.3f} |out|={o.norm():.3f} ratio={o.norm() / ref.norm():.3f}")


if __name__ == "__main__":
    main()
