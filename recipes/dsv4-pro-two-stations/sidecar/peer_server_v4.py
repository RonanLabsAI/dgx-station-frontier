# Derived from original-el8/dgx-station-gb300-research @ 7699ae54, deepseek-v4.1-flash/m3/sidecar/peer_server2.py
# Copyright 2026 Jason Cook. Licensed under the Apache License 2.0.
# Modifications Copyright 2026 RonanLabs (Apache License 2.0): DeepSeek-V4-Pro-0813 geometry (H 7,168, I 3,072),
# a variable warm-expert count per layer (one b12x weight plan per distinct count), per-EP-rank expert selection,
# BF16 rows on the wire, W4A8-MX (A16 selectable but returns NaN on this geometry, 2026-10-08), handshake/table words, and paths from arguments.
"""RTX PRO 6000 warm-expert sidecar for DeepSeek-V4-Pro (E3), one per EP rank / Station.

Loads this EP rank's warm experts for every hooked layer (rowmap "warm" ids that fall in [rank*192, rank*192+192),
sorted; the sorted position is the wire id) straight from the checkpoint, prepares them for b12x's SM120 fused MoE
(MXFP4 E2M1 weights + E8M0 per-32 scales, W4A8 MX: b12x master's W4A16 path returned NaN for H7168/I3072 in
tests/b12x_layer_check.py, so A16 parity with marlin is not available), and
captures one CUDA graph per (layer, row bucket). Per published sequence it copies the compacted BF16 rows in, pads
the bucket with masked routes, replays, and copies the rows' outputs back.

  CUDA_VISIBLE_DEVICES=<6000> python peer_server_v4.py --rowmap rowmap.json --model /model --ep-rank 0
"""
import argparse
import gc
import json
import os
import struct
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.environ.get("E3_HOOK_DIR", "/e3/hook"))
import peer_tier_v4 as pt  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

H, I, K, LIMIT, E = pt.HIDDEN, 3072, pt.TOPK, 10.0, 384
BUCKETS = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if m <= pt.MAX_ROWS]


def tensor_reader(model):
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def tensor(name):
        file = os.path.join(model, index[name])
        if file not in headers:
            with open(file, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                headers[file] = (8 + size, json.loads(f.read(size)))
        base, header = headers[file]
        meta = header[name]
        start, end = meta["data_offsets"]
        with open(file, "rb") as f:
            f.seek(base + start)
            data = f.read(end - start)
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(meta["shape"])

    return tensor


def request_for_capacity(decl, *, name, calls, benchmark_calls=None):
    # = b12x benchmarks/moe_preparation.py (master 2cc7f66a), which the wheel does not ship.
    if benchmark_calls is None:
        benchmark_calls = calls
    if getattr(decl, "token_counts", None) is None:
        return decl.request(name=name, prepare_call=next(iter(calls.values())),
                            benchmark_call=next(iter(benchmark_calls.values())))
    return decl.request(name=name, prepare_calls=calls, benchmark_calls=benchmark_calls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rowmap", required=True)
    p.add_argument("--model", default="/model")
    p.add_argument("--ep-rank", type=int, required=True)
    p.add_argument("--ep-size", type=int, default=2)
    p.add_argument("--mode", choices=("a16", "a8"), default=os.environ.get("E3_MODE", "a8"))
    p.add_argument("--logdir", default=os.environ.get("E3_LOGDIR", "/logs"))
    a = p.parse_args()

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    n_local = E // a.ep_size
    rowmap = pt.load_rowmap(a.rowmap)
    warm = pt.rank_warm(rowmap, a.ep_rank, n_local)
    layers = sorted(warm)
    total = sum(len(v) for v in warm.values())
    print(f"ep_rank {a.ep_rank}: {total} warm experts over {len(layers)} layers "
          f"({layers[0]}..{layers[-1]}), mode {a.mode}, hash {pt.warm_hash(warm):#x}", flush=True)

    # Shared buffer first, marked not-serving, so a GB300 worker that starts early waits instead of failing.
    buf = pt.open_shared(create=True)
    host, _ = pt.register(buf, dev)
    host[:pt.X_OFF].zero_()
    words = np.frombuffer(buf, dtype=np.int64, count=16)
    table = np.frombuffer(buf, dtype=np.int32, count=pt.MAX_LAYERS, offset=pt.TABLE_OFF)
    header = np.frombuffer(buf, dtype=np.int64, count=pt.HEADER, offset=pt.SLOT)

    tensor = tensor_reader(a.model)
    rows_max = max(BUCKETS)
    a_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    ids_buf = torch.full((rows_max, K), -1, dtype=torch.int32, device=dev)
    w_buf = torch.zeros(rows_max, K, dtype=torch.float32, device=dev)
    out_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    a_buf.normal_(0.0, 0.3, generator=g)
    mode = fm.ActivationMode.A16 if a.mode == "a16" else fm.ActivationMode.A8

    plans, experts, decls, keep = {}, {}, {}, []
    shared_scratch, shared_src = {}, {}
    start = time.time()
    for layer in layers:
        ids = warm[layer]
        n = len(ids)
        # Tune on realistic compacted rows: one or two live warm routes per row, the rest masked (an all-masked
        # autotune times empty work and picks slow configurations). Ids must be < n for this layer.
        live = torch.rand(rows_max, K, device=dev, generator=g) < (1.6 / K)
        live[:, 0] = True
        ids_buf.copy_(torch.where(live, torch.randint(0, n, (rows_max, K), device=dev, generator=g,
                                                      dtype=torch.int32), -1))
        w_buf.copy_(torch.where(live, torch.rand(rows_max, K, device=dev, generator=g) * 0.3, 0.0))
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in ids:
            pre = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{pre}.w1.weight"), tensor(f"{pre}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{pre}.w1.scale"), tensor(f"{pre}.w3.scale")]))
            parts["w2"].append(tensor(f"{pre}.w2.weight"))
            parts["s2"].append(tensor(f"{pre}.w2.scale"))
        W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
        del parts
        if layer == layers[0]:
            print(f"layer {layer}: w13 {tuple(W['w13'].shape)} s13 {tuple(W['s13'].shape)} "
                  f"w2 {tuple(W['w2'].shape)} s2 {tuple(W['s2'].shape)}", flush=True)
        ones = torch.ones(n, dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        if n not in plans:
            plans[n] = fm.plan_weights(
                source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"),
                                       w13_layout=fm.W13Layout("w31")),
                activation=fm.ActivationSpec(mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16,
                                             swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
                geometry=fm.MoEGeometry(num_experts=n, hidden_size=H, intermediate_size=I))
        # b12x's DeepSeek contract: E8M0 scales clamped to 247, [w1, w3] rows as layout "w31".
        ex = fm.prepare_weights(plan=plans[n], weights=fm.PackedWeights(
            w13=W["w13"], w2=W["w2"],
            w13_block_scales=W["s13"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w2_block_scales=W["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
            max_tokens=rows_max, top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))

        def call(st, m, ex=ex):
            # Trial calls share one scratch set and one input copy per bucket across layers; grow the shared
            # buffer to the largest request seen (b12x accepts any buffer at least as large as required).
            specs = st.scratch.scratch_specs()
            cur = shared_scratch.get(m)
            if cur is None or len(cur) != len(specs) or any(
                    t.numel() < int(sp.shape[0]) or t.dtype != sp.dtype for t, sp in zip(cur, specs)):
                shared_scratch[m] = tuple(
                    torch.empty(max(int(sp.shape[0]), cur[i].numel() if cur and i < len(cur) else 0),
                                dtype=sp.dtype, device=sp.device)
                    for i, sp in enumerate(specs))
            if m not in shared_src:
                shared_src[m] = a_buf[:m].clone()
            binding = st.bind(scratch=shared_scratch[m], a=a_buf[:m], experts=ex, topk_ids=ids_buf[:m],
                              topk_weights=w_buf[:m], output=out_buf[:m], input_scales_static=True)
            return PreparedCall(run=binding.run, output=out_buf[:m], owners=(binding,))

        calls = {m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}

        def race(st, m):
            trial = call(st, m)
            return replace(trial, produce=lambda: a_buf[:m].copy_(shared_src[m]))

        session = PreparationSession(device=dev)
        session.prepare((request_for_capacity(
            decl, name=f"e3.warm.l{layer}", calls=calls,
            benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
        del session, calls
        gc.collect()
        torch.cuda.empty_cache()
        experts[layer], decls[layer] = ex, decl
        keep.append(W)  # prepare_weights may rewrite these in place; prepared experts can alias their storage
        table[layer] = n
        print(f"layer {layer}: {n} warm experts prepared, {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated",
              flush=True)
    print(f"prepared {len(layers)} layers in {time.time() - start:.0f}s; "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    from b12x.preparation.types import require_prepared
    for layer in layers:
        for m in BUCKETS:
            variant = decls[layer].variants[m]
            specs = require_prepared(variant, variant.component_id).scratch.scratch_specs()
            assert all(t.numel() >= int(s.shape[0]) and t.dtype == s.dtype
                       for t, s in zip(shared_scratch[m], specs)), f"scratch too small: layer {layer} bucket {m}"
    scratch = {m: shared_scratch[m] for m in BUCKETS}

    x_host = host[pt.X_OFF:pt.IDS_OFF].view(torch.bfloat16)
    ids_host = host[pt.IDS_OFF:pt.W_OFF].view(torch.int32)
    w_host = host[pt.W_OFF:pt.OUT_OFF].view(torch.float32)
    out_host = host[pt.OUT_OFF:pt.OUT_OFF + rows_max * H * 2].view(torch.bfloat16)
    stream = torch.cuda.Stream(dev)
    torch.cuda.set_stream(stream)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    ids_buf.fill_(-1)
    ids_buf[:, 0] = 0
    w_buf.zero_()
    w_buf[:, 0] = 0.1
    for layer in layers:
        for m in BUCKETS:
            binding = fm.bind(decls[layer].variants[m], scratch=scratch[m], a=a_buf[:m], experts=experts[layer],
                              topk_ids=ids_buf[:m], topk_weights=w_buf[:m], output=out_buf[:m],
                              input_scales_static=True)

            def work(m=m, binding=binding):
                small = m <= pt.SMALL_ROWS
                if small:
                    # Small buckets copy whole: one replay per decode layer. The GB300 masks rows past its count.
                    a_buf[:m].view(-1).copy_(x_host[: m * H], non_blocking=True)
                    ids_buf[:m].view(-1).copy_(ids_host[: m * K], non_blocking=True)
                    w_buf[:m].view(-1).copy_(w_host[: m * K], non_blocking=True)
                fm.run(binding=binding)
                if small:
                    out_host[: m * H].copy_(out_buf[:m].view(-1), non_blocking=True)

            work()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                work()
            graphs[(layer, m)] = graph
    ids_buf.fill_(-1)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} graphs; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
          f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB peak reserved", flush=True)

    words[8] = pt.warm_hash(warm)
    words[9] = os.getpid()
    words[10] = a.ep_rank
    words[7] = pt.MAGIC
    print("serving", flush=True)
    last, served, busy, rows_total = int(words[0]), 0, 0.0, 0
    stats_file = os.path.join(a.logdir, f"e3_peer_stats_rank{a.ep_rank}.json")
    stats_reset = os.path.join(a.logdir, "e3_peer_stats.reset")
    stats, last_dump = {}, 0.0
    bucket_of = [0] + [next(m for m in BUCKETS if m >= r) for r in range(1, rows_max + 1)]
    while True:
        seq = int(words[0])
        if seq == last:
            continue
        if int(header[0]) != seq:
            continue  # header not yet visible for this sequence
        layer, rows = int(header[1]), int(header[2])
        if rows > 0:
            t0 = time.perf_counter()
            m = bucket_of[rows]
            if m <= pt.SMALL_ROWS:
                graphs[(layer, m)].replay()
            else:
                a_buf[:rows].view(-1).copy_(x_host[: rows * H], non_blocking=True)
                ids_buf[:rows].view(-1).copy_(ids_host[: rows * K], non_blocking=True)
                w_buf[:rows].view(-1).copy_(w_host[: rows * K], non_blocking=True)
                if rows < m:
                    ids_buf[rows:m].fill_(-1)
                graphs[(layer, m)].replay()
                out_host[: rows * H].copy_(out_buf[:rows].view(-1), non_blocking=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            busy += dt
            served += 1
            rows_total += rows
            s = stats.setdefault(m, [0, 0.0, 0])
            s[0] += 1
            s[1] += dt
            s[2] += rows
            if served % 256 == 0:
                now = time.time()
                if now - last_dump > 2:
                    last_dump = now
                    if os.path.exists(stats_reset):
                        stats.clear()
                        os.remove(stats_reset)
                    with open(stats_file + ".tmp", "w") as f:
                        json.dump({str(k): {"calls": v[0], "mean_us": round(v[1] / v[0] * 1e6, 1),
                                            "mean_rows": round(v[2] / v[0], 1)} for k, v in sorted(stats.items())},
                                  f)
                    os.replace(stats_file + ".tmp", stats_file)
            if served % 5000 == 0:
                print(f"served {served} warm layer calls, mean {busy / served * 1e6:.1f} us, "
                      f"mean rows {rows_total / served:.1f}", flush=True)
        words[1] = seq
        last = seq


if __name__ == "__main__":
    main()
