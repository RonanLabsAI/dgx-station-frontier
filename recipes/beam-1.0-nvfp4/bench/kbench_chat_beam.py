#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""kbench_chat_beam.py BASE MODEL POOL.jsonl OUT.json C [--max-tokens 1024] [--natural] [--think off|on] [--warm W.jsonl]
    [--nonce STR] [--head N]

Real-text decode bench through the server's own chat template (Beam copy of recipes/kimi-k3/bench/kbench_chat_ds.py). Deltas
against that file: (1) "--think off" (the default) sends chat_template_kwargs {reasoning_effort: "minimal"}: the Beam template
always reasons, and minimal is its smallest budget; forced tokens count reasoning and content deltas alike; (2) every request
also counts its streamed delta chunks, and the summary reports chunk_acc = completion tokens / chunks. Under SGLang speculative
decoding one chunk is one verify step (stream_interval 1), so chunk_acc is the observed accepted length (1.0 without a drafter);
(3) SGLang's /metrics gauges sglang:spec_accept_length and sglang:cache_hit_rate are recorded. SGLang exposes none of the vLLM
counters this script also reads, so prefix_hit_pct / prefix_queries / spec_* read 0 on SGLang; prefix-cache hygiene is the
--nonce (prepends "[ref STR-<i>]" to every prompt, so the first KV block differs per run) plus disjoint slices and a separate
--warm slice.

Streaming; per request TTFT and TPOT = (t_last - t_first) / (completion_tokens - 1) with completion_tokens from the final usage
chunk. decode_agg = C x 1000 / mean TPOT (excludes TTFT and queueing; not user-seen throughput). output_tok_s = all completion
tokens / wall-clock of the measured run (includes TTFT). NP = number of prompts in POOL (all sent, C in flight); --head N sends
only the first N. Default: 1,024 FORCED tokens (ignore_eos + min_tokens), temperature 0. --natural: EOS honoured.
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def metrics(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    v = {"hits": 0.0, "queries": 0.0, "drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "sgl_accept_len": None,
         "sgl_cache_hit_rate": None}
    for line in txt.splitlines():
        # SGLang exposes gauges, not vLLM counters; recorded as-is, prefix hygiene = nonce only
        if line.startswith("sglang:spec_accept_length"):
            v["sgl_accept_len"] = float(line.split()[-1])
        elif line.startswith("sglang:cache_hit_rate"):
            v["sgl_cache_hit_rate"] = float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_hits_total"):
            v["hits"] += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_queries_total"):
            v["queries"] += float(line.split()[-1])
        elif line.startswith(("vllm:spec_decode_num_drafts_total{", "vllm:spec_decode_num_drafts_total ")):
            v["drafts"] += float(line.split()[-1])
        elif line.startswith(("vllm:spec_decode_num_draft_tokens_total{", "vllm:spec_decode_num_draft_tokens_total ")):
            v["draft_tokens"] += float(line.split()[-1])
        elif line.startswith(("vllm:spec_decode_num_accepted_tokens_total{",
                              "vllm:spec_decode_num_accepted_tokens_total ")):
            v["accepted"] += float(line.split()[-1])
    return v


def one(base, model, content, max_tokens, natural, think):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
            "messages": [{"role": "user", "content": content}]}
    if think == "off":
        body["chat_template_kwargs"] = {"reasoning_effort": "minimal"}  # beam: template always reasons; smallest budget
    if not natural:
        body.update({"ignore_eos": True, "min_tokens": max_tokens})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    nchunk = 0
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])
            delta = ch[0].get("delta", {}) if ch else {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                now = time.time()
                first = first or now
                nchunk += 1
                last = now
    n = (usage or {}).get("completion_tokens") or 0
    return {"ttft": (first - t0) if first else None, "ctoks": n, "chunks": nchunk, "ptoks": (usage or {}).get("prompt_tokens"),
            "tpot_ms": 1000 * (last - first) / max(n - 1, 1) if first and n > 1 else None,
            "wall": (last or time.time()) - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("pool")
    ap.add_argument("out")
    ap.add_argument("c", type=int)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--think", choices=("off", "on"), default="off")
    ap.add_argument("--warm", default=None)
    ap.add_argument("--maxhit", type=float, default=2.0)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    items = [json.loads(x)["content"] for x in open(a.pool)]
    if a.head:
        items = items[:a.head]
    if a.nonce:
        items = [f"[ref {a.nonce}-{i}]\n{c}" for i, c in enumerate(items)]
    if a.warm:
        warm = [json.loads(x)["content"] for x in open(a.warm)]
        if a.nonce:
            warm = [f"[ref {a.nonce}-w{i}]\n{c}" for i, c in enumerate(warm)]
        with ThreadPoolExecutor(len(warm)) as ex:
            list(ex.map(lambda c: one(base, a.model, c, 64, True, a.think), warm))
    m0 = metrics(base)
    t0 = time.time()
    with ThreadPoolExecutor(a.c) as ex:
        res = list(ex.map(lambda c: one(base, a.model, c, a.max_tokens, a.natural, a.think), items))
    wall = time.time() - t0
    m1 = metrics(base)
    dq, dh = m1["queries"] - m0["queries"], m1["hits"] - m0["hits"]
    hit = 100 * dh / dq if dq > 0 else 0.0
    ok = [r for r in res if r["tpot_ms"]]
    mt = statistics.mean(r["tpot_ms"] for r in ok) if ok else float("nan")
    summ = {"C": a.c, "np": len(items), "ok": len(ok), "mean_tpot_ms": round(mt, 2),
            "decode_agg": round(a.c * 1000 / mt, 2) if ok else None,
            "output_tok_s": round(sum(r["ctoks"] for r in res) / wall, 2), "wall_s": round(wall, 1),
            "median_ttft_s": round(statistics.median(r["ttft"] for r in ok), 2) if ok else None,
            "mean_ptoks": round(statistics.mean(r["ptoks"] or 0 for r in res), 1),
            "mean_ctoks": round(statistics.mean(r["ctoks"] for r in res), 1),
            "prefix_hit_pct": round(hit, 2), "prefix_queries": int(dq), "think": a.think,
            "spec_drafts": int(m1["drafts"] - m0["drafts"]), "spec_draft_tokens": int(m1["draft_tokens"] - m0["draft_tokens"]),
            "spec_accepted": int(m1["accepted"] - m0["accepted"]),
            "acceptance_len": (round(1 + (m1["accepted"] - m0["accepted"]) / (m1["drafts"] - m0["drafts"]), 3)
                               if m1["drafts"] > m0["drafts"] else None),
            "forced": not a.natural, "pool": a.pool, "nonce": a.nonce,
            "chunk_acc": (round(sum(r["ctoks"] for r in ok) / sum(r["chunks"] for r in ok), 3)
                          if ok and sum(r["chunks"] for r in ok) else None),
            "sglang_accept_len_gauge": m1["sgl_accept_len"], "sglang_cache_hit_rate_gauge": m1["sgl_cache_hit_rate"]}
    json.dump({"summary": summ, "req": res}, open(a.out, "w"))
    print(json.dumps(summ), flush=True)
    if hit > a.maxhit:
        print(f"PREFIX-CACHE CONTAMINATION {hit:.2f}% > {a.maxhit}% -- do not report", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
