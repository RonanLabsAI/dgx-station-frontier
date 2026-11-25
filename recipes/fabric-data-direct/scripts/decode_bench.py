#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Concurrent decode throughput against an OpenAI-compatible server.
Method mirrors the October A/B: prose prompt with a uuid nonce, 256 output tokens (ignore_eos), 3 reps per concurrency.
Aggregate tok/s = sum(completion_tokens) / wall time of the batch. env BASE_URL MODEL TAG CONCS REPS."""
import json, os, statistics, sys, threading, time, urllib.request, uuid

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8004/v1")
MODEL = os.getenv("MODEL", "glm-5.3")
TAG = os.getenv("TAG", "x")
CONCS = [int(c) for c in os.getenv("CONCS", "1,8,16,32").split(",")]
REPS = int(os.getenv("REPS", "3"))
NTOK = 256
PROMPT = ("Write a long, flowing essay on why careful engineers keep written records of their experiments, "
          "how those records build intellectual humility, and what a good lab notebook looks like. Nonce: {n}")


def one(out, i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(n=uuid.uuid4())}],
            "max_tokens": NTOK, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out[i] = json.load(r)["usage"]["completion_tokens"]


def batch(c):
    out = [0] * c
    th = [threading.Thread(target=one, args=(out, i)) for i in range(c)]
    t0 = time.monotonic()
    for t in th: t.start()
    for t in th: t.join()
    dt = time.monotonic() - t0
    return sum(out) / dt, out


batch(4)  # warm-up
for c in CONCS:
    batch(c)  # warm pass at this concurrency (graph capture / allocator)
    reps = []
    for _ in range(REPS):
        agg, toks = batch(c)
        assert all(t == NTOK for t in toks), f"short completions: {toks}"
        reps.append(agg)
    med = statistics.median(reps)
    print(f"[{TAG}] conc={c:>2}: {med:8.1f} agg tok/s {med / c:7.2f} per-stream "
          f"(reps {'/'.join(f'{r:.0f}' for r in reps)})", flush=True)
