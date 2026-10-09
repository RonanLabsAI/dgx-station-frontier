#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# llama-server decode bench: concurrency C, N requests, n_predict P, ignore_eos, cache_prompt false (no prompt reuse;
# each prompt also starts with its own index). Prompt: about 130 tokens. Temperature 0.7.
# per_user_decode_median = median of llama-server timings.predicted_per_second; aggregate_output = output tokens / wall
# (wall includes prefill). usage: kbench.py PORT C N P   e.g. C1: kbench.py 8015 1 3 256; C4: 8015 4 4 128; C16: 8015 16 16 128
import json, sys, threading, time, urllib.request
port, C, N, P = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
BASE = ("You are a flight instructor. Explain, in careful detail, the aerodynamics of a stall, the factors that "
        "change stall speed (weight, load factor, CG, flaps, power, contamination), and how a student pilot should "
        "recognize and recover from power-on and power-off stalls in a light single-engine airplane. ")
def req(i, out):
    body = json.dumps({"prompt": f"[{i}] " + BASE * 2, "n_predict": P, "ignore_eos": True, "temperature": 0.7,
                       "cache_prompt": False}).encode()
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                   {"Content-Type": "application/json"}), timeout=3600).read())
    tm = r["timings"]; out.append((tm["predicted_n"], tm["predicted_per_second"], tm["prompt_per_second"], tm["prompt_n"], time.time() - t))
res, lock, idx = [], threading.Lock(), [0]
def worker():
    while True:
        with lock:
            if idx[0] >= N: return
            i = idx[0]; idx[0] += 1
        req(i, res)
t0 = time.time(); th = [threading.Thread(target=worker) for _ in range(C)]
[x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t0
dec = sorted(r[1] for r in res); toks = sum(r[0] for r in res)
print(f"RESULT kimi C{C} N{N} P{P} per_user_decode_median {dec[len(dec)//2]:.2f} tok/s min {dec[0]:.2f} max {dec[-1]:.2f} "
      f"aggregate_output {toks/wall:.2f} tok/s wall {wall:.1f}s prefill_median {sorted(r[2] for r in res)[len(res)//2]:.1f} tok/s prompt_n {res[0][3]}", flush=True)
