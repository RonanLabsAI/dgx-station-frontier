#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""flush.py BASE -- PC: one eager forward (unique ~3K-token random prompt, max_tokens 1, no decode step) so the E3 hook
copies its tier counters to host; prints the start time. Its own routes land in bucket 2 (T > 64), never in decode."""
import json, random, sys, time, urllib.request
t0 = time.time()
words = " ".join(f"w{random.randint(0, 10**9)}" for _ in range(900))
body = {"model": "deepseek-v4-pro", "prompt": words, "max_tokens": 1, "temperature": 0}
req = urllib.request.Request(sys.argv[1] + "/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.load(r)
print(json.dumps({"t_flush": t0, "prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "dt": round(time.time() - t0, 2)}))
