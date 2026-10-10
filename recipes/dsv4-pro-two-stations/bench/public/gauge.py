#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""gauge.py BASE OUT.tsv [PERIOD=2] -- V4P: sample vLLM /metrics every PERIOD s (epoch, num_requests_running,
num_requests_waiting, generation_tokens_total, kv usage) so each bench window's REAL concurrency is on record
(KV 9.05 GiB = ~32K tokens holds only ~3.5 requests of 8K+1K; "C16" on 8K prompts is KV-capped).
gauge.py --window OUT.tsv T0 T1 -> mean/max running and waiting inside [T0, T1]."""
import sys
import time
import urllib.request

KEYS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:generation_tokens_total", "vllm:kv_cache_usage_perc")


def scrape(base):
    v = {k: 0.0 for k in KEYS}
    with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            if name in v:
                try:
                    v[name] += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return v


def window(path, t0, t1):
    rows = []
    for line in open(path):
        p = line.split("\t")
        if len(p) >= 3 and t0 <= float(p[0]) <= t1:
            rows.append((float(p[1]), float(p[2])))
    if not rows:
        return "running n/a"
    run = [r for r, _ in rows]; wt = [w for _, w in rows]
    return f"running mean {sum(run) / len(run):.1f} max {max(run):.0f} waiting mean {sum(wt) / len(wt):.1f}"


if __name__ == "__main__":
    if sys.argv[1] == "--window":
        print(window(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
        sys.exit(0)
    base, out = sys.argv[1], sys.argv[2]
    period = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    with open(out, "a") as f:
        while True:
            try:
                v = scrape(base)
                f.write(f"{time.time():.2f}\t" + "\t".join(f"{v[k]:.4g}" for k in KEYS) + "\n"); f.flush()
            except Exception:
                pass
            time.sleep(period)
