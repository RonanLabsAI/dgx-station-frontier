#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""route_snap.py PORT SERVED OUTDIR LABEL -- snapshot el8's MEGA_COUNT route counter on this host.

Same method as pass2/tools/mega_snapshot.py / el8 calibrate_routes.py:flush_snapshot, but local and with a REAL-TEXT
flush: the hook copies device counts to host only on eager passes (at most every 10 s) and a thread rewrites
/prof/route-counts.json every 20 s. So: wait 11 s, force one eager pass with a unique ~410-token CNN snippet
(> the largest cudagraph size 128; max_tokens 1), then poll the file until its "time" is newer than the pass.
The snippet's own routes (~410 real-text tokens) land in this or the next interval; negligible against >= 40K.
Flush snippets: POOL/flush/NN.txt, used in order via a counter file (never reused in the session).
Never fails the run: on timeout it prints a NO-SNAP line and exits 0.
"""
import json
import os
import sys
import time
import urllib.request

port, served, outdir, label = sys.argv[1:5]
counts = os.path.expanduser(os.environ.get("COUNTS", "prof/route-counts.json"))
idxf = os.path.join(outdir, "..", ".flushidx")
os.makedirs(outdir, exist_ok=True)
i = int(open(idxf).read()) if os.path.exists(idxf) else 0
open(idxf, "w").write(str(i + 1))
flush = open(os.path.join(os.environ.get("POOL", "pool"), "flush", f"{i % 64:02d}.txt")).read()
time.sleep(11)
t0 = time.time()
body = {"model": served, "prompt": flush, "max_tokens": 1, "temperature": 0}
try:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
except Exception as exc:  # diagnostic only
    print(f"snap {label}: flush failed {exc!r}")
for _ in range(40):
    try:
        d = json.load(open(counts))
    except Exception:
        d = None
    if d and d.get("time", 0) > t0 + 1:
        d["label"], d["flush_idx"], d["flush_t0"] = label, i, t0
        json.dump(d, open(f"{outdir}/{label}.json", "w"))
        tot = sum(sum(v) for v in d["layers"].values())
        print(f"snap {label}: {tot:,} cumulative routes, dump time {time.strftime('%H:%M:%S', time.localtime(d['time']))}, flush #{i}")
        sys.exit(0)
    time.sleep(5)
print(f"snap {label}: NO-SNAP (no fresh dump after 200 s; is MEGA_COUNT set and /prof mounted?)")
