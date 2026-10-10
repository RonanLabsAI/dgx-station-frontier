#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""mixfit.py -- V4P (2026-10-09): build the MIXED placement from training-slice route captures.

Input: a PC-format snapshot dir (pc/pc-bench.sh layout: <name>-r{0,1}.json written by the E3_COUNT hook of commit
b09fb2d, order in index.txt). Each capture workload W was run between snapshots "pre-W" and "W"; its counts are the
delta (cumulative counters) in decode buckets 0 (T == 1) + 1 (2 <= T <= 64: decode steps at C2-C16; also prefills of
<= 64-token prompts, < ~3% of rows here). Bucket 2 (T > 64: prefill chunks, mixed steps) is excluded, as in PC.
Rank 0 and rank 1 histograms are over all 384 global ids (identical routing); they are averaged.

Writes OUT/counts-<W>.json ({layer: [384]}), runs pc/refit.py with every workload at equal weight (each normalised per
layer, so the score is the mean routing distribution over the workloads given, e.g. CNN/DM and ShareGPT, thinking on/off),
and OUT/mixfit-summary.json: per workload, predicted Grace (cold) routes per token per rank and Grace share under the
OLD map (--old-e2/--old-e3: e.g. the maps the capture server ran) and the new map. Predictions on training slices are in-sample by construction; the held-out test is
the measured run that follows.

usage: mixfit.py SNAPDIR OUTDIR --work cnn-off,cnn-on,sg-off,sg-on [--tag v4ppub] --old-e2 F --old-e3 F [--pcdir DIR]
(--pcdir = the folder holding refit.py and analyze_pc.py: this recipe's tools/)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

L, E = 61, 384


def counts(snapdir, name, r):
    j = json.load(open(os.path.join(snapdir, f"{name}-r{r}.json")))
    bb = j["by_bucket"]
    return sum(np.array([bb[b][str(l)] for l in range(L)], float) for b in ("0", "1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapdir"); ap.add_argument("outdir")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", default="v4pmix")
    ap.add_argument("--pcdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pc"))
    ap.add_argument("--old-e2", required=True); ap.add_argument("--old-e3", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.pcdir)
    import analyze_pc as A  # noqa: E402
    os.makedirs(a.outdir, exist_ok=True)
    work = a.work.split(",")
    files, info = {}, {}
    for w in work:
        d = sum(counts(a.snapdir, w, r) - counts(a.snapdir, f"pre-{w}", r) for r in (0, 1)) / 2.0
        assert (d >= 0).all(), w
        tok = float(d.sum(1).mean() / 6.0)
        info[w] = {"decode_tokens_per_layer": round(tok)}
        if tok < 500:
            print(f"WARNING {w}: only {tok:.0f} decode tokens captured", flush=True)
        f = os.path.join(a.outdir, f"counts-{w}.json")
        json.dump({str(l): d[l].tolist() for l in range(L)}, open(f, "w"))
        files[w] = f
    cmd = [sys.executable, os.path.join(a.pcdir, "refit.py"), "--tag", a.tag, "--outdir", a.outdir]
    for w in work:
        cmd += ["--src", f"{files[w]}=1.0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(a.outdir, "refit.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
    old = A.tiers_from_rowmaps(a.old_e2, a.old_e3)
    new = A.tiers_from_rowmaps(os.path.join(a.outdir, f"rowmap-dsv4pro-{a.tag}.json"),
                               os.path.join(a.outdir, f"rowmap-e3-on-e2-{a.tag}.json"))
    summ = {"tag": a.tag, "work": info, "pred": {}}
    for w in work:
        c = np.array([json.load(open(files[w]))[str(l)] for l in range(L)], float)
        po, pn = A.predicted_from_counts(c, old), A.predicted_from_counts(c, new)
        summ["pred"][w] = {m: {f"r{k}": {"cold": round(v["cold"], 2), "warm": round(v["warm"], 2),
                                         "grace_share": round(v["grace_share_of_local"], 4)} for k, v in p.items()}
                           for m, p in (("old", po), ("mixed", pn))}
    json.dump(summ, open(os.path.join(a.outdir, "mixfit-summary.json"), "w"), indent=1)
    for w in work:
        p = summ["pred"][w]
        print(f"{w}: Grace routes/token/rank old {p['old']['r0']['cold']}/{p['old']['r1']['cold']} -> mixed "
              f"{p['mixed']['r0']['cold']}/{p['mixed']['r1']['cold']} (tokens {info[w]['decode_tokens_per_layer']})")


if __name__ == "__main__":
    main()
