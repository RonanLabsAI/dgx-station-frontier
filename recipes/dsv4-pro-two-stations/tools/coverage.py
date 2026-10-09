#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Warm-tier coverage of a route histogram under a rowmap, per EP rank.

  coverage.py ROWMAP COUNTS.json [COUNTS2.json ...]
COUNTS: routing-counts/v1 (E1 GSM8K file with phases, or e3_hook E3_COUNT dumps with "layers").
Reports, per rank and phase: share of all local routes, and of Grace-layer local routes, that land on the 6000.
"""
import json
import sys

import numpy as np

E, N_LOCAL = 384, 192


def main():
    rm = json.load(open(sys.argv[1]))
    grace = rm["placement"]["grace_layers"]
    warm = {int(k): set(v["warm"]) for k, v in rm["layers"].items()}
    for path in sys.argv[2:]:
        d = json.load(open(path))
        phases = d.get("phases") or {"all": d["layers"]}
        for ph, layers in phases.items():
            arr = {int(k): np.asarray(v, float) for k, v in layers.items()}
            for rank in (0, 1):
                lo, hi = rank * N_LOCAL, (rank + 1) * N_LOCAL
                tot = sum(a[lo:hi].sum() for a in arr.values())
                gtot = sum(a[lo:hi].sum() for k, a in arr.items() if k in grace)
                w = sum(a[e] for k, a in arr.items() for e in warm.get(k, ()) if lo <= e < hi)
                print(f"{path.split('/')[-1]} {ph} rank{rank}: warm {w / max(gtot, 1):.3f} of Grace-layer routes, "
                      f"{w / max(tot, 1):.3f} of all local routes (Grace layers carry {gtot / max(tot, 1):.3f})")


if __name__ == "__main__":
    main()
