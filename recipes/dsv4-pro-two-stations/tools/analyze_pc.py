#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""analyze_pc.py -- the two helpers refit.py and mixfit.py import, copied verbatim from our placement-check tool
(2026-10-09). The check itself (live tiers vs rowmap files, cross-rank Grace critical path) needs calibration counts
from private traffic and is not included.

  tiers_from_rowmaps(e2, e3)       -> code[rank][layer, global id]: 0 remote 1 hot 2 warm 3 cold
  predicted_from_counts(cnt, codes) -> per decode token, per rank: expected routes per tier and the Grace share
"""
import json

import numpy as np

E, L, NL = 384, 61, 192


def tiers_from_rowmaps(e2, e3):
    """code[rank][l, g]: 0 remote 1 hot 2 warm 3 cold."""
    e2l = {int(k): set(v["hot"]) for k, v in json.load(open(e2))["layers"].items()}
    e3l = {int(k): set(v["warm"]) for k, v in json.load(open(e3))["layers"].items()}
    codes = {}
    for r in (0, 1):
        c = np.zeros((L, E), int)
        for l in range(L):
            for g in range(r * NL, r * NL + NL):
                c[l, g] = 1 if g in e2l.get(l, ()) else 2 if g in e3l.get(l, ()) else 3
        codes[r] = c
    return codes


def predicted_from_counts(cnt, codes):
    """Per decode token, per rank: expected routes per tier (sum over layers) given per-layer expert frequencies
    cnt [L,E] (any traffic), and expected cold hit layers / unique cold experts at T == 1 (unique == routes)."""
    out = {}
    tok = cnt.sum(1) / 6.0                  # tokens per layer
    for r in (0, 1):
        res = {}
        for i, name in enumerate(("remote", "hot", "warm", "cold")):
            m = codes[r] == i
            res[name] = float(((cnt * m).sum(1) / np.maximum(tok, 1)).sum())
        loc = res["hot"] + res["warm"] + res["cold"]
        res["grace_share_of_local"] = res["cold"] / max(loc, 1e-9)
        out[r] = res
    return out
