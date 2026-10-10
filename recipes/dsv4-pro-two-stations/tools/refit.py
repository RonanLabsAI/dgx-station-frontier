#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""refit.py -- PC (2026-10-09): re-fit the E2 (HBM hot) + E3 (6000 warm) placement to measured decode counts, with the
SAME budgets and rules as the as-run maps:
  E2 (tools/build_rowmap_e2.py): 5,568 hot rows per EP rank, global greedy by score, >= 8 hot and >= 8 cold per layer.
  E3 (tools/build_rowmap_e3.py --e2-rowmap): 2,400 warm rows per rank from E2-cold rows, >= 8 per layer first, then global
     greedy, >= 1 Grace row left per layer.
Score = sum_i w_i * (per-layer-normalised decode counts of source i). Sources are JSON count matrices {layer: [384]}.

  refit.py --src counts-cnn-off.json=1.0 [--src counts-sg-off.json=1.0 ...] --tag v4ppub --outdir DIR [--eval name=file ...]
"""
import argparse
import json
import os
import sys

import numpy as np

L, E, NL = 61, 384, 192
HOT_BUDGET, MIN_HOT, MIN_COLD = 5568, 8, 8
WARM_CAP, WARM_MIN = 2400, 8
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_pc as A  # noqa: E402


def mat(d):
    return np.array([d[str(l)] for l in range(L)], float)


def norm(m):
    return m / np.maximum(m.sum(1, keepdims=True), 1)


def fit_hot(score):
    hot = np.zeros((L, E), bool)
    for r in range(2):
        sl = slice(NL * r, NL * (r + 1))
        s = score[:, sl]
        take = np.zeros((L, NL), bool)
        order = np.argsort(-s, axis=1, kind="stable")
        for l in range(L):
            take[l, order[l, :MIN_HOT]] = True
        rem = HOT_BUDGET - take.sum()
        cand = sorted(((s[l, e], l, e) for l in range(L) for e in range(NL) if not take[l, e]), reverse=True)
        for _, l, e in cand:
            if rem <= 0:
                break
            if NL - take[l].sum() <= MIN_COLD:
                continue
            take[l, e] = True
            rem -= 1
        hot[:, sl] = take
    return hot


def fit_warm(score, hot):
    warm = np.zeros((L, E), bool)
    for r in range(2):
        lo = r * NL
        sub = score[:, lo:lo + NL].copy()
        sub[hot[:, lo:lo + NL]] = -1.0
        chosen = np.zeros_like(sub, bool)
        for i in range(L):
            avail = int((sub[i] >= 0).sum())
            top = [j for j in np.argsort(-sub[i], kind="stable") if sub[i, j] >= 0][:min(WARM_MIN, avail - 1)]
            chosen[i, top] = True
        cap = (sub >= 0).sum(1) - 1
        need = max(0, WARM_CAP - int(chosen.sum()))
        rest = sorted((-sub[i, j], i, j) for i in range(L) for j in range(NL) if not chosen[i, j] and sub[i, j] >= 0)
        for _, i, j in rest:
            if need == 0:
                break
            if chosen[i].sum() < cap[i]:
                chosen[i, j] = True
                need -= 1
        warm[:, lo:lo + NL] = chosen
    return warm


def codes_from(hot, warm):
    out = {}
    for r in range(2):
        c = np.zeros((L, E), int)
        c[:, r * NL:(r + 1) * NL] = 3
        c[warm & (c == 3)] = 2
        c[hot & ((c == 3) | (c == 2))] = 1
        out[r] = c
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", action="append", required=True, help="file.json=weight (layers->[384] decode counts)")
    p.add_argument("--eval", action="append", default=[], help="name=file.json (layers->[384])")
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    score = np.zeros((L, E))
    for s in a.src:
        f, w = s.rsplit("=", 1)
        score += float(w) * norm(mat(json.load(open(f))))
    # tiny deterministic prior so never-seen experts keep the as-run order (E2's own score breaks ties)
    hot = fit_hot(score)
    warm = fit_warm(score, hot)
    codes = codes_from(hot, warm)
    for r in range(2):
        sl = slice(NL * r, NL * (r + 1))
        assert hot[:, sl].sum() == HOT_BUDGET and warm[:, sl].sum() == WARM_CAP, (hot[:, sl].sum(), warm[:, sl].sum())
        assert ((NL - hot[:, sl].sum(1) - warm[:, sl].sum(1)) >= 1).all()
        assert not (hot & warm).any()
    os.makedirs(a.outdir, exist_ok=True)
    e2p = os.path.join(a.outdir, f"rowmap-dsv4pro-{a.tag}.json")
    e3p = os.path.join(a.outdir, f"rowmap-e3-on-e2-{a.tag}.json")
    nh = [hot[:, NL * r:NL * (r + 1)].sum(1).tolist() for r in range(2)]
    nw = [warm[:, NL * r:NL * (r + 1)].sum(1).tolist() for r in range(2)]
    ev = {}
    for e in a.eval:
        nm, f = e.split("=", 1)
        pr = A.predicted_from_counts(mat(json.load(open(f))), codes)
        ev[nm] = {r: {k: round(v, 3) for k, v in pr[r].items()} for r in pr}
    json.dump({"name": os.path.basename(e2p), "budget_per_rank": HOT_BUDGET, "weights": a.src, "per_rank_hot": [HOT_BUDGET] * 2,
               "n_hot_rank0": nh[0], "n_hot_rank1": nh[1], "eval": ev, "source": "PC refit.py 2026-10-09",
               "layers": {str(l): {"hot": [int(x) for x in np.nonzero(hot[l])[0]]} for l in range(L)}}, open(e2p, "w"))
    json.dump({"format": "e3-rowmap/v1", "model": "deepseek-ai/DeepSeek-V4-Pro-0813", "e2_rowmap": os.path.basename(e2p),
               "placement": {"ep_size": 2, "expert_placement": "linear", "offload_gib": 200.0, "grace_layers": list(range(L)),
                             "hbm_layers": [], "tiers": "hot=E2 hot rows (HBM); warm=6000 (listed, from E2 cold rows); cold=E2 cold rows minus warm (Grace)"},
               "source": {"counts": a.src, "capacity_per_rank": WARM_CAP, "min_per_layer": WARM_MIN, "tool": "pc/refit.py"},
               "report": {f"rank{r}": {"warm_experts": WARM_CAP, "per_layer_min": int(min(nw[r])), "per_layer_max": int(max(nw[r]))} for r in range(2)},
               "layers": {str(l): {"warm": [int(x) for x in np.nonzero(warm[l])[0]]} for l in range(L)}}, open(e3p, "w"), indent=1)
    print(json.dumps({"e2": e2p, "e3": e3p, "hot_per_layer_r0": [min(nh[0]), max(nh[0])], "warm_per_layer": [[min(x), max(x)] for x in nw],
                      "eval": ev}, indent=1))


if __name__ == "__main__":
    main()
