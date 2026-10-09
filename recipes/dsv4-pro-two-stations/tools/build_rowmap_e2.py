# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""build_rowmap_e2.py -- E2 hot-expert rowmap for DS-V4-Pro TP2+EP2 (61 layers x 384 experts; EP rank r owns
experts [192r, 192r+192)). Same algorithm as the map used for our E2/E3 numbers; that map itself is not published
because it was fitted to a private workload (see the recipe README).

Score per (layer, expert) = W_DOMAIN * decode_share(domain fit split) + W_GSM * decode_share(GSM8K)
                            + PRE_W * mean(prefill shares)          (shares normalised per layer per source)
Each EP rank gets BUDGET hot rows (default 5568 = E1's 29 HBM-resident layers x 192), allocated GLOBALLY across its
61 layers by score (greedy), with at least MIN_HOT hot and MIN_COLD cold rows per layer.
Cold-share evaluation on the held-out split: positional (E1 stock: layers 0..31 fully in Grace), this rowmap, and the
oracle (the held-out split's own top rows, same per-layer counts).

Inputs (routing-counts JSON from tools/routes_calib.py and tools/gsm8k200_v4.py with ROUTES=...):
  DOMAIN.json  {"splits": {"fit": {"decode": {...}, "prefill": {...}}, "heldout": {...}}}   per-layer [384] counts
  GSM.json     {"phases": {"decode": {...}, "prefill": {...}}}   e.g. calibration/routes-e1-gsm8k100.json
With no domain file (DOMAIN=-), the map is fitted to GSM8K alone and evaluated in-sample only.

usage: python3 build_rowmap_e2.py out.json DOMAIN.json|- GSM.json [BUDGET]
"""
import json
import sys

import numpy as np

OUT, DOM, GSM = sys.argv[1:4]
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 5568
L, E, NL = 61, 384, 192
W = {'domain_fit': 0.7, 'gsm': 0.3}; PRE_W = 0.05; MIN_HOT, MIN_COLD = 8, 8
UVA_LAYERS = 32  # E1 stock: layers 0..31 local experts in Grace (200.81 GiB / 6.28 GiB per layer)


def mat(d):
    return np.array([d[str(l)] for l in range(L)], dtype=np.float64)


gsm = json.load(open(GSM))['phases']
gd, gp = mat(gsm['decode']), mat(gsm['prefill'])
norm = lambda m: m / np.maximum(m.sum(1, keepdims=True), 1)  # noqa: E731
if DOM != '-':
    sp = json.load(open(DOM))['splits']
    fd, fp = mat(sp['fit']['decode']), mat(sp['fit']['prefill'])
    hd, hp = mat(sp['heldout']['decode']), mat(sp['heldout']['prefill'])
    score = W['domain_fit'] * norm(fd) + W['gsm'] * norm(gd) + PRE_W * (norm(fp) + norm(gp)) / 2
else:
    W = {'gsm': 1.0}
    fd = fp = hd = hp = None
    score = norm(gd) + PRE_W * norm(gp)

hot = np.zeros((L, E), dtype=bool)
for r in range(2):
    sl = slice(NL * r, NL * (r + 1))
    s = score[:, sl].copy()
    take = np.zeros((L, NL), dtype=bool)
    order = np.argsort(-s, axis=1)
    for l in range(L):
        take[l, order[l, :MIN_HOT]] = True
    rem = BUDGET - take.sum()
    cand = [(s[l, e], l, e) for l in range(L) for e in range(NL) if not take[l, e]]
    cand.sort(reverse=True)
    for _, l, e in cand:
        if rem <= 0:
            break
        if NL - take[l].sum() <= MIN_COLD:
            continue
        take[l, e] = True
        rem -= 1
    hot[:, sl] = take


def cold_share(counts, hotmask):
    return float((counts * ~hotmask).sum() / counts.sum())


pos = np.zeros((L, E), dtype=bool)
pos[UVA_LAYERS:, :] = True
ev = {'gsm_decode(in-sample)': {'positional_E1': round(cold_share(gd, pos), 4), 'rowmap': round(cold_share(gd, hot), 4)}}
if hd is not None:
    oracle = np.zeros((L, E), dtype=bool)
    for r in range(2):
        sl = slice(NL * r, NL * (r + 1))
        for l in range(L):
            k = int(hot[l, sl].sum())
            idx = np.argsort(-hd[l, sl])[:k]
            oracle[l, NL * r + idx] = True
    for name, c in (('heldout_decode', hd), ('heldout_prefill', hp), ('fit_decode(in-sample)', fd)):
        ev[name] = {'positional_E1': round(cold_share(c, pos), 4), 'rowmap': round(cold_share(c, hot), 4)}
    ev['heldout_decode']['oracle'] = round(cold_share(hd, oracle), 4)
    ev['heldout_decode']['regret_x'] = round(ev['heldout_decode']['rowmap'] / max(ev['heldout_decode']['oracle'], 1e-9), 2)
per_rank = [int(hot[:, NL * r:NL * (r + 1)].sum()) for r in range(2)]
nh = hot[:, :NL].sum(1).tolist(), hot[:, NL:].sum(1).tolist()
out = {'name': OUT.rsplit('/', 1)[-1], 'budget_per_rank': BUDGET, 'weights': W, 'prefill_w': PRE_W, 'per_rank_hot': per_rank,
       'eval_cold_share': ev, 'n_hot_rank0': nh[0], 'n_hot_rank1': nh[1],
       'layers': {str(l): {'hot': [int(e) for e in np.nonzero(hot[l])[0]]} for l in range(L)}}
json.dump(out, open(OUT, 'w'))
print(json.dumps({k: out[k] for k in ('per_rank_hot', 'eval_cold_share')}, indent=1))
print('hot/layer rank0 min/max', min(nh[0]), max(nh[0]), ' rank1', min(nh[1]), max(nh[1]), ' layers0-2 r0', nh[0][:3])
