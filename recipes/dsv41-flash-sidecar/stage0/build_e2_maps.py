#!/usr/bin/env python3
# Derived from original-el8/dgx-station-gb300-research @87af662, deepseek-v4.1-flash/m3/tools/build_rowmap.py and
# build_cal_rowmaps.py (Copyright 2026 Jason Cook, Apache License 2.0; see the repository NOTICE).
# Modified by RonanLabs 2026-10-10 (E2): a different hot count for layers 0-20 and 21-39 at an equal total.
# Modifications Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Build the E2 hot/cold maps from el8's PUBLIC calibration counts only.

  build_e2_maps.py --m3 <el8 clone>/deepseek-v4.1-flash/m3 [--out maps]

Score per (layer, expert) = 0.3 D + 0.3 E (decode text) + 0.2 A + 0.2 B (prefill text) of per-layer route shares:
exactly the weighting of el8's rowmap-mix-v1 (its meta: "0.6 decode (D, E) + 0.4 prefill (A, B)"). Hot = the n
highest-scoring experts of each layer (ties by expert id), n = N_LO for layers 0-20 and N_HI for layers 21-39.

Self-checks (the script refuses to write on failure):
  * at 285/285 the rebuilt map equals rowmap-mix-v1 exactly, every layer (the scoring is el8's, not a lookalike);
  * every layer of every map is a partition of 0..383; hot counts are as declared;
  * input files match their sha256 (el8 @87af662 as cloned 2026-10-10).
Writes maps/rowmap-mix-v1.json (A0, byte copy), maps/rowmap-e2-x315.json, maps/rowmap-e2-x330.json, maps/SHA256SUMS.
No other counts are read: the maps never see our real-text counts or random ids.
"""
import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
M3 = None   # pass --m3: a clone of github.com/original-el8/dgx-station-gb300-research (7699ae54 or 87af662; the inputs are sha-pinned)
L, E, LO = 40, 384, range(0, 21)
W = {"D": 0.3, "E": 0.3, "A": 0.2, "B": 0.2}
ARMS = {"x315": (315, 252), "x330": (330, 235)}   # ADVANCE-DSV41.md section 4, E2 arms
PIN = {
    "calibration/counts-A.json": "e7e7c64c7391c2b57c4c0badfdfced974913c19f9c2dbe0b72f89fe9edd5ca0f",
    "calibration/counts-B.json": "f30867c3bb3e6e15a899cb9f261df662212cbf6c44e8f1f4066741711fda3219",
    "calibration/counts-D.json": "60df92be850595519461700d757ba31a6dad3819a619ef4eb3771b9598e2ce86",
    "calibration/counts-E.json": "faa136fe229085fce8f5786caf95c8ff21a47f66b8cb740d974b0e9130d1cb94",
    "hook/rowmap-mix-v1.json": "2be120dff20761eb231c0f5d409aea87c40d79ece99324d46cec810c6ca4e60d",
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def counts(m3, name):
    return {int(k): v for k, v in json.load(open(os.path.join(m3, f"calibration/counts-{name}.json")))["layers"].items()}


def score(m3):
    norm = {}
    for k in W:
        c = counts(m3, k)
        norm[k] = {l: [x / max(1, sum(c[l])) for x in c[l]] for l in range(L)}
    return {l: [sum(W[k] * norm[k][l][e] for k in W) for e in range(E)] for l in range(L)}


def build(sc, n_lo, n_hi):
    out = {}
    for l in range(L):
        n = n_lo if l in LO else n_hi
        order = sorted(range(E), key=lambda e: (-sc[l][e], e))
        out[str(l)] = {"hot": sorted(order[:n]), "cold": sorted(order[n:])}
    return out


def check(layers, n_lo, n_hi):
    for l in range(L):
        h, c = set(layers[str(l)]["hot"]), set(layers[str(l)]["cold"])
        assert h | c == set(range(E)) and not h & c, f"layer {l} not a partition"
        assert len(h) == (n_lo if l in LO else n_hi), f"layer {l} hot count {len(h)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "maps"))
    a = ap.parse_args()
    for rel, want in PIN.items():
        got = sha(os.path.join(a.m3, rel))
        if got != want:
            raise SystemExit(f"REFUSED: {rel} sha256 {got} != pinned {want}")
    sc = score(a.m3)
    mix = json.load(open(os.path.join(a.m3, "hook/rowmap-mix-v1.json")))["layers"]
    rebuilt = build(sc, 285, 285)
    diff = [l for l in range(L) if rebuilt[str(l)]["hot"] != sorted(mix[str(l)]["hot"])]
    if diff:
        raise SystemExit(f"REFUSED: 285/285 rebuild differs from rowmap-mix-v1 on layers {diff}")
    print("validation: 285/285 rebuild == rowmap-mix-v1 on all 40 layers")
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(os.path.join(a.m3, "hook/rowmap-mix-v1.json"), os.path.join(a.out, "rowmap-mix-v1.json"))
    names = ["rowmap-mix-v1.json"]
    for arm, (n_lo, n_hi) in ARMS.items():
        layers = build(sc, n_lo, n_hi)
        check(layers, n_lo, n_hi)
        total = 21 * n_lo + 19 * n_hi
        added = sum(len(set(layers[str(l)]["hot"]) - set(mix[str(l)]["hot"])) for l in range(L))
        dropped = sum(len(set(mix[str(l)]["hot"]) - set(layers[str(l)]["hot"])) for l in range(L))
        # The meta strings are kept exactly as at run time so that a rebuild is byte-identical to the maps we served.
        meta = {"calibration": "0.6 decode (el8 public D: prose, E: code/QA/reasoning) + 0.4 prefill (el8 public A, "
                               "B) = rowmap-mix-v1 scoring; hot count per layer: layers 0-20 = %d, 21-39 = %d" % (n_lo, n_hi),
                "source": "original-el8/dgx-station-gb300-research @87af662 deepseek-v4.1-flash/m3/calibration "
                          "(Apache-2.0); builder research/advance-dsv41/e2/build_e2_maps.py",
                "n_hot_layers_0_20": n_lo, "n_hot_layers_21_39": n_hi, "total_hot_slots": total,
                "total_hot_slots_mix_v1": 40 * 285, "public_only": True, "arm": arm.upper()}
        name = f"rowmap-e2-{arm}.json"
        json.dump({"layers": layers, "meta": meta}, open(os.path.join(a.out, name), "w"))
        names.append(name)
        print(f"{name}: {n_lo}/{n_hi}, total hot slots {total} (mix-v1 {40 * 285}, {total - 40 * 285:+d}), "
              f"vs mix-v1: {added} experts made hot, {dropped} made cold")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write(f"{sha(os.path.join(a.out, n))}  {n}\n")
    print(open(os.path.join(a.out, "SHA256SUMS")).read(), end="")


if __name__ == "__main__":
    main()
