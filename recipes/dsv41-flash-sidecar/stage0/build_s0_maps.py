#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# Stage 0 maps (RonanLabs 2026-10-10). Scoring = ../e2/build_e2_maps.py (derived from original-el8/dgx-station-gb300-
# research @87af662 m3/tools/build_rowmap.py + build_cal_rowmaps.py, Copyright 2026 Jason Cook, Apache-2.0).
"""Build the Stage 0 hot/cold maps from el8's PUBLIC calibration counts only (PREREG.md note 1).

  build_s0_maps.py --m3 <el8 clone>/deepseek-v4.1-flash/m3 [--out maps]

U300: top-300 in layers 0-20, top-285 in layers 21-39 (= rowmap-mix-v1 there); nested over A0 in every layer.
P45 : placebo, 285 everywhere; layers 0-20 swap A0's 45 lowest-scored hot (ranks 241-285) for the 45 highest-scored
      cold (ranks 286-330); layers 21-39 = A0.
Refuses to write unless: input sha256 pins match, the 285/285 rebuild equals rowmap-mix-v1 on all 40 layers, every
layer is a partition of 0..383 with the declared hot count, U300 is nested over A0 (equal in 21-39), P45 moves exactly
45 in / 45 out per layer 0-20 and nothing in 21-39, and the 45 "in" are ranks 286-330 (U300 adds ranks 286-300).
Writes maps/rowmap-mix-v1.json (A0 byte copy), rowmap-s0-u300.json, rowmap-s0-p45.json, maps/SHA256SUMS.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_e2_maps as e2  # noqa: E402  (PIN, score, build, check, sha, M3, L, E, LO)

# SRC and CAL are kept exactly as at run time so that a rebuild is byte-identical to the maps we served.
SRC = ("original-el8/dgx-station-gb300-research @87af662 deepseek-v4.1-flash/m3/calibration (Apache-2.0); builder "
       "research/advance-dsv41/stage0/build_s0_maps.py (scoring = e2/build_e2_maps.py)")
CAL = "0.6 decode (el8 public D: prose, E: code/QA/reasoning) + 0.4 prefill (el8 public A, B) = rowmap-mix-v1 scoring"


def ranks(sc, l):
    return sorted(range(e2.E), key=lambda e: (-sc[l][e], e))


def build_p45(sc, mix):
    out = {}
    for l in range(e2.L):
        if l in e2.LO:
            r = ranks(sc, l)
            hot = r[:240] + r[285:330]
        else:
            hot = list(mix[str(l)]["hot"])
        out[str(l)] = {"hot": sorted(hot), "cold": sorted(set(range(e2.E)) - set(hot))}
    return out


def check_u300(u, mix):
    for l in range(e2.L):
        h, a = set(u[str(l)]["hot"]), set(mix[str(l)]["hot"])
        if l in e2.LO:
            assert a <= h and len(h - a) == 15, f"U300 layer {l} not nested over A0 (+15)"
        else:
            assert h == a, f"U300 layer {l} differs from A0"


def check_p45(p, mix, u, sc):
    for l in range(e2.L):
        h, a = set(p[str(l)]["hot"]), set(mix[str(l)]["hot"])
        if l in e2.LO:
            r = ranks(sc, l)
            assert len(h - a) == 45 and len(a - h) == 45, f"P45 layer {l}: {len(h - a)} in / {len(a - h)} out"
            assert h - a == set(r[285:330]) and a - h == set(r[240:285]), f"P45 layer {l} wrong swap"
            assert set(u[str(l)]["hot"]) - a == set(r[285:300]), f"U300 layer {l} extras are not ranks 286-300"
        else:
            assert h == a, f"P45 layer {l} differs from A0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "maps"))
    a = ap.parse_args()
    for rel, want in e2.PIN.items():
        got = e2.sha(os.path.join(a.m3, rel))
        if got != want:
            raise SystemExit(f"REFUSED: {rel} sha256 {got} != pinned {want}")
    sc = e2.score(a.m3)
    mix = json.load(open(os.path.join(a.m3, "hook/rowmap-mix-v1.json")))["layers"]
    rebuilt = e2.build(sc, 285, 285)
    diff = [l for l in range(e2.L) if rebuilt[str(l)]["hot"] != sorted(mix[str(l)]["hot"])]
    if diff:
        raise SystemExit(f"REFUSED: 285/285 rebuild differs from rowmap-mix-v1 on layers {diff}")
    print("validation: 285/285 rebuild == rowmap-mix-v1 on all 40 layers")
    u300 = e2.build(sc, 300, 285)
    e2.check(u300, 300, 285)
    check_u300(u300, mix)
    p45 = build_p45(sc, mix)
    e2.check(p45, 285, 285)
    check_p45(p45, mix, u300, sc)
    print("checks: U300 nested over A0 (+15/layer in 0-20, = A0 in 21-39); P45 swaps ranks 241-285 -> 286-330 in 0-20")
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(os.path.join(a.m3, "hook/rowmap-mix-v1.json"), os.path.join(a.out, "rowmap-mix-v1.json"))
    names = ["rowmap-mix-v1.json"]
    for arm, layers, n_lo, note in (
            ("u300", u300, 300, "U300 nested union: top-300 in layers 0-20, A0's top-285 in 21-39"),
            ("p45", p45, 285, "P45 placebo: A0 with ranks 241-285 swapped for 286-330 in layers 0-20")):
        total = sum(len(layers[str(l)]["hot"]) for l in range(e2.L))
        meta = {"calibration": CAL, "source": SRC, "arm": arm.upper(), "note": note, "n_hot_layers_0_20": n_lo,
                "n_hot_layers_21_39": 285, "total_hot_slots": total, "total_hot_slots_mix_v1": 40 * 285,
                "public_only": True}
        name = f"rowmap-s0-{arm}.json"
        json.dump({"layers": layers, "meta": meta}, open(os.path.join(a.out, name), "w"))
        names.append(name)
        moved = sum(len(set(layers[str(l)]["hot"]) ^ set(mix[str(l)]["hot"])) for l in range(e2.L))
        print(f"{name}: total hot slots {total} ({total - 40 * 285:+d} vs mix-v1); experts whose tier differs from A0: "
              f"{moved}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write(f"{e2.sha(os.path.join(a.out, n))}  {n}\n")
    print(open(os.path.join(a.out, "SHA256SUMS")).read(), end="")


if __name__ == "__main__":
    main()
