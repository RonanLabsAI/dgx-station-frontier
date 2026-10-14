#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""build_pool_k3.py RT_POOL_DIR OUT_DIR -- public real-text prompt slices for the Kimi K3 follow-ups.

Source: the pool written by ../../dsv41-flash-sidecar/bench/realtext/build_realtext_pool.py (CNN/DailyMail 3.0.0 test
articles, abisee/cnn_dailymail @ 96df5e68, concatenated to ~8,190 DS-V4.1 tokens + a fixed summarise-and-compare
instruction ("long"), and ShareGPT V3 first human turns, 12-498 tokens ("short"); seed 20261009). Those prompts
are pre-rendered in DeepSeek-V4.1 chat format; here only the user CONTENT is kept (the K3 server renders its own
chat template; K3's tokenizer differs, so a long prompt is ~8K K3 tokens, not exactly 8,192 -- usage reports it).

Slices (disjoint, deterministic; every arm uses the SAME slices, so A and B see identical prompts; within one boot each
slice is sent once, so cross-run prefix reuse is impossible by construction; the server-side 2% guard still runs):
  long/eval-c1-r{1,2,3}-try{1,2}.jsonl   3 prompts each (+ -warm: 1)     real-text C1 x 3 reps
  long/eval-c16-try{1,2}.jsonl           32 prompts (+ -warm: 4)         real-text C16 (2 waves, as catid NP=2C)
  long/fit.jsonl                         the rest (capture traffic for K-E2/K-E3 placement)
  short/fit.jsonl                        all ShareGPT prompts (capture only; natural length outputs)
"""
import hashlib
import json
import os
import random
import sys

BOS, USER, ASSIST, THINK_END = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"
SEED = 20261009


def contents(d):
    out, seen = [], set()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(d, fn)):
            p = json.loads(line)["prompt"]
            if not (p.startswith(BOS + USER) and p.endswith(ASSIST + THINK_END)):
                raise SystemExit(f"unexpected chat format in {fn}")
            c = p[len(BOS + USER):-len(ASSIST + THINK_END)]
            key = c[:256]
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    rng = random.Random(SEED)
    long_c, short_c = contents(os.path.join(src, "long")), contents(os.path.join(src, "short"))
    rng.shuffle(long_c)
    rng.shuffle(short_c)
    plan = [(f"eval-c1-r{r}-try{t}", 3, 1) for t in (1, 2) for r in (1, 2, 3)] + \
           [(f"eval-c16-try{t}", 32, 4) for t in (1, 2)]
    os.makedirs(os.path.join(out, "long"), exist_ok=True)
    os.makedirs(os.path.join(out, "short"), exist_ok=True)
    i = 0
    manifest = {"seed": SEED, "source": os.path.basename(os.path.abspath(src)), "files": {}}

    def write(path, items):
        with open(path, "w") as f:
            for c in items:
                f.write(json.dumps({"content": c, "id": hashlib.sha1(c.encode()).hexdigest()[:12]}) + "\n")
        manifest["files"][os.path.relpath(path, out)] = {
            "n": len(items), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}

    for label, n, w in plan:
        write(os.path.join(out, "long", f"{label}.jsonl"), long_c[i:i + n])
        i += n
        write(os.path.join(out, "long", f"{label}-warm.jsonl"), long_c[i:i + w])
        i += w
    write(os.path.join(out, "long", "fit.jsonl"), long_c[i:])
    write(os.path.join(out, "short", "fit.jsonl"), short_c)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({"long_total": len(long_c), "long_fit": len(long_c) - i, "short_fit": len(short_c),
                      "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
