#!/usr/bin/env python3
"""Public prompt pools for the TensorFold B1 probe (GLM-5.3-Flash, 2026-10-10).

Same pinned sources and hygiene as rt/build_realtext_pool.py, but written as plain chat `content` (TensorFold renders
GLM's chat template server-side), measured with GLM's tokenizer:

  long8k / long32k : CNN/DailyMail 3.0.0 test (abisee/cnn_dailymail @ 96df5e68). Articles in a seeded order are joined
                     ("Article k:\\n<text>") until the content is TARGET tokens, last article trimmed on a token
                     boundary, then a fixed summarise instruction. Every article used once across the whole pool.
  short            : ShareGPT V3 (anon8231489123/ShareGPT_Vicuna_unfiltered @ 192ab218), first human turn, 8-512 tokens.

Hygiene: every prompt distinct; no two prompts in the pool share their first PREFIX_UNIQ (64) tokens, so TensorFold's
kept-prompt reuse cannot hit between prompts (the bench also checks usage.cached_tokens per request). Each bench step
gets its own disjoint slot, warm-ups included.

usage: build_pool_glm.py CNN_TEST_PARQUET SHAREGPT_JSON GLM_TOKENIZER_JSON OUTDIR [--seed 20261010]
"""
import argparse
import hashlib
import json
import os
import random

import pandas as pd
from tokenizers import Tokenizer

PREFIX_UNIQ = 64
LONG_INSTR = ("\n\nFor each article above, write a detailed summary of its main events, people and claims. "
              "Then explain how the articles relate to one another.")
SHORT_SUFFIX = ""
# (slot label, pool, count). C1 = 3 runs x 5 prompts; C4 = 4 x 4 prompts; prefill = 3 prompts at each length.
PLAN = [
    ("warm-short", "short", 4), ("warm-long8k", "long8k", 1),
    ("c1-r1", "short", 5), ("c1-r2", "short", 5), ("c1-r3", "short", 5),
    ("c4-r1", "short", 16),
    ("c1long-r1", "long8k", 3),
    ("prefill8k", "long8k", 3), ("warm-long32k", "long32k", 1), ("prefill32k", "long32k", 3),
    ("spare-short", "short", 24), ("spare-long8k", "long8k", 3),
    # appended 2026-10-10 for the speed-up A/B (earlier slots unchanged: pools are consumed in order)
    ("warm-c16", "short", 16), ("c16-r1", "short", 64), ("prof-warm", "short", 2), ("prof", "short", 1),
]
TARGET = {"long8k": 8192, "long32k": 32768}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnn"), ap.add_argument("sharegpt"), ap.add_argument("tokenizer"), ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    rng = random.Random(a.seed)
    seen_prefix = set()

    def unique(text):
        ids = tok.encode(text, add_special_tokens=False).ids
        key = tuple(ids[:PREFIX_UNIQ])
        if key in seen_prefix:
            return None
        seen_prefix.add(key)
        return ids

    arts = pd.read_parquet(a.cnn)["article"].tolist()
    rng.shuffle(arts)
    art_i = 0
    instr_n = len(tok.encode(LONG_INSTR, add_special_tokens=False).ids)

    def long_prompt(target):
        nonlocal art_i
        parts, n, k = [], 0, 1
        while n < target - instr_n:
            body = f"Article {k}:\n{arts[art_i]}\n\n"
            art_i += 1
            ids = tok.encode(body, add_special_tokens=False).ids
            room = target - instr_n - n
            if len(ids) > room:
                body = tok.decode(ids[:room])
                ids = ids[:room]
            parts.append(body)
            n += len(ids)
            k += 1
        return "".join(parts).rstrip() + LONG_INSTR

    sg = json.load(open(a.sharegpt))
    rng.shuffle(sg)
    shorts = []
    for conv in sg:
        turns = conv.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        n = len(tok.encode(text, add_special_tokens=False).ids)
        if 8 <= n <= 512 and unique(text) is not None:
            shorts.append(text)
        if len(shorts) >= 200:
            break

    os.makedirs(a.out, exist_ok=True)
    files, stats = {}, {}
    si = 0
    for label, pool, count in PLAN:
        rows = []
        for _ in range(count):
            if pool == "short":
                text = shorts[si]
                si += 1
            else:
                while True:
                    text = long_prompt(TARGET[pool])
                    if unique(text) is not None:
                        break
            n = len(tok.encode(text, add_special_tokens=False).ids)
            rows.append({"content": text, "content_tokens": n})
        path = os.path.join(a.out, f"{label}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"{label}.jsonl"] = sha(path)
        ns = [r["content_tokens"] for r in rows]
        stats[label] = {"pool": pool, "n": len(rows), "min_tokens": min(ns), "max_tokens": max(ns)}
    manifest = {
        "seed": a.seed, "prefix_uniq_tokens": PREFIX_UNIQ, "plan": PLAN, "long_instruction": LONG_INSTR,
        "sources": {
            "long": "abisee/cnn_dailymail @ 96df5e686bee6baa90b8bee7c28b81fa3fa6223d 3.0.0/test-00000-of-00001.parquet",
            "short": "anon8231489123/ShareGPT_Vicuna_unfiltered @ 192ab2185289094fc556ec8ce5ce1e8e587154ca "
                     "ShareGPT_V3_unfiltered_cleaned_split.json",
            "tokenizer": "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold @ 078455ff tokenizer.json",
            "sha256": {"test.parquet": sha(a.cnn), "sg.json": sha(a.sharegpt), "tokenizer.json": sha(a.tokenizer)},
        },
        "files": files, "stats": stats, "cnn_articles_used": art_i,
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
