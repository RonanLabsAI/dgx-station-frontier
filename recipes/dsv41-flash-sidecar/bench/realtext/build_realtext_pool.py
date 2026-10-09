#!/usr/bin/env python3
"""Build the public real-text prompt pools for the DS-V4.1-Flash sidecar A/B (RT, 2026-10-09).

Two variants, both rendered in DeepSeek-V4.1 chat format with thinking OFF
(`<|begin_of_sentence|><|User|>{content}<|Assistant|></think>`, i.e. encoding.py thinking_mode="chat"):

  long  : CNN/DailyMail 3.0.0 *test* split (abisee/cnn_dailymail @ 96df5e68, Apache-2.0). Articles in a seeded random
          order are concatenated ("Article k:\n<text>") until the prompt is exactly TARGET (8,192) tokens, the last article
          trimmed on a token boundary, followed by a fixed summarise-and-compare instruction. Every article is used once.
  short : ShareGPT V3 (anon8231489123/ShareGPT_Vicuna_unfiltered @ 192ab218, ShareGPT_V3_unfiltered_cleaned_split.json).
          First human turn of each conversation, seeded random order, 8-512 tokens, as written (natural prompts).

Hygiene: every prompt in a pool is distinct, and no two prompts in a pool share their first PREFIX_UNIQ (64) tokens
after the 2-token chat header, so the server's prefix cache cannot hit between prompts (block-level hits need >= 1
full block in common). Each bench invocation gets its own disjoint slice (slot) of the pool; both arms use the SAME
slices, so A and B see identical prompts. Each slot holds 2 tries (the 2% prefix-hit guard retries once).

Usage: build_realtext_pool.py CNN_TEST_PARQUET SHAREGPT_JSON TOKENIZER_JSON OUTDIR [--seed 20261009]
Writes OUTDIR/{long,short}/<label>-try<t>[-warm].jsonl ({"prompt": str, "output_tokens": int}), OUTDIR/flush/*.txt
(unique ~400-token CNN snippets for route-counter snapshots) and OUTDIR/manifest.json (sources, seed, slot plan,
per-file sha256, token statistics, source ids per prompt).
"""
import argparse
import hashlib
import json
import os
import random

import pandas as pd
from tokenizers import Tokenizer

BOS, USER, ASSIST, THINK_END = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"
TARGET = 8192
PREFIX_UNIQ = 64
LONG_INSTR = ("\n\nFor each article above, write a detailed summary of its main events, people and claims. "
              "Then explain how the articles relate to one another.")
# slot plan, identical for every variant and both arms: (label, warm-up prompts, measured prompts)
PLAN = [("decode-c1-r1", 1, 5), ("decode-c1-r2", 1, 5), ("decode-c1-r3", 1, 5), ("decode-c16-r1", 4, 80),
        ("decode-c32-r1", 4, 160)]
TRIES = 2


def chat(content):
    return f"{BOS}{USER}{content}{ASSIST}{THINK_END}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnn")
    ap.add_argument("sharegpt")
    ap.add_argument("tok")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tok)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    need = sum(TRIES * (w + n) for _, w, n in PLAN)
    head = ntok(chat(""))  # 4 tokens: BOS, User, Assistant, </think>

    # ---------------- long: CNN/DM test, exact 8,192-token prompts ----------------
    df = pd.read_parquet(a.cnn)
    order = list(range(len(df)))
    random.Random(a.seed).shuffle(order)
    instr_t = ntok(LONG_INSTR)
    long_pool, used, seen_pref = [], 0, set()
    it = iter(order)
    flush = []
    while len(long_pool) < need:
        parts, ids, k = [], [], 0
        budget = TARGET - head - instr_t
        while True:
            i = next(it)
            text = " ".join(str(df.iloc[i]["article"]).split())
            piece = f"{'' if k == 0 else chr(10) + chr(10)}Article {k + 1}:\n{text}"
            t = ntok("".join(parts) + piece)
            if t <= budget:
                parts.append(piece); ids.append(str(df.iloc[i]["id"])); k += 1
                if t == budget:
                    break
                continue
            # trim this article on a token boundary to land exactly on TARGET (re-encode can merge; walk down)
            enc = tok.encode(piece, add_special_tokens=False)
            cut = budget - ntok("".join(parts))
            while cut > 0:
                cand = tok.decode(enc.ids[:cut])
                if ntok("".join(parts) + cand) <= budget:
                    break
                cut -= 1
            if cut > 0:
                parts.append(cand); ids.append(str(df.iloc[i]["id"]) + f":trim{cut}")
            break
        body = "".join(parts)
        p = chat(body + LONG_INSTR)
        pid = tok.encode(p, add_special_tokens=False).ids
        key = tuple(pid[2:2 + PREFIX_UNIQ])
        if key in seen_pref:
            continue
        seen_pref.add(key)
        long_pool.append({"prompt": p, "tokens": len(pid), "src": ids})
    # flush snippets for route-counter snapshots: next unused articles, first ~400 tokens, unique
    while len(flush) < 64:
        i = next(it)
        enc = tok.encode(" ".join(str(df.iloc[i]["article"]).split()), add_special_tokens=False)
        if len(enc.ids) < 420:
            continue
        flush.append({"text": chat(tok.decode(enc.ids[:400]) + "\n\nSummarise this in one sentence."),
                      "src": str(df.iloc[i]["id"])})

    # ---------------- short: ShareGPT first human turns, natural ----------------
    sg = json.load(open(a.sharegpt))
    order = list(range(len(sg)))
    random.Random(a.seed).shuffle(order)
    short_pool, seen_pref, seen_txt = [], set(), set()
    for i in order:
        conv = sg[i].get("conversations") or []
        if not conv or conv[0].get("from") != "human":
            continue
        txt = (conv[0].get("value") or "").strip()
        if not txt or txt in seen_txt:
            continue
        p = chat(txt)
        pid = tok.encode(p, add_special_tokens=False).ids
        if not (8 <= len(pid) - head <= 512):
            continue
        key = tuple(pid[2:2 + PREFIX_UNIQ])
        if key in seen_pref:
            continue
        seen_txt.add(txt); seen_pref.add(key)
        short_pool.append({"prompt": p, "tokens": len(pid), "src": [str(sg[i].get("id"))]})
        if len(short_pool) >= need:
            break

    # ---------------- slots ----------------
    man = {"seed": a.seed, "target_long_tokens": TARGET, "prefix_uniq_tokens": PREFIX_UNIQ, "tries": TRIES,
           "plan": PLAN, "chat_format": "DeepSeek-V4.1 encoding.py thinking_mode=chat (thinking off), pre-rendered; "
                                        "bench uses --skip-chat-template",
           "sources": {"long": "abisee/cnn_dailymail @ 96df5e686bee6baa90b8bee7c28b81fa3fa6223d 3.0.0/test-00000-of-00001.parquet",
                       "short": "anon8231489123/ShareGPT_Vicuna_unfiltered @ 192ab2185289094fc556ec8ce5ce1e8e587154ca ShareGPT_V3_unfiltered_cleaned_split.json",
                       "sha256": {os.path.basename(a.cnn): hashlib.sha256(open(a.cnn, "rb").read()).hexdigest(),
                                  os.path.basename(a.sharegpt): hashlib.sha256(open(a.sharegpt, "rb").read()).hexdigest(),
                                  "tokenizer.json": hashlib.sha256(open(a.tok, "rb").read()).hexdigest()}},
           "long_instruction": LONG_INSTR, "files": {}, "stats": {}}
    for name, pool, olen in (("long", long_pool, 1024), ("short", short_pool, 1024)):
        assert len(pool) >= need, (name, len(pool), need)
        assert len({x["prompt"] for x in pool}) == len(pool)
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        pos = 0
        for label, w, n in PLAN:
            for t in range(1, TRIES + 1):
                for suffix, cnt in (("-warm", w), ("", n)):
                    rows = pool[pos:pos + cnt]; pos += cnt
                    fn = f"{name}/{label}-try{t}{suffix}.jsonl"
                    with open(f"{a.out}/{fn}", "w") as f:
                        for r in rows:
                            f.write(json.dumps({"prompt": r["prompt"], "output_tokens": olen}, ensure_ascii=False) + "\n")
                    man["files"][fn] = {"n": cnt, "sha256": hashlib.sha256(open(f"{a.out}/{fn}", "rb").read()).hexdigest(),
                                        "tokens": [r["tokens"] for r in rows], "src": [r["src"] for r in rows]}
        toks = [x["tokens"] for x in pool[:need]]
        man["stats"][name] = {"prompts": need, "tokens_min": min(toks), "tokens_max": max(toks),
                              "tokens_mean": round(sum(toks) / len(toks), 1)}
    os.makedirs(f"{a.out}/flush", exist_ok=True)
    for j, fl in enumerate(flush):
        open(f"{a.out}/flush/{j:02d}.txt", "w").write(fl["text"])
    man["flush_src"] = [fl["src"] for fl in flush]
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man["stats"]), "flush", len(flush))


if __name__ == "__main__":
    main()
