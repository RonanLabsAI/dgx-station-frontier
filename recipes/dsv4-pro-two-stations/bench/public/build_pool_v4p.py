#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""build_pool_v4p.py -- public real-text prompt pools for the V4-Pro final run (V4P, 2026-10-09).

Derived from recipes/dsv41-flash-sidecar/bench/realtext/build_realtext_pool.py (DS-V4.1-Flash). Same sources (sha-pinned),
same seed policy (20261009), same hygiene, with three changes:
  * DeepSeek-V4-Pro-0813 tokenizer.json (sha differs from V4.1's), and prompts rendered with the model's own
    encoding/encoding_dsv4.py: thinking OFF = encode_messages(.., "chat") (...<|Assistant|></think>) and thinking ON =
    encode_messages(.., "thinking") (...<|Assistant|><think>, default effort adds nothing).
  * Separate slots for thinking off ("c1-r*", "c16-r1") and on ("tc1-r*", "tc16-r1"), each drawn from fresh prompts, so
    the two modes never share a prompt body (no cross-mode prefix-cache hit).
  * TRAINING slots ("train" / "ttrain") disjoint from every measured slot, for the mixed-placement route capture.
    Every article / conversation is used at most once across the whole pool.

  long : CNN/DailyMail 3.0.0 test articles concatenated to exactly 8,192 tokens + fixed summarise-and-compare instruction.
  short: ShareGPT V3 first human turns, 8-512 content tokens, natural.
Prefix hygiene: no two prompts in a pool share their first 64 tokens after the 2-token header.

usage: build_pool_v4p.py CNN_TEST_PARQUET SHAREGPT_JSON TOKENIZER_JSON ENCODING_DSV4_PY OUTDIR [--seed 20261009]
Writes OUTDIR/{long,short}/<label>-try<t>[-warm].jsonl ({"prompt", "output_tokens"}) and OUTDIR/manifest.json.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random

import pandas as pd
from tokenizers import Tokenizer

TARGET = 8192
PREFIX_UNIQ = 64
LONG_INSTR = ("\n\nFor each article above, write a detailed summary of its main events, people and claims. "
              "Then explain how the articles relate to one another.")
TRIES = 2
# (label, mode, warm-up prompts, measured prompts per variant {long, short})
PLAN = [("c1-r1", "chat", 1, 3), ("c1-r2", "chat", 1, 3), ("c1-r3", "chat", 1, 3), ("c16-r1", "chat", 4, 32),
        ("train", "chat", 0, {"long": 16, "short": 48}),
        ("tc1-r1", "thinking", 1, 3), ("tc1-r2", "thinking", 1, 3), ("tc1-r3", "thinking", 1, 3),
        ("tc16-r1", "thinking", 4, 32), ("ttrain", "thinking", 0, {"long": 16, "short": 48})]


def n_of(n, name):
    return n[name] if isinstance(n, dict) else n


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnn")
    ap.add_argument("sharegpt")
    ap.add_argument("tok")
    ap.add_argument("encoding")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("encoding_dsv4", a.encoding)
    enc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc_mod)

    def render(content, mode):
        return enc_mod.encode_messages([{"role": "user", "content": content}], thinking_mode=mode)

    tok = Tokenizer.from_file(a.tok)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    head = {m: ntok(render("", m)) for m in ("chat", "thinking")}
    assert head["chat"] == head["thinking"] == 4, head

    def need(name):
        return sum(TRIES * (w + n_of(n, name)) for _, _, w, n in PLAN)

    # ---------------- long ----------------
    df = pd.read_parquet(a.cnn)
    order = list(range(len(df)))
    random.Random(a.seed).shuffle(order)
    instr_t = ntok(LONG_INSTR)
    bodies, seen_pref = [], set()
    it = iter(order)
    while len(bodies) < need("long"):
        parts, ids, k = [], [], 0
        budget = TARGET - 4 - instr_t
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
        body = "".join(parts) + LONG_INSTR
        pid = tok.encode(render(body, "chat"), add_special_tokens=False).ids
        key = tuple(pid[2:2 + PREFIX_UNIQ])
        if key in seen_pref:
            continue
        seen_pref.add(key)
        bodies.append({"body": body, "src": ids})

    # ---------------- short ----------------
    sg = json.load(open(a.sharegpt))
    order = list(range(len(sg)))
    random.Random(a.seed).shuffle(order)
    sbodies, seen_pref, seen_txt = [], set(), set()
    for i in order:
        conv = sg[i].get("conversations") or []
        if not conv or conv[0].get("from") != "human":
            continue
        txt = (conv[0].get("value") or "").strip()
        if not txt or txt in seen_txt:
            continue
        pid = tok.encode(render(txt, "chat"), add_special_tokens=False).ids
        if not (8 <= len(pid) - 4 <= 512):
            continue
        key = tuple(pid[2:2 + PREFIX_UNIQ])
        if key in seen_pref:
            continue
        seen_txt.add(txt); seen_pref.add(key)
        sbodies.append({"body": txt, "src": [str(sg[i].get("id"))]})
        if len(sbodies) >= need("short"):
            break

    man = {"seed": a.seed, "target_long_tokens": TARGET, "prefix_uniq_tokens": PREFIX_UNIQ, "tries": TRIES,
           "plan": PLAN, "chat_format": "DeepSeek-V4-Pro-0813 encoding/encoding_dsv4.py encode_messages: label c* = "
           "thinking_mode chat (thinking off, ...</think>), tc*/ttrain = thinking_mode thinking (...<think>); bench uses "
           "--skip-chat-template",
           "sources": {"long": "abisee/cnn_dailymail @ 96df5e686bee6baa90b8bee7c28b81fa3fa6223d 3.0.0/test-00000-of-00001.parquet",
                       "short": "anon8231489123/ShareGPT_Vicuna_unfiltered @ 192ab2185289094fc556ec8ce5ce1e8e587154ca "
                                "ShareGPT_V3_unfiltered_cleaned_split.json",
                       "sha256": {os.path.basename(a.cnn): sha(a.cnn), os.path.basename(a.sharegpt): sha(a.sharegpt),
                                  "tokenizer.json": sha(a.tok), "encoding_dsv4.py": sha(a.encoding)}},
           "long_instruction": LONG_INSTR, "files": {}, "stats": {}}
    for name, pool in (("long", bodies), ("short", sbodies)):
        assert len(pool) >= need(name), (name, len(pool), need(name))
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        pos, toks_all = 0, []
        for label, mode, w, n in PLAN:
            for t in range(1, TRIES + 1):
                for suffix, cnt in (("-warm", w), ("", n_of(n, name))):
                    if cnt == 0:
                        continue
                    rows = pool[pos:pos + cnt]; pos += cnt
                    fn = f"{name}/{label}-try{t}{suffix}.jsonl"
                    toks = []
                    with open(f"{a.out}/{fn}", "w") as f:
                        for r in rows:
                            p = render(r["body"], mode)
                            toks.append(ntok(p))
                            f.write(json.dumps({"prompt": p, "output_tokens": 1024}, ensure_ascii=False) + "\n")
                    toks_all += toks
                    man["files"][fn] = {"n": cnt, "mode": mode, "sha256": sha(f"{a.out}/{fn}"), "tokens": toks,
                                        "src": [r["src"] for r in rows]}
        man["stats"][name] = {"prompts": pos, "tokens_min": min(toks_all), "tokens_max": max(toks_all),
                              "tokens_mean": round(sum(toks_all) / len(toks_all), 1)}
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man["stats"]))


if __name__ == "__main__":
    main()
