#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""beam_rand.py catid|ttft BASE MODEL OUT.json [--isl N] [--osl N] [--reps R] [--seed S]

Random-token-id rows for an SGLang server, without `vllm bench serve` (not in lmsysorg/sglang:beam) and without a dataset file.
Contract: prompt = ISL uniform random token ids, temperature 0, ignore_eos + min_tokens = OSL forced, one request in flight,
decode tok/s = 1000 / TPOT with TPOT = (t_last - t_first) / (completion_tokens - 1); every rep has its own seed and the warm-up
its own seed, so no prefix is ever reused (the SGLang cache_hit_rate gauge is recorded too). Ids are drawn from [1000, 150000),
below every special token (151643+) of the Beam tokenizer. Random ids are a control: a drafter accepts almost nothing on them.

  catid: ISL 8192 / OSL 1024 (defaults), R reps (3), one prompt each, C1.  Prints 'catid rep K: decode_agg X tok/s ...' per rep.
  ttft : ISL N (8192 / 65536), OSL 8, R reps; TTFT = first streamed token; prefill tok/s = prompt_tokens / TTFT.

POST /v1/completions with `prompt` as a list of ids (SGLang CompletionRequest accepts List[int]; min_tokens / ignore_eos are fields
of that request in the image used here). Streams with include_usage; chunk_acc = completion tokens / streamed chunks (one
chunk per verify step under speculative decoding)."""
import argparse
import json
import random
import statistics
import time
import urllib.request


def gauges(base):
    g = {"accept_len": None, "cache_hit_rate": None}
    try:
        txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return g
    for line in txt.splitlines():
        if line.startswith("sglang:spec_accept_length"):
            g["accept_len"] = float(line.split()[-1])
        elif line.startswith("sglang:cache_hit_rate"):
            g["cache_hit_rate"] = float(line.split()[-1])
    return g


def one(base, model, ids, osl):
    body = {"model": model, "prompt": ids, "max_tokens": osl, "min_tokens": osl, "ignore_eos": True, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.time()
                first = first or now
                last = now
                chunks += 1
    n = (usage or {}).get("completion_tokens") or 0
    p = (usage or {}).get("prompt_tokens")
    return {"ttft_s": (first - t0) if first else None, "ptoks": p, "ctoks": n, "chunks": chunks,
            "tpot_ms": 1000 * (last - first) / max(n - 1, 1) if first and n > 1 else None,
            "prefill_tok_s": round(p / (first - t0), 1) if first and p else None, "wall_s": round(time.time() - t0, 2)}


def ids_for(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("catid", "ttft"))
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000 * 1000)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    osl = a.osl if a.osl is not None else (1024 if a.mode == "catid" else 8)
    # warm-up: separate seed, short shape (never reported)
    one(base, a.model, ids_for(a.seed + 500000, 512), 16)
    res = []
    for k in range(1, a.reps + 1):
        g0 = gauges(base)
        r = one(base, a.model, ids_for(a.seed + 100 * k, a.isl), osl)
        r.update({"rep": k, "seed": a.seed + 100 * k, "accept_len_gauge": gauges(base)["accept_len"],
                  "cache_hit_rate_gauge_before": g0["cache_hit_rate"]})
        if r["tpot_ms"]:
            r["decode_agg"] = round(1000 / r["tpot_ms"], 2)
        r["chunk_acc"] = round(r["ctoks"] / r["chunks"], 3) if r["chunks"] else None
        res.append(r)
        if a.mode == "catid":
            print(f"catid rep {k}: decode_agg {r.get('decode_agg')} tok/s TTFT {r['ttft_s']} s ptoks {r['ptoks']} ctoks {r['ctoks']} "
                  f"chunk_acc {r['chunk_acc']} gauge_acc {r['accept_len_gauge']}", flush=True)
        else:
            print(f"ttft isl {a.isl} rep {k}: TTFT {r['ttft_s']} s prefill {r['prefill_tok_s']} tok/s ptoks {r['ptoks']}", flush=True)
    ok = [r for r in res if r["ttft_s"]]
    summ = {"mode": a.mode, "isl": a.isl, "osl": osl, "reps": a.reps, "ok": len(ok)}
    if a.mode == "catid":
        d = [r["decode_agg"] for r in res if r.get("decode_agg")]
        summ.update({"decode_agg_reps": d, "decode_agg_median": round(statistics.median(d), 2) if d else None,
                     "chunk_acc_median": round(statistics.median(r["chunk_acc"] for r in ok if r["chunk_acc"]), 3) if ok else None})
    else:
        t = [r["ttft_s"] for r in ok]
        summ.update({"ttft_reps_s": [round(x, 3) for x in t], "ttft_median_s": round(statistics.median(t), 3) if t else None,
                     "prefill_tok_s_median": round(statistics.median(r["prefill_tok_s"] for r in ok), 1) if ok else None})
    json.dump({"summary": summ, "req": res}, open(a.out, "w"))
    print(json.dumps(summ), flush=True)
    raise SystemExit(0 if len(ok) == a.reps else 4)


if __name__ == "__main__":
    main()
