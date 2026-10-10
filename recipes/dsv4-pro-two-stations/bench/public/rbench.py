#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""rbench.py -- V4P (2026-10-09): a token-accurate streaming TPOT client with explicit prompt sets, thinking
on/off, and index lists, so training and held-out slices are disjoint and named.

usage: rbench.py BASE MODEL OUT.json C N --set pool|gsm --idx SPEC [--think 0|1] [--max 1024] [--ignore-eos 1]
                 [--pool FILE] [--gsm FILE]
  (Our run also had a private prompt set; it is removed from this copy. Only the pool and gsm sets remain, unchanged.)
  pool: a pre-rendered pool jsonl (build_pool_v4p.py), raw prompt to /v1/completions (thinking mode is in the prompt).
  gsm : GSM8K test questions by row index (gsm8k200_v4.py uses rows 0-199; V4P training uses 1100-1163).
  SPEC: comma list of a or a-b ranges; request j gets SPEC[j % len(SPEC)].
TPOT = (t_last - t_first) / (completion_tokens - 1), usage from the stream; decode_agg = C x 1000 / mean TPOT.
Reasoning chunks count as tokens (thinking on). Client socket timeout 900 s (inactivity per read, not wall).
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def parse_idx(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("C", type=int); ap.add_argument("N", type=int)
    ap.add_argument("--set", choices=["gsm", "pool"], default="pool")
    ap.add_argument("--idx", required=True)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--max", type=int, default=1024)
    ap.add_argument("--ignore-eos", type=int, default=1)
    ap.add_argument("--gsm", default="")
    ap.add_argument("--pool", default="", help="pre-rendered jsonl ({prompt}); sent raw to /v1/completions")
    a = ap.parse_args()
    idx = parse_idx(a.idx)
    if a.set == "gsm":
        prompts = [json.loads(line)["question"] for line in open(a.gsm)]
    else:
        prompts = [json.loads(line)["prompt"] for line in open(a.pool)]

    def one(j):
        i = idx[j % len(idx)]
        body = {"model": a.model, "temperature": 0.0, "max_tokens": a.max, "stream": True,
                "stream_options": {"include_usage": True}}
        if a.set == "pool":
            body["prompt"] = prompts[i]
        else:
            body.update(chat_template_kwargs={"thinking": bool(a.think)}, messages=[{"role": "user", "content": prompts[i]}])
        if a.ignore_eos:
            body["ignore_eos"] = True
        req = urllib.request.Request(a.base + ("/v1/completions" if a.set == "pool" else "/v1/chat/completions"), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); first = last = None; n = 0; toks = None
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    toks = d["usage"].get("completion_tokens")
                ch = (d.get("choices") or [{}])[0]
                delta = ch.get("delta", {})
                if ch.get("text") or delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.time(); first = first or now; last = now; n += 1
        toks = toks or n
        return {"i": i, "ttft": first - t0, "chunks": n, "tokens": toks,
                "tpot_ms": 1000 * (last - first) / max(toks - 1, 1), "wall": last - t0}

    t0 = time.time()
    with ThreadPoolExecutor(a.C) as ex:
        res = list(ex.map(one, range(a.N)))
    wall = time.time() - t0
    mt = sum(r["tpot_ms"] for r in res) / len(res)
    summ = {"set": a.set, "think": a.think, "idx": a.idx, "C": a.C, "n": a.N, "mean_tpot_ms": round(mt, 2),
            "decode_agg": round(a.C * 1000 / mt, 2), "median_ttft_s": round(sorted(r["ttft"] for r in res)[len(res) // 2], 2),
            "mean_tokens": sum(r["tokens"] for r in res) / a.N,
            "tok_per_chunk": round(sum(r["tokens"] for r in res) / max(1, sum(r["chunks"] for r in res)), 3),
            "out_tput": round(sum(r["tokens"] for r in res) / wall, 2), "wall_s": round(wall, 1)}
    json.dump({"summary": summ, "req": res}, open(a.out, "w"))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
