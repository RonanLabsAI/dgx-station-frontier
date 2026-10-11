#!/usr/bin/env python3
"""Streaming bench client for a TensorFold OpenAI-compatible server (B1, 2026-10-10). Stdlib only.

usage: bench_tf.py URL SLOT.jsonl OUT.jsonl --conc N [--max-tokens 1024] [--think 0|1] [--seed S] [--label L]

Per request (streamed /v1/chat/completions, stream_options.include_usage):
  ttft_s        first content/reasoning delta minus send time
  decode_tps    (completion_tokens - 1) / (t_last_delta - t_first_delta)   <- "decode tok/s", excludes prefill
  output_tps    completion_tokens / (t_last_delta - t_send)                <- "output tok/s", includes prefill
  prefill_tps   prompt_tokens / ttft_s                                     <- meaningful for long prompts only
  cached_tokens usage.cached_tokens (TensorFold kept-prompt reuse); > 2% of prompt_tokens marks the run CONTAMINATED
Run summary: mean/median of per-request values, aggregate tok/s = sum(completion_tokens) / wall, max cached fraction.
Every invocation gets its own seed (sent as `seed`); warm-ups are separate invocations on their own slot.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one(url, content, max_tokens, think, seed, timeout=1800, model="glm"):
    body = {
        "model": model, "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": seed,
        "enable_thinking": bool(think), "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage, text, reasoning, finish, n_deltas = {}, [], [], None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if piece or rpiece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n_deltas += 1
                    text.append(piece)
                    reasoning.append(rpiece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    ct = int(usage.get("completion_tokens", 0))
    pt = int(usage.get("prompt_tokens", 0))
    cached = int(usage.get("cached_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0)
    ttft = (first - t0) if first else None
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else None
    return {
        "t_send": t0, "t_end": t_end, "ttft_s": ttft, "prompt_tokens": pt, "completion_tokens": ct,
        "cached_tokens": cached, "cached_frac": (cached / pt) if pt else 0.0, "decode_tps": dec,
        "output_tps": ct / ((last or t_end) - t0) if ct else None, "prefill_tps": (pt / ttft) if ttft else None,
        "finish_reason": finish, "deltas": n_deltas, "usage": usage,
        "text_head": "".join(text)[:400], "reasoning_chars": sum(map(len, reasoning)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"), ap.add_argument("slot"), ap.add_argument("out")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default="glm")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.slot)]
    res, lock = [], threading.Lock()

    def job(i):
        r = one(a.url, rows[i]["content"], a.max_tokens, a.think, a.seed * 1000 + i, model=a.model)
        r["i"] = i
        with lock:
            res.append(r)
        return r

    w0 = time.perf_counter()
    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(job, range(len(rows))))
    wall = time.perf_counter() - w0
    res.sort(key=lambda r: r["i"])

    def stat(k):
        v = [r[k] for r in res if r[k] is not None]
        return {"mean": statistics.fmean(v), "median": statistics.median(v), "n": len(v)} if v else None

    summ = {
        "label": a.label, "slot": a.slot, "conc": a.conc, "seed": a.seed, "think": a.think, "max_tokens": a.max_tokens,
        "requests": len(res), "wall_s": wall, "sum_completion_tokens": sum(r["completion_tokens"] for r in res),
        "aggregate_output_tps": sum(r["completion_tokens"] for r in res) / wall,
        "decode_tps": stat("decode_tps"), "output_tps": stat("output_tps"), "ttft_s": stat("ttft_s"),
        "prefill_tps": stat("prefill_tps"), "max_cached_frac": max(r["cached_frac"] for r in res),
        "finish_reasons": sorted({str(r["finish_reason"]) for r in res}),
    }
    summ["contaminated"] = summ["max_cached_frac"] > 0.02
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summ}) + "\n")
    print(json.dumps(summ, indent=1))
    sys.exit(3 if summ["contaminated"] else 0)


if __name__ == "__main__":
    main()
