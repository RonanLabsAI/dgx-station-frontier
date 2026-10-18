#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""lc_probe.py BASE MODEL HAYSTACK_JSONL OUT.json [--lengths 125000,250000,500000,1000000] [--seed 1016] [--decode-probe]
Long-context needle probe for an OpenAI-compatible vLLM server (DeepSeek-V4.1-Flash at --max-model-len 1,048,576;
results/long-context-1m.md). Report only: it prints scores, it does not pass or fail anything.

  * lengths 125K / 250K / 500K / 1M prompt tokens; haystack = public CNN/DailyMail text, one {"content": ...} JSON object per
    line, read from line 300 to the end and then lines 0..299 (never repeated); characters per token are calibrated once
    against the server (a 40,000-character request with max_tokens 1);
  * thinking off ({"thinking": false}); temperature 0; max_tokens 64; streamed (stream_options.include_usage) so every item records
    TTFT and prefill tok/s = prompt_tokens / TTFT;
  * each of the 6 items per length starts with its own nonce line "[ref <nonce>-<L>-<i>]", so every item is a COLD prefill (no prefix-
    cache reuse between the 6 questions); /metrics prefix_cache hits / queries deltas are recorded per item;
  * validity: every item's prompt_tokens within [0.95, 1.05] x target AND 0 request errors; an HTTP 400 gets one retry at 97% of
    the text;
  * per-item timeout = max(300, 900 x L / 1,000,000) s (15 min at 1M); a timeout / OOM / HTTP error is a recorded row result;
  * ttft_s_median / prefill_tok_s_median are the upper median of the items (sorted list, index n // 2);
  * --decode-probe: after the largest length, one more cold request with that document and 256 forced output tokens (ignore_eos,
    min_tokens) -> decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token).
Needles: single "The registration code of the vault named <V> is <6 digits>." at depths 0.10 / 0.35 / 0.60 / 0.90; two-hop chains
"Agent <A> keeps their records in the vault named <V>." at 0.05 / 0.20 + the code sentence at 0.50 / 0.75. Correct iff the code appears
in message.content. No prompt text is written to OUT. Last stdout line: JSON {"chars_per_token", "lengths", "decode_probe"}.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

DEPTH_SINGLE = (0.10, 0.35, 0.60, 0.90)
DEPTH_HOP = ((0.05, 0.50), (0.20, 0.75))
NAMES = ("Orion Cobalt Juniper Sable Tamarack Vesper Halcyon Larkspur Meridian Quill Saffron Tundra Umber Wren Zephyr Basalt "
         "Cinder Dahlia Ember Fennel Garnet Heron Iris Jasper").split()
AGENTS = "Kestrel Marlow Pike Sorrel Thistle Varga Wexford Yarrow".split()
PT_LO, PT_HI = 0.95, 1.05
LENGTHS = "125000,250000,500000,1000000"


def corpus(fit, line0=300):
    rows = [json.loads(l)["content"] for l in open(fit) if l.strip()]
    return "\n\n".join(rows[line0:] + rows[:line0])


def plan(seed, tag):
    r = random.Random(f"{seed}-{tag}")
    v = r.sample(NAMES, 6)
    a = r.sample(AGENTS, 2)
    codes = [str(r.randrange(100000, 1000000)) for _ in range(6)]
    needles, qs = [], []
    for k, d in enumerate(DEPTH_SINGLE):
        needles.append((d, f"The registration code of the vault named {v[k]} is {codes[k]}."))
        qs.append({"kind": "single", "depth": d, "q": f"What is the registration code of the vault named {v[k]}?", "want": codes[k]})
    for h, (d1, d2) in enumerate(DEPTH_HOP):
        vv, cc = v[4 + h], codes[4 + h]
        needles.append((d1, f"Agent {a[h]} keeps their records in the vault named {vv}."))
        needles.append((d2, f"The registration code of the vault named {vv} is {cc}."))
        qs.append({"kind": "two-hop", "depth": [d1, d2],
                   "q": f"What is the registration code of the vault where Agent {a[h]} keeps their records?", "want": cc})
    return sorted(needles), qs


def build(text, nchars, needles):
    doc = text[:nchars]
    for d, s in sorted(needles, reverse=True):
        p = int(len(doc) * d)
        p = doc.rfind(" ", 0, p) + 1 if doc.rfind(" ", 0, p) > 0 else p
        doc = doc[:p] + s + " " + doc[p:]
    return doc


def timeout_for(L):
    return max(300.0, 900.0 * L / 1_000_000)


def metrics(base):
    try:
        t = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return None
    v = {"hits": 0.0, "queries": 0.0}
    for line in t.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            v["hits"] += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_queries_total"):
            v["queries"] += float(line.split()[-1])
    return v


def stream(base, body, timeout):
    """-> dict(content, ptoks, ctoks, ttft_s, t_first, t_last, wall_s, finish)"""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    out = {"content": "", "ptoks": None, "ctoks": None, "t_first": None, "t_last": None, "finish": None}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece or (ch.get("delta") or {}).get("reasoning_content") or (ch.get("delta") or {}).get("reasoning"):
                    now = time.time()
                    out["t_first"] = out["t_first"] or now
                    out["t_last"] = now
                out["content"] += piece
                out["finish"] = ch.get("finish_reason") or out["finish"]
            if d.get("usage"):
                out["ptoks"], out["ctoks"] = d["usage"].get("prompt_tokens"), d["usage"].get("completion_tokens")
    out["wall_s"] = round(time.time() - t0, 2)
    out["ttft_s"] = round(out["t_first"] - t0, 3) if out["t_first"] else None
    return out


def item(base, model, doc, question, max_tokens, timeout, forced=False):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question}]}
    if forced:
        body.update(ignore_eos=True, min_tokens=max_tokens)
    return stream(base, body, timeout)


def run(a):
    base = a.base.rstrip("/")
    text = corpus(a.fit)
    probe = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(
        {"model": a.model, "max_tokens": 1, "temperature": 0.0, "chat_template_kwargs": {"thinking": False},
         "messages": [{"role": "user", "content": text[:40000]}]}).encode(), {"Content-Type": "application/json"}), timeout=600))
    cpt = 40000 / max(1, probe["usage"]["prompt_tokens"])
    res = {"chars_per_token": round(cpt, 3), "corpus_chars": len(text), "lengths": {}, "req": [], "decode_probe": None}
    nonce = a.nonce or str(int(time.time()) % 1000000)
    Ls = [int(x) for x in a.lengths.split(",")]
    for L in Ls:
        needles, qs = plan(a.seed, L)
        frac, n_ok, errs, pts = 1.0, 0, 0, []
        for qi, q in enumerate(qs):
            rec = {"len": L, "i": qi, "kind": q["kind"], "depth": q["depth"]}
            for attempt in (0, 1):
                doc = "[ref %s-%d-%d]\n" % (nonce, L, qi) + build(text, int(L * cpt * frac), needles)
                m0 = metrics(base)
                try:
                    r = item(base, a.model, doc, q["q"] + " Answer with the 6-digit code only.", 64, timeout_for(L))
                    m1 = metrics(base)
                    ok = q["want"] in r["content"]
                    hit = (100.0 * (m1["hits"] - m0["hits"]) / (m1["queries"] - m0["queries"])
                           if m0 and m1 and m1["queries"] > m0["queries"] else None)
                    rec.update(ok=ok, ptoks=r["ptoks"], ttft_s=r["ttft_s"], wall_s=r["wall_s"], finish=r["finish"],
                               prefill_tok_s=round(r["ptoks"] / r["ttft_s"], 1) if r["ptoks"] and r["ttft_s"] else None,
                               prefix_hit_pct=round(hit, 2) if hit is not None else None, attempt=attempt)
                    n_ok += int(ok)
                    pts.append(r["ptoks"] or 0)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 400 and attempt == 0:
                        frac *= 0.97
                        continue
                    rec.update(error="HTTP %d %s" % (e.code, e.read()[:200].decode("utf-8", "replace")))
                    errs += 1
                    break
                except Exception as e:  # noqa: BLE001  (timeouts, connection resets = a row result)
                    rec.update(error="%s: %s" % (type(e).__name__, str(e)[:200]))
                    errs += 1
                    break
            res["req"].append(rec)
            print(json.dumps(rec), flush=True)
        valid = errs == 0 and len(pts) == len(qs) and all(PT_LO * L <= p <= PT_HI * L for p in pts)
        okr = [x for x in res["req"] if x["len"] == L and x.get("ptoks")]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        res["lengths"][str(L)] = {"correct": n_ok, "n": len(qs), "errors": errs, "valid": valid,
                                  "ptoks_min": min(pts) if pts else None, "ptoks_max": max(pts) if pts else None,
                                  "ttft_s_median": med([x["ttft_s"] for x in okr if x.get("ttft_s")]),
                                  "prefill_tok_s_median": med([x["prefill_tok_s"] for x in okr if x.get("prefill_tok_s")])}
    if a.decode_probe:
        L = max(Ls)
        needles, _ = plan(a.seed, L)
        doc = "[ref %s-%d-decode]\n" % (nonce, L) + build(text, int(L * cpt * 0.99), needles)
        try:
            r = item(base, a.model, doc, "Write a long, detailed summary of the articles above.", 256, timeout_for(L) + 300, forced=True)
            dec = ((r["ctoks"] - 1) / (r["t_last"] - r["t_first"])) if r["ctoks"] and r["t_last"] and r["t_first"] and r["t_last"] > r["t_first"] else None
            res["decode_probe"] = {"len": L, "ptoks": r["ptoks"], "ctoks": r["ctoks"], "ttft_s": r["ttft_s"],
                                   "decode_tok_s": round(dec, 2) if dec else None, "wall_s": r["wall_s"]}
        except Exception as e:  # noqa: BLE001
            res["decode_probe"] = {"len": L, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        print(json.dumps({"decode_probe": res["decode_probe"]}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("fit"); ap.add_argument("out")
    ap.add_argument("--lengths", default=LENGTHS)
    ap.add_argument("--seed", type=int, default=1016)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--decode-probe", action="store_true")
    a = ap.parse_args()
    res = run(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chars_per_token", "lengths", "decode_probe")}), flush=True)


if __name__ == "__main__":
    main()
