#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""lc_probe_glm.py BASE MODEL HAYSTACK_JSONL OUT.json [--lengths 125000,250000,500000,1000000] [--seed 1016] [--decode-probe] [--nonce N]
       lc_probe_glm.py unmeasured OUT.json        -> CSV of the lengths in OUT.json with 0 successful items
Long-context needle probe for GLM-5.3-Flash on SGLang (results/long-context-1m.md). Report only.

A copy of tools/lc_probe.py (same needles, depths, seed, haystack order, nonce-per-item cold prefill, validity band, 400 retry,
upper-median summaries). GLM deltas, each forced by the model or the server:
  * chat_template_kwargs {"reasoning_effort": "low"}: GLM's chat template has no thinking switch, and the Qwen/DeepSeek
    thinking-off kwargs return an EMPTY answer.
  * max_tokens 1500 per item (the assistant turn always opens with reasoning). TTFT = first streamed token, reasoning or
    content; correct iff the 6-digit code is in message content.
  * per-item timeout max(300, 1800 x L / 1,000,000) s (30 min at 1M).
  * two consecutive timeouts at a length -> the rest of that length and every longer length are recorded "skipped" (errors);
    a connection failure with /v1/models unreachable -> server_dead_at = L and every remaining item is "skipped: server dead".
  * --decode-probe runs at the LONGEST length with >= 1 successful item (not blindly the largest), 256 forced tokens.
  * prefix-cache /metrics: read if the server exposes them (SGLang needs --enable-metrics); otherwise prefix_hit_pct is null
    and the server log's "#cached-token" field is the check. Cold by construction: the nonce is the first content line, so
    at most the chat-template preamble can match the radix cache.
Validity: every item's prompt_tokens within [0.95, 1.05] x target AND 0 errors. prompt_tokens is the server's count, i.e.
GLM's own tokenizer. No prompt text is written to OUT.
"""
import argparse
import json
import random
import sys
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
KW = {"reasoning_effort": "low"}
MAX_TOKENS = 1500
DECODE_TOKENS = 256


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


def item_doc(text, cpt, L, frac, needles, nonce, tag):
    """The exact user content of one item."""
    return "[ref %s-%d-%s]\n" % (nonce, L, tag) + build(text, int(L * cpt * frac), needles)


def question_text(q):
    return q["q"] + " Answer with the 6-digit code only."


def timeout_for(L):
    return max(300.0, 1800.0 * L / 1_000_000)


def metrics(base):
    try:
        t = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return None
    v = {"hits": 0.0, "queries": 0.0}
    for line in t.splitlines():
        if line.startswith(("vllm:prefix_cache_hits_total", "sglang:prefix_cache_hits_total")):
            v["hits"] += float(line.split()[-1])
        elif line.startswith(("vllm:prefix_cache_queries_total", "sglang:prefix_cache_queries_total")):
            v["queries"] += float(line.split()[-1])
    return v


def alive(base):
    try:
        urllib.request.urlopen(base + "/v1/models", timeout=20).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def stream(base, body, timeout):
    """-> dict(content, rchars, ptoks, ctoks, ttft_s, t_first, t_last, wall_s, finish)"""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    out = {"content": "", "rchars": 0, "ptoks": None, "ctoks": None, "t_first": None, "t_last": None, "finish": None}
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
                dl = ch.get("delta") or {}
                piece = dl.get("content") or ""
                rpiece = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if piece or rpiece:
                    now = time.time()
                    out["t_first"] = out["t_first"] or now
                    out["t_last"] = now
                out["content"] += piece
                out["rchars"] += len(rpiece)
                out["finish"] = ch.get("finish_reason") or out["finish"]
            if d.get("usage"):
                out["ptoks"], out["ctoks"] = d["usage"].get("prompt_tokens"), d["usage"].get("completion_tokens")
    out["wall_s"] = round(time.time() - t0, 2)
    out["ttft_s"] = round(out["t_first"] - t0, 3) if out["t_first"] else None
    return out


def item(base, model, doc, question, max_tokens, timeout, forced=False):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": dict(KW),
            "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question}]}
    if forced:
        body.update(ignore_eos=True, min_tokens=max_tokens)
    return stream(base, body, timeout)


def is_timeout(e):
    s = "%s %s" % (type(e).__name__, e)
    return "timed out" in s or "Timeout" in s


def calibrate(base, model, text):
    probe = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(
        {"model": model, "max_tokens": 1, "temperature": 0.0, "chat_template_kwargs": dict(KW),
         "messages": [{"role": "user", "content": text[:40000]}]}).encode(), {"Content-Type": "application/json"}), timeout=600))
    return 40000 / max(1, probe["usage"]["prompt_tokens"])


def run(a):
    base = a.base.rstrip("/")
    text = corpus(a.fit)
    cpt = calibrate(base, a.model, text)
    res = {"chars_per_token": round(cpt, 3), "corpus_chars": len(text), "lengths": {}, "req": [], "decode_probe": None,
           "server_dead_at": None, "skipped_from": None, "kwargs": KW, "max_tokens": MAX_TOKENS}
    nonce = a.nonce or str(int(time.time()) % 1000000)
    Ls = [int(x) for x in a.lengths.split(",")]
    skip = None                                                     # reason string once a skip rule fired
    for L in Ls:
        needles, qs = plan(a.seed, L)
        frac, n_ok, errs, pts, consec_to = 1.0, 0, 0, [], 0
        for qi, q in enumerate(qs):
            rec = {"len": L, "i": qi, "kind": q["kind"], "depth": q["depth"]}
            if skip:
                rec.update(error="skipped: " + skip)
                errs += 1
                res["req"].append(rec)
                print(json.dumps(rec), flush=True)
                continue
            for attempt in (0, 1):
                doc = item_doc(text, cpt, L, frac, needles, nonce, str(qi))
                m0 = metrics(base)
                try:
                    r = item(base, a.model, doc, question_text(q), MAX_TOKENS, a.item_timeout or timeout_for(L))
                    m1 = metrics(base)
                    ok = q["want"] in r["content"]
                    hit = (100.0 * (m1["hits"] - m0["hits"]) / (m1["queries"] - m0["queries"])
                           if m0 and m1 and m1["queries"] > m0["queries"] else None)
                    rec.update(ok=ok, ptoks=r["ptoks"], ctoks=r["ctoks"], ttft_s=r["ttft_s"], wall_s=r["wall_s"], finish=r["finish"],
                               reasoning_chars=r["rchars"], answer_chars=len(r["content"]),
                               prefill_tok_s=round(r["ptoks"] / r["ttft_s"], 1) if r["ptoks"] and r["ttft_s"] else None,
                               prefix_hit_pct=round(hit, 2) if hit is not None else None, attempt=attempt)
                    n_ok += int(ok)
                    pts.append(r["ptoks"] or 0)
                    consec_to = 0
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 400 and attempt == 0:
                        frac *= 0.97
                        continue
                    rec.update(error="HTTP %d %s" % (e.code, e.read()[:200].decode("utf-8", "replace")))
                    errs += 1
                    consec_to = 0
                    break
                except Exception as e:  # noqa: BLE001  (timeouts, connection resets = a row result)
                    rec.update(error="%s: %s" % (type(e).__name__, str(e)[:200]))
                    errs += 1
                    if is_timeout(e):
                        consec_to += 1
                        if consec_to >= 2:
                            skip = "2 consecutive timeouts at %d" % L
                            res["skipped_from"] = L
                    if not alive(base):
                        skip = "server dead (at %d)" % L
                        res["server_dead_at"] = L
                    break
            res["req"].append(rec)
            print(json.dumps(rec), flush=True)
        valid = errs == 0 and len(pts) == len(qs) and all(PT_LO * L <= p <= PT_HI * L for p in pts)
        okr = [x for x in res["req"] if x["len"] == L and x.get("ptoks")]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        res["lengths"][str(L)] = {"correct": n_ok, "n": len(qs), "errors": errs, "valid": valid, "measured": len(okr),
                                  "ptoks_min": min(pts) if pts else None, "ptoks_max": max(pts) if pts else None,
                                  "ttft_s_median": med([x["ttft_s"] for x in okr if x.get("ttft_s")]),
                                  "prefill_tok_s_median": med([x["prefill_tok_s"] for x in okr if x.get("prefill_tok_s")])}
    if a.decode_probe:
        done = [L for L in Ls if res["lengths"][str(L)]["measured"] > 0]
        if not done or res["server_dead_at"] is not None and not alive(base):
            res["decode_probe"] = {"len": max(done) if done else None, "error": "skipped: no measured length or server dead"}
        else:
            L = max(done)
            needles, _ = plan(a.seed, L)
            doc = item_doc(text, cpt, L, 0.99, needles, nonce, "decode")
            try:
                r = item(base, a.model, doc, "Write a long, detailed summary of the articles above.", DECODE_TOKENS, (a.item_timeout or timeout_for(L)) + (0 if a.item_timeout else 300), forced=True)
                dec = ((r["ctoks"] - 1) / (r["t_last"] - r["t_first"])) if r["ctoks"] and r["t_last"] and r["t_first"] and r["t_last"] > r["t_first"] else None
                res["decode_probe"] = {"len": L, "ptoks": r["ptoks"], "ctoks": r["ctoks"], "ttft_s": r["ttft_s"],
                                       "decode_tok_s": round(dec, 2) if dec else None, "wall_s": r["wall_s"]}
            except Exception as e:  # noqa: BLE001
                res["decode_probe"] = {"len": L, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        print(json.dumps({"decode_probe": res["decode_probe"]}), flush=True)
    return res


def unmeasured(path):
    """Lengths in an OUT.json with 0 successful items (CSV, ascending; empty when every length has >= 1)."""
    try:
        r = json.load(open(path))
    except (OSError, ValueError):
        return None
    return ",".join(L for L in sorted(r["lengths"], key=int) if r["lengths"][L].get("measured", 0) == 0)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "unmeasured":
        u = unmeasured(sys.argv[2])
        if u is None:
            return 1
        print(u)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("fit"); ap.add_argument("out")
    ap.add_argument("--lengths", default=LENGTHS)
    ap.add_argument("--seed", type=int, default=1016)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--decode-probe", action="store_true")
    ap.add_argument("--item-timeout", type=float, default=0.0, help="tests only: fixed per-item timeout s (default: the pre-registered formula)")
    a = ap.parse_args()
    res = run(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chars_per_token", "lengths", "decode_probe", "server_dead_at", "skipped_from")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
