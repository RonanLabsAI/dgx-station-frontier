#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""lc_probe_2hop_ds.py BASE MODEL FIT_JSONL OUT.json --length L [--seeds 1016,...,1021] [--arms T,S] [--nonce N] [--item-timeout S]
Two-hop chaining vs retrieval for DS-V4.1-Flash-1017 at one context length. REPORT ONLY. In results/kv-offload.md it is used
only as a library by kvo_probe.py and kvr_probe.py (corpus, needle plan, build, stream, calibrate).

Derived from tools/lc_probe.py (results/long-context-1m.md) via a thinking-on variant. Kept IDENTICAL: the needle
generator plan(seed, L) (names, agents, codes, depths, the hop1 / hop2 question wording), build(), corpus order (fit.jsonl lines
300.. then 0..299), the chars-per-token calibration, the stream parser, the 400 -> x0.97 retry, temperature 0, the validity band
[0.95, 1.05] x L and "correct iff the wanted string is in message.content". So the 125K rows for seeds 1016-1021 rebuild the same
documents as an earlier 125K two-hop run (not published here), apart from the per-item nonce line.
Deltas (pre-registered; no env switches -- the arms are constants):
  * arms are fixed:  T = two-hop, chat_template_kwargs {"thinking": true, "reasoning_effort": 100}, max_tokens 4000
                     S = split hops (hop1 "which vault", hop2 "that vault's code"), {"thinking": false}, max_tokens 64
  * many seeds in one call; order seed-major (seed -> arm -> items), so a cut run leaves both arms balanced;
  * one length per call (the runner decides seeds per length from its wall budget);
  * every item is a cold prefill: first line "[ref <nonce>-<L>-<seed>-<arm>-<i>]";
  * n = the items ASKED (the template's split arm reported n = all 10 plan items);
  * per item: completion tokens, reasoning chars, finish reason, TTFT, prefill tok/s = prompt_tokens / TTFT, and for a miss a
    fail_mode: length-cut (finish=length, answer absent) | empty | intermediate (a two-hop answer naming the chain's vault) |
    other-code (another needle's code) | other;
  * skip rules (from tools/lc_probe_glm.py): 2 consecutive timeouts -> the rest of the length is "skipped";
    a failure with /v1/models unreachable -> server_dead_at = L and the rest is "skipped: server dead".
  * timeout per item max(300, 900 x L / 1,000,000) s (the template's).
No prompt text is written to OUT (answers keep only their last 80 characters). Last stdout line: JSON summary of the length.
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
SEEDS = "1016,1017,1018,1019,1020,1021"
FAMILY = "ds"
ARMS = {
    "T": {"kinds": ("two-hop",), "kw": {"thinking": True, "reasoning_effort": 100}, "max_tokens": 4000},
    "S": {"kinds": ("hop1", "hop2"), "kw": {"thinking": False}, "max_tokens": 64},
}
CAL_KW = {"thinking": False}


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
                   "q": f"What is the registration code of the vault where Agent {a[h]} keeps their records?", "want": cc,
                   "chain": h, "vault": vv})
        qs.append({"kind": "hop1", "depth": [d1], "q": f"In which vault does Agent {a[h]} keep their records?", "want": vv,
                   "fmt": " Answer with the vault name only.", "chain": h, "vault": vv})
        qs.append({"kind": "hop2", "depth": [d2], "q": f"What is the registration code of the vault named {vv}?", "want": cc,
                   "chain": h, "vault": vv})
    return sorted(needles), qs, codes


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


def item(base, model, doc, question, arm, timeout):
    body = {"model": model, "temperature": 0.0, "max_tokens": ARMS[arm]["max_tokens"], "chat_template_kwargs": dict(ARMS[arm]["kw"]),
            "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question}]}
    return stream(base, body, timeout)


def is_timeout(e):
    s = "%s %s" % (type(e).__name__, e)
    return "timed out" in s or "Timeout" in s


def fail_mode(q, content, finish, codes):
    if q["want"] in content:
        return None
    if finish == "length":
        return "length-cut"
    if not content.strip():
        return "empty"
    if q["kind"] == "two-hop" and q["vault"] in content:
        return "intermediate"
    if any(c in content for c in codes if c != q["want"]):
        return "other-code"
    return "other"


def calibrate(base, model, text):
    probe = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(
        {"model": model, "max_tokens": 1, "temperature": 0.0, "chat_template_kwargs": dict(CAL_KW),
         "messages": [{"role": "user", "content": text[:40000]}]}).encode(), {"Content-Type": "application/json"}), timeout=600))
    return 40000 / max(1, probe["usage"]["prompt_tokens"])


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def arm_summary(L, recs):
    """Per-arm block for one length (pure; th_summary.py re-derives the same numbers from req)."""
    ok = [r for r in recs if "error" not in r]
    pts = [r.get("ptoks") or 0 for r in ok]
    fin, fm = {}, {}
    for r in ok:
        fin[str(r.get("finish"))] = fin.get(str(r.get("finish")), 0) + 1
        if r.get("fail_mode"):
            fm[r["fail_mode"]] = fm.get(r["fail_mode"], 0) + 1
    by_kind = {}
    for r in recs:
        k = by_kind.setdefault(r["kind"], {"correct": 0, "n": 0})
        k["n"] += 1
        k["correct"] += int(bool(r.get("ok")))
    chains = {}
    for r in recs:
        chains.setdefault((r["seed"], r["chain"]), []).append(bool(r.get("ok")))
    errs = len(recs) - len(ok)
    return {"correct": sum(int(bool(r.get("ok"))) for r in recs), "n": len(recs), "errors": errs, "by_kind": by_kind,
            "chains_all_correct": sum(1 for v in chains.values() if all(v)), "chains": len(chains),
            "valid": errs == 0 and bool(recs) and all(PT_LO * L <= p <= PT_HI * L for p in pts),
            "ptoks_min": min(pts) if pts else None, "ptoks_max": max(pts) if pts else None,
            "ctoks_median": med([r.get("ctoks") for r in ok]), "ctoks_max": max([r.get("ctoks") or 0 for r in ok], default=None),
            "finish": fin, "fail_modes": fm,
            "ttft_s_median": med([r.get("ttft_s") for r in ok]), "prefill_tok_s_median": med([r.get("prefill_tok_s") for r in ok])}


def run(a):
    base = a.base.rstrip("/")
    text = corpus(a.fit)
    cpt = calibrate(base, a.model, text)
    L = int(a.length)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    arms = [x for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            raise SystemExit("unknown arm %r (have %s)" % (x, ",".join(ARMS)))
    res = {"family": FAMILY, "length": L, "seeds": seeds, "arms": {x: ARMS[x] for x in arms}, "chars_per_token": round(cpt, 3),
           "corpus_chars": len(text), "req": [], "server_dead_at": None, "skipped_from": None, "by_arm": {}}
    nonce = a.nonce or str(int(time.time()) % 1000000)
    skip, consec_to = None, 0
    for seed in seeds:
        needles, qs, codes = plan(seed, L)
        for arm in arms:
            for qi, q in enumerate([q for q in qs if q["kind"] in ARMS[arm]["kinds"]]):
                rec = {"len": L, "seed": seed, "arm": arm, "i": qi, "kind": q["kind"], "chain": q["chain"], "depth": q["depth"]}
                if skip:
                    rec.update(error="skipped: " + skip)
                    res["req"].append(rec)
                    print(json.dumps(rec), flush=True)
                    continue
                frac = 1.0
                for attempt in (0, 1):
                    doc = "[ref %s-%d-%d-%s-%d]\n" % (nonce, L, seed, arm, qi) + build(text, int(L * cpt * frac), needles)
                    m0 = metrics(base)
                    try:
                        r = item(base, a.model, doc, q["q"] + q.get("fmt", " Answer with the 6-digit code only."), arm,
                                 a.item_timeout or timeout_for(L))
                        m1 = metrics(base)
                        ok = q["want"] in r["content"]
                        hit = (100.0 * (m1["hits"] - m0["hits"]) / (m1["queries"] - m0["queries"])
                               if m0 and m1 and m1["queries"] > m0["queries"] else None)
                        rec.update(ok=ok, fail_mode=fail_mode(q, r["content"], r["finish"], codes), ctoks=r.get("ctoks"),
                                   reasoning_chars=r["rchars"], answer=r["content"][-80:], ptoks=r["ptoks"], ttft_s=r["ttft_s"],
                                   wall_s=r["wall_s"], finish=r["finish"],
                                   prefill_tok_s=round(r["ptoks"] / r["ttft_s"], 1) if r["ptoks"] and r["ttft_s"] else None,
                                   prefix_hit_pct=round(hit, 2) if hit is not None else None, attempt=attempt)
                        consec_to = 0
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 400 and attempt == 0:
                            frac *= 0.97
                            continue
                        rec.update(error="HTTP %d %s" % (e.code, e.read()[:200].decode("utf-8", "replace")))
                        consec_to = 0
                        break
                    except Exception as e:  # noqa: BLE001  (timeouts, connection resets = a row result)
                        rec.update(error="%s: %s" % (type(e).__name__, str(e)[:200]))
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
    for arm in arms:
        res["by_arm"][arm] = arm_summary(L, [r for r in res["req"] if r["arm"] == arm])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("fit"); ap.add_argument("out")
    ap.add_argument("--length", required=True, type=int)
    ap.add_argument("--seeds", default=SEEDS)
    ap.add_argument("--arms", default="T,S")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--item-timeout", type=float, default=0.0, help="tests only: fixed per-item timeout s (default: the pre-registered formula)")
    a = ap.parse_args()
    res = run(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("family", "length", "seeds", "chars_per_token", "server_dead_at", "skipped_from", "by_arm")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
