#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""kvo_probe.py BASE MODEL FIT_JSONL RUNDIR PHASE [--k K] [--length L] [--n N] [--conc C] [--pass P] [--nonce X] [--pool T] [--timeout S] [--scale F (tests only)]
results/kv-offload.md, test 2 (DeepSeek, first host-tier run): one measurement phase against one booted arm. REPORT ONLY. Writes RUNDIR/<phase-tag>.json
and prints a one-line JSON summary last. Real text only (fit.jsonl public news articles); no prompt text is written anywhere.

Reused UNCHANGED from tools/lc_probe_2hop_ds.py (the two-hop long-context needle probe): corpus(), plan(), build(), stream(),
calibrate(), alive(), the chars-per-token calibration and the validity band [0.95, 1.05] x L. Every request: temperature 0,
chat_template_kwargs {"thinking": false}.

Phases (constants; the runner calls them in the pre-registered order, the same order in both arms):
  corr    --pass 1|2   2 documents (seeds 1016, 1017) of 250K tokens each with the twohop needles; per document 10 questions
                       (4 single, 2 hop1, 2 hop2, 2 two-hop), max_tokens 64. The document is a shared prefix of its 10 items. Pass 2
                       re-sends pass 1's prompts BYTE-IDENTICAL (nonce kept in RUNDIR/state-corr.json) after a GPU flush, so in
                       arm ON its KV comes back from the host tier and in arm OFF it is recomputed. Full answers (<= 400 chars of
                       model output) are kept for the identity comparison.
  turn    --k 1..5     8 sessions x 1 turn. All sessions share one system message P (~250K tokens). Each turn appends a user
                       message (~8K tokens of tool output, real text) and asks for a one-sentence topic (max_tokens 96); the
                       assistant reply is appended to the session history (RUNDIR/state-turns.json). Sessions run in order
                       1..8, C1. Session 1 of turns 2-5 is the "first toucher" of P after a flush; sessions 2-8 are "followers".
  cap     --length L --n N   N distinct documents of L tokens sent at once (C=N), max_tokens 16. A sampler scrapes /metrics every
                       1 s: running / waiting requests, GPU KV usage, preemptions, offload load / store bytes. Reports the
                       maximum concurrent running ("held"), queueing, preemptions, wall and aggregate prefill tok/s. N > the
                       GPU pool's concurrency, so every cap phase also flushes the GPU prefix cache.
  press   --pool T --conc 4  N = ceil(1.15 x T / 250K) distinct 250K documents at C4, max_tokens 1 (the flush before turn 5; aggregate C4 prefill tok/s).
  decode               4 fresh 250K documents primed one by one (max_tokens 1: C1 cold prefill tok/s, n=4), then decode with
                       ignore_eos, max_tokens 512: C1 on documents 1 and 2, then C4 on all four (prefix hits; decode tok/s =
                       (completion_tokens - 1) / (t_last - t_first) per stream, plus the C4 aggregate).
Per item: prompt_tokens, TTFT, wall, finish, and /metrics deltas around it (GPU prefix hit tokens, connector (host) hit tokens,
offload load / store bytes). Errors are rows; 2 consecutive timeouts or a dead server end the phase (skipped_from / server_dead).
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_probe_2hop_ds as B  # noqa: E402

KW = {"thinking": False}
CORR_L, CORR_SEEDS = 250000, (1016, 1017)
TURN_P_L, TURN_TOOL_L, SESSIONS, TURNS = 250000, 8000, 8, 5
TURN_MAX, CAP_MAX, CORR_MAX, DEC_MAX = 96, 16, 64, 512
DEC_L = 250000
DOC_FILL = 0.98   # batch documents are built at 0.98 x L tokens (inside the [0.95, 1.05] band; a 1M document stays under --max-model-len)
PT_LO, PT_HI = B.PT_LO, B.PT_HI
# metric name fragments -> our keys (summed over label sets; counters may carry _total)
MKEYS = {
    "gpu_hit_tok": ("vllm:prefix_cache_hits",), "gpu_query_tok": ("vllm:prefix_cache_queries",),
    "ext_hit_tok": ("vllm:external_prefix_cache_hits", "vllm:connector_prefix_cache_hits"),
    "ext_query_tok": ("vllm:external_prefix_cache_queries", "vllm:connector_prefix_cache_queries"),
    "load_bytes": ("vllm:kv_offload_load_bytes",), "store_bytes": ("vllm:kv_offload_store_bytes",),
    "load_time": ("vllm:kv_offload_load_time",), "store_time": ("vllm:kv_offload_store_time",),
    "preempt": ("vllm:num_preemptions",), "running": ("vllm:num_requests_running",), "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "alloc_fail": ("vllm:kv_offload_allocation_failure",),
}


def parse_metrics(text):
    """Prometheus text -> {key: summed value} for MKEYS (pure; tested). Histograms' _bucket/_count/_sum lines are ignored."""
    out = {k: 0.0 for k in MKEYS}
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith(("_bucket", "_count", "_sum", "_created")):
            continue
        base = name[:-6] if name.endswith("_total") else name
        for k, frags in MKEYS.items():
            if base in frags:
                try:
                    out[k] += float(line.rsplit(" ", 1)[-1])
                    seen.add(k)
                except ValueError:
                    pass
    out["_seen"] = sorted(seen)
    return out


def scrape(base):
    try:
        return parse_metrics(urllib.request.urlopen(base + "/metrics", timeout=30).read().decode())
    except Exception:  # noqa: BLE001
        return None


def delta(m0, m1):
    if not m0 or not m1:
        return None
    return {k: round(m1[k] - m0[k], 3) for k in ("gpu_hit_tok", "gpu_query_tok", "ext_hit_tok", "ext_query_tok", "load_bytes",
                                                   "store_bytes", "load_time", "store_time", "preempt", "alloc_fail")}


def text_at(text, start, nchars):
    """nchars of real text starting at a word boundary at `start` (wrapping around the corpus)."""
    start %= len(text)
    s = text.find(" ", start) + 1 or start
    t = text[s:]
    while len(t) < nchars:
        t += "\n\n" + text
    return t[:nchars]


def body(model, messages, max_tokens, extra=None):
    b = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": dict(KW), "messages": messages}
    b.update(extra or {})
    return b


def one(base, b, timeout):
    """-> record dict (never raises): stream() fields + metric deltas, or error."""
    m0 = scrape(base)
    rec = {}
    try:
        r = B.stream(base, b, timeout)
        rec.update(ptoks=r["ptoks"], ctoks=r["ctoks"], ttft_s=r["ttft_s"], wall_s=r["wall_s"], finish=r["finish"],
                   t_first=r["t_first"], t_last=r["t_last"], content=r["content"])
        rec["prefill_tok_s"] = round(r["ptoks"] / r["ttft_s"], 1) if r["ptoks"] and r["ttft_s"] else None
    except urllib.error.HTTPError as e:
        rec["error"] = "HTTP %d %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    rec["m"] = delta(m0, scrape(base))
    return rec


def is_timeout(rec):
    e = rec.get("error", "")
    return "timed out" in e or "Timeout" in e


class Guard:
    """2 consecutive timeouts or a dead server -> the rest of the phase is skipped."""
    def __init__(self, base):
        self.base, self.consec, self.skip = base, 0, None

    def after(self, rec):
        if "error" not in rec:
            self.consec = 0
            return
        if is_timeout(rec):
            self.consec += 1
            if self.consec >= 2:
                self.skip = "2 consecutive timeouts"
        if not B.alive(self.base):
            self.skip = "server dead"


def valid_len(recs, L):
    pts = [r.get("ptoks") or 0 for r in recs if "error" not in r]
    return bool(recs) and len(pts) == len(recs) and all(PT_LO * L <= p <= PT_HI * L for p in pts)


def q(xs, f):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, math.ceil(f * len(xs)) - 1))]


def load_state(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_state(path, st):
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, path)


def strip(rec, keep_content=0):
    c = rec.pop("content", None)
    if keep_content and c is not None:
        rec["answer"] = c[:keep_content]
    elif c is not None:
        rec["answer_tail"] = c[-80:]
    return rec


# ---------------------------------------------------------------- phases
def phase_corr(a, text, cpt):
    stp = os.path.join(a.rundir, "state-corr.json")
    st = load_state(stp)
    if a.pass_ == 1:
        st = {"nonce": a.nonce, "cpt": cpt}
        save_state(stp, st)
    elif not st:
        raise SystemExit("corr pass 2 needs state-corr.json from pass 1")
    nonce, cpt = st["nonce"], st["cpt"]
    res = {"phase": "corr", "pass": a.pass_, "length": CORR_L, "req": []}
    g = Guard(a.base)
    for seed in CORR_SEEDS:
        needles, qs, codes = B.plan(seed, CORR_L)
        doc = "[kvo %s-corr-%d]\n" % (nonce, seed) + B.build(text, int(CORR_L * cpt), needles)
        for qi, qq in enumerate(qs):
            rec = {"seed": seed, "i": qi, "kind": qq["kind"]}
            if g.skip:
                rec["error"] = "skipped: " + g.skip
            else:
                rec.update(one(a.base, body(a.model, [{"role": "user", "content": doc + "\n\nQuestion: " + qq["q"] +
                                                        qq.get("fmt", " Answer with the 6-digit code only.")}], CORR_MAX), a.timeout))
                g.after(rec)
                if "error" not in rec:
                    rec["ok"] = qq["want"] in rec["content"]
            res["req"].append(strip(rec, keep_content=400))
            print(json.dumps(res["req"][-1]), flush=True)
    res["valid"] = valid_len(res["req"], CORR_L)
    res["correct"] = sum(1 for r in res["req"] if r.get("ok"))
    res["n"] = len(res["req"])
    return res


def session_turn_msg(text, cpt, s, k):
    off = 2_000_000 + ((s - 1) * TURNS + (k - 1)) * int(TURN_TOOL_L * cpt * 1.1)
    head = "[session %d turn %d] Tool output follows.\n" % (s, k) if k > 1 else \
        "[session %d] You are agent %d. You will receive tool outputs; for each, reply with one sentence naming its main topic.\n" \
        "Tool output follows.\n" % (s, s)
    return head + text_at(text, off, int(TURN_TOOL_L * cpt)) + "\n\nIn one sentence, what is the main topic of this tool output?"


def apply_scale(f):
    """TESTS ONLY (--scale, default 1 = the pre-registered sizes): shrink every token length so the CPU tests fit a small corpus."""
    global CORR_L, TURN_P_L, TURN_TOOL_L, DEC_L
    CORR_L, TURN_P_L, TURN_TOOL_L, DEC_L = (max(200, int(x * f)) for x in (250000, 250000, 8000, 250000))


def phase_turn(a, text, cpt):
    stp = os.path.join(a.rundir, "state-turns.json")
    st = load_state(stp)
    if a.k == 1:
        st = {"nonce": a.nonce, "cpt": cpt, "hist": {str(s): [] for s in range(1, SESSIONS + 1)}, "done": 0}
    elif st.get("done") != a.k - 1:
        raise SystemExit("turn %d needs turn %d done first (state says %s)" % (a.k, a.k - 1, st.get("done")))
    nonce, cpt = st["nonce"], st["cpt"]
    P = "[kvo %s-shared] Shared reference material for all agents follows.\n" % nonce + text_at(text, 0, int(TURN_P_L * cpt))
    res = {"phase": "turn", "k": a.k, "p_len": TURN_P_L, "tool_len": TURN_TOOL_L, "req": []}
    g = Guard(a.base)
    for s in range(1, SESSIONS + 1):
        h = st["hist"][str(s)]
        h.append({"role": "user", "content": session_turn_msg(text, cpt, s, a.k)})
        rec = {"session": s, "turn": a.k, "role": "first" if s == 1 else "follower"}
        if g.skip:
            rec["error"] = "skipped: " + g.skip
            h.pop()
        else:
            rec.update(one(a.base, body(a.model, [{"role": "system", "content": P}] + h, TURN_MAX), a.timeout))
            g.after(rec)
            if "error" in rec:
                h.pop()
            else:
                h.append({"role": "assistant", "content": rec["content"]})
        res["req"].append(strip(rec))
        print(json.dumps(res["req"][-1]), flush=True)
    st["done"] = a.k if not any("error" in r for r in res["req"]) else -1
    save_state(stp, st)
    lo = TURN_P_L + (a.k - 1) * TURN_TOOL_L
    res["valid"] = all("error" not in r for r in res["req"]) and all(
        PT_LO * lo <= (r.get("ptoks") or 0) <= PT_HI * (lo + TURN_TOOL_L) + TURNS * TURN_MAX for r in res["req"])
    return res


def run_batch(a, docs, max_tokens, conc, tag, extra=None):
    """docs sent with at most `conc` in flight; a 1 s /metrics sampler runs alongside. -> (recs, samples, wall)"""
    recs = [None] * len(docs)
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            m = scrape(a.base)
            if m:
                samples.append({"t": round(time.time(), 2), **{k: m[k] for k in ("running", "waiting", "kv_usage", "preempt",
                                                                                   "load_bytes", "store_bytes", "ext_hit_tok")}})
            stop.wait(1.0)

    sem = threading.Semaphore(conc)

    def worker(i):
        with sem:
            r = {"i": i}
            r.update(one(a.base, body(a.model, [{"role": "user", "content": docs[i]}], max_tokens, extra), a.timeout))
            recs[i] = strip(r)
            print(json.dumps(recs[i]), flush=True)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    m0 = scrape(a.base)
    t0 = time.time()
    ws = [threading.Thread(target=worker, args=(i,)) for i in range(len(docs))]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    wall = round(time.time() - t0, 2)
    stop.set()
    th.join(timeout=5)
    return recs, samples, wall, delta(m0, scrape(a.base))


def distinct_docs(text, cpt, L, n, tag, nonce, q_):
    step = max(1, len(text) // max(n, 1))
    return ["[kvo %s-%s-%d-%d]\n" % (nonce, tag, L, j) + text_at(text, 7919 + j * step, int(L * cpt * DOC_FILL)) + "\n\n" + q_ for j in range(n)]


def batch_summary(recs, samples, wall, d, L):
    ok = [r for r in recs if r and "error" not in r]
    pts = sum(r.get("ptoks") or 0 for r in ok)
    pre0 = samples[0]["preempt"] if samples else None
    return {"n": len(recs), "n_ok": len(ok), "errors": len(recs) - len(ok), "wall_s": wall,
            "max_running": max((s["running"] for s in samples), default=None),
            "max_waiting": max((s["waiting"] for s in samples), default=None),
            "peak_kv_usage": max((s["kv_usage"] for s in samples), default=None),
            "preemptions": (d or {}).get("preempt"), "preempt_sampled": (samples[-1]["preempt"] - pre0) if samples else None,
            "load_bytes": (d or {}).get("load_bytes"), "store_bytes": (d or {}).get("store_bytes"),
            "ext_hit_tok": (d or {}).get("ext_hit_tok"), "agg_prefill_tok_s": round(pts / wall, 1) if wall else None,
            "ttft_p50": q([r.get("ttft_s") for r in ok], 0.5), "ttft_p90": q([r.get("ttft_s") for r in ok], 0.9),
            "valid": valid_len([r for r in recs if r], L) and len(ok) == len(recs), "samples": len(samples)}


def phase_cap(a, text, cpt):
    L, n = int(a.length), int(a.n)
    docs = distinct_docs(text, cpt, L, n, "cap", a.nonce, "In one sentence, what is this text mostly about?")
    recs, samples, wall, d = run_batch(a, docs, CAP_MAX, n, "cap")
    res = {"phase": "cap", "length": L, "n": n, "conc": n, "req": recs, "samples": samples, "delta": d}
    res["summary"] = batch_summary(recs, samples, wall, d, L)
    return res


def phase_press(a, text, cpt):
    n, c = (int(a.n) or math.ceil(1.15 * a.pool / DEC_L)), int(a.conc or 4)
    docs = distinct_docs(text, cpt, DEC_L, n, "press", a.nonce, "Say OK.")
    recs, samples, wall, d = run_batch(a, docs, 1, c, "press")
    res = {"phase": "press", "length": DEC_L, "n": n, "conc": c, "req": recs, "samples": samples, "delta": d}
    res["summary"] = batch_summary(recs, samples, wall, d, DEC_L)
    return res


def dec_rate(r):
    if not r or "error" in r or not r.get("ctoks") or not r.get("t_first") or not r.get("t_last") or r["t_last"] <= r["t_first"]:
        return None
    return round((r["ctoks"] - 1) / (r["t_last"] - r["t_first"]), 1)


def phase_decode(a, text, cpt):
    qd = "Write a long, detailed, section-by-section summary of the text above."
    docs = distinct_docs(text, cpt, DEC_L, 4, "dec", a.nonce, qd)
    res = {"phase": "decode", "length": DEC_L, "prime": [], "c1": [], "c4": []}
    g = Guard(a.base)
    for i, doc in enumerate(docs):            # cold C1 prefills (prefill tok/s, n=4); leaves the 4 documents GPU-resident
        r = {"i": i, "error": "skipped: " + g.skip} if g.skip else {"i": i, **one(a.base, body(a.model, [{"role": "user", "content": doc}], 1), a.timeout)}
        g.after(r)
        res["prime"].append(strip(r))
    ext = {"ignore_eos": True}
    for i in (0, 1):
        r = {"i": i, "error": "skipped: " + g.skip} if g.skip else {"i": i, **one(a.base, body(a.model, [{"role": "user", "content": docs[i]}], DEC_MAX, ext), a.timeout)}
        g.after(r)
        r["decode_tok_s"] = dec_rate(r)
        res["c1"].append(strip(r))
    if not g.skip:
        recs, samples, wall, d = run_batch(a, docs, DEC_MAX, 4, "dec4", ext)
        for r in recs:
            if r:
                r["decode_tok_s"] = dec_rate(r)
        res["c4"] = recs
        ok = [r for r in recs if r and r.get("decode_tok_s")]
        tf = [r["t_first"] for r in ok]
        tl = [r["t_last"] for r in ok]
        res["c4_agg_tok_s"] = round(sum(r["ctoks"] - 1 for r in ok) / (max(tl) - min(tf)), 1) if ok and max(tl) > min(tf) else None
    pr = [r for r in res["prime"] if "error" not in r]
    res["summary"] = {"prefill_c1_tok_s_median": q([r.get("prefill_tok_s") for r in pr], 0.5), "prime_valid": valid_len(res["prime"], DEC_L),
                      "decode_c1_tok_s_median": q([r.get("decode_tok_s") for r in res["c1"]], 0.5),
                      "decode_c4_per_stream_median": q([r.get("decode_tok_s") for r in res["c4"] if r], 0.5),
                      "decode_c4_agg_tok_s": res.get("c4_agg_tok_s"),
                      "ctoks": [r.get("ctoks") for r in res["c1"] + [x for x in res["c4"] if x]]}
    return res


PHASES = {"corr": phase_corr, "turn": phase_turn, "cap": phase_cap, "press": phase_press, "decode": phase_decode}


def tag_of(a):
    return {"corr": "corr-p%d" % a.pass_, "turn": "turn-%d" % (a.k or 0), "cap": "cap-%s" % a.length, "press": "press",
            "decode": "decode"}[a.phase]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("fit"); ap.add_argument("rundir"); ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--k", type=int, default=0); ap.add_argument("--length", type=int, default=0); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--conc", type=int, default=0); ap.add_argument("--pass", dest="pass_", type=int, default=1, choices=(1, 2))
    ap.add_argument("--nonce", default=""); ap.add_argument("--timeout", type=float, default=900.0); ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="tests only: shrink the pre-registered token lengths")
    a = ap.parse_args()
    if a.scale != 1.0:
        apply_scale(a.scale)
    a.base = a.base.rstrip("/")
    a.nonce = a.nonce or str(int(time.time()) % 1000000)
    if a.phase == "turn" and not 1 <= a.k <= TURNS:
        raise SystemExit("--k 1..%d" % TURNS)
    if a.phase == "cap" and (a.length <= 0 or a.n <= 0):
        raise SystemExit("cap needs --length and --n")
    if a.phase == "press" and a.n <= 0 and a.pool <= 0:
        raise SystemExit("press needs --pool (or --n)")
    text = B.corpus(a.fit)
    cpt = B.calibrate(a.base, a.model, text)
    res = PHASES[a.phase](a, text, cpt)
    res.update(chars_per_token=round(cpt, 3), nonce=a.nonce, scale=a.scale, t_end=round(time.time(), 1))
    out = os.path.join(a.rundir, tag_of(a) + ".json")
    json.dump(res, open(out, "w"), indent=1)
    sm = {k: res[k] for k in ("phase", "valid", "correct", "n", "summary") if k in res}
    sm["out"] = out
    print(json.dumps(sm), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
