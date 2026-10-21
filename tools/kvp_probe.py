#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""kvp_probe.py API MODEL FIT.jsonl OUTDIR PHASE [--length L] [--sessions N] [--rounds R] [--scale X] [--timeout S]
results/kv-offload.md, test 1. One serving arm = one GPU (RTX PRO 6000 over PCIe 5, or GB300 over NVLink-C2C) running
vLLM af7f9488 with the native OffloadingConnector (host tier in Grace LPDDR). Phases (same text on both GPUs; the prompt text is never
written to disk, only its sha256):

  ladder --length L   one document of L tokens (public news text, fit.jsonl) with 4 inserted facts at depths 0.15 / 0.40 / 0.65 /
                      0.90; questions Q1..Q4 differ only after the document. Sequence (C1, temperature 0, thinking off, max_tokens 48):
                        reset(gpu+host) -> Q1 cold1        (full prefill = RECOMPUTE)
                        for k in 1..4: reset(gpu only) -> Qk reload_k   (prefix in host memory only = RELOAD over the link)
                        for k in 1..4: Qk gpu_k            (prefix in GPU memory = GPU HIT)
                        reset(gpu+host) -> Q1 cold2        (second recompute; cold2 vs cold1 = the nondeterminism baseline)
                      reset = POST /reset_prefix_cache[?reset_external=true] (dev endpoint; VLLM_SERVER_DEV_MODE=1, server bound to
                      127.0.0.1), retried until {"success": true} (an in-flight async store makes it fail; the scheduler then retries).
  rr --sessions N --length L --rounds R
                      N distinct documents of L tokens whose KV together exceeds the GPU pool; R rounds of one question per session,
                      round-robin at C1, after one reset(gpu+host). Round 1 is all recompute; later rounds find each prefix in host
                      memory (evicted from the GPU by the other sessions) or still on the GPU.
  decode --length L   one document, primed once, then 2 requests (GPU hit) with ignore_eos, max_tokens 512: decode tok/s =
                      (completion_tokens - 1) / (t_last - t_first) from the stream. The KV stays in GPU memory: the reference speed.
  calib               chars per token of the corpus through POST /tokenize (written to OUTDIR/calib.json, used by the other phases).

Every request records: TTFT (first content token, streamed), total time, prompt / cached tokens (usage.prompt_tokens_details), the
answer (<= 400 chars), the prompt sha256, and /metrics deltas around it: vllm:kv_offload_{load,store}_{bytes,time}, external and GPU
prefix-cache hits. Effective link bandwidth per request = load_bytes delta / load_time delta (the connector's own transfer timer).
Validity: prompt_tokens within [0.95, 1.05] x L and no error. --scale shrinks every length (tests only). Rows: OUTDIR/<phase>[-L].jsonl;
the last stdout line is a one-line summary.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

DEPTHS = (0.15, 0.40, 0.65, 0.90)
ITEMS = ("north archive", "harbor ledger", "valley survey", "summit report")
METRICS = ("vllm:kv_offload_load_bytes", "vllm:kv_offload_load_time", "vllm:kv_offload_store_bytes", "vllm:kv_offload_store_time",
           "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
           "vllm:num_preemptions")
DOC_SEED = {"ladder": 101, "rr": 202, "decode": 303}


# ---------------------------------------------------------------- pure helpers (tested)
def parse_metrics(text):
    """Prometheus text -> {base name: summed value}; counters with or without _total; histogram buckets ignored."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)", line)
        if not m:
            continue
        name = m.group(1)
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        base = name[:-6] if name.endswith("_total") else name
        if base in METRICS:
            try:
                out[base] = out.get(base, 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def mdelta(a, b):
    return {k: round(b.get(k, 0.0) - a.get(k, 0.0), 6) for k in METRICS}


def link_gbps(d, direction="load"):
    t = d.get("vllm:kv_offload_%s_time" % direction, 0.0)
    b = d.get("vllm:kv_offload_%s_bytes" % direction, 0.0)
    return round(b / t / 1e9, 2) if t > 0 and b > 0 else None


def code_for(seed, k):
    return "%06d" % random.Random(seed * 1000 + k).randrange(100000, 999999)


def fact(seed, k):
    return "Archive note: the reference code for the %s is %s." % (ITEMS[k], code_for(seed, k))


def question(k):
    return "\n\nQuestion: What is the reference code for the %s? Answer with the six digits only." % ITEMS[k]


def corpus(path):
    return [json.loads(line)["content"] for line in open(path) if line.strip()]


def build_doc(arts, nchars, seed):
    """Deterministic document of ~nchars from the corpus (start article chosen by seed) with the 4 facts inserted at DEPTHS."""
    i = random.Random(seed).randrange(len(arts))
    parts, n = [], 0
    while n < nchars:
        a = arts[i % len(arts)]
        parts.append(a)
        n += len(a) + 2
        i += 1
    text = "\n\n".join(parts)[:nchars]
    for k in reversed(range(len(DEPTHS))):   # insert deepest first so earlier offsets stay valid
        p = int(len(text) * DEPTHS[k])
        s = text.rfind(". ", max(0, p - 2000), p)   # snap back to a sentence end within 2,000 chars, else insert at p
        p = s + 2 if s >= 0 else p
        text = text[:p] + fact(seed, k) + "\n" + text[p:]
    return "Read the documents below. Each answer is stated in them.\n\n" + text


def answer_ok(ans, seed, k):
    return code_for(seed, k) in (ans or "")


def valid(prompt_tokens, target):
    return bool(prompt_tokens) and 0.95 * target <= prompt_tokens <= 1.05 * target


def ladder_plan():
    """The ladder's request order: (state, question index, reset kind)."""
    p = [("cold1", 0, "all")]
    p += [("reload%d" % (k + 1), k, "gpu") for k in range(4)]
    p += [("gpu%d" % (k + 1), k, None) for k in range(4)]
    p += [("cold2", 0, "all")]
    return p


# ---------------------------------------------------------------- HTTP
class Api:
    def __init__(self, base, model, timeout):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout

    def _post(self, path, payload=None, timeout=None):
        data = json.dumps(payload).encode() if payload is not None else b""
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def metrics(self):
        try:
            return parse_metrics(urllib.request.urlopen(self.base + "/metrics", timeout=30).read().decode())
        except Exception:  # noqa: BLE001
            return {}

    def tokens(self, text):
        r = json.load(self._post("/tokenize", {"model": self.model, "prompt": text}, 120))
        return int(r.get("count") or len(r.get("tokens", [])))

    def reset(self, external):
        path = "/reset_prefix_cache" + ("?reset_external=true" if external else "")
        for i in range(60):
            try:
                if json.load(self._post(path, None, 60)).get("success"):
                    return i + 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        raise RuntimeError("reset_prefix_cache(external=%s) never succeeded" % external)

    def chat(self, content, max_tokens=48, ignore_eos=False):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        if ignore_eos:
            body["ignore_eos"] = True
        t0 = time.perf_counter()
        first = last = None
        text, usage, nchunks = [], {}, 0
        with self._post("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices") or []:
                    piece = (c.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                        nchunks += 1
                        text.append(piece)
        t1 = time.perf_counter()
        det = usage.get("prompt_tokens_details") or {}
        return {"ttft_s": round((first or t1) - t0, 4), "total_s": round(t1 - t0, 4),
                "decode_s": round((last - first), 4) if first is not None and last is not None else None,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": det.get("cached_tokens"), "answer": "".join(text)[:400], "chunks": nchunks}


# ---------------------------------------------------------------- phases
def one(api, out, row, content, **kw):
    m0 = api.metrics()
    try:
        r = api.chat(content, **kw)
        r["error"] = None
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e)[:300]}
    m1 = api.metrics()
    d = mdelta(m0, m1)
    row.update(r)
    row.update({"prompt_sha256": hashlib.sha256(content.encode()).hexdigest(), "metrics_delta": d,
                "load_gbps": link_gbps(d, "load"), "store_gbps": link_gbps(d, "store"), "t": round(time.time(), 1)})
    out.write(json.dumps(row, sort_keys=True) + "\n")
    out.flush()
    return row


def cpt(outdir):
    try:
        return float(json.load(open(os.path.join(outdir, "calib.json")))["chars_per_token"])
    except Exception:  # noqa: BLE001
        return None


def phase_calib(api, arts, outdir):
    sample = "\n\n".join(arts[:40])[:200000]
    n = api.tokens(sample)
    c = round(len(sample) / n, 4)
    json.dump({"chars": len(sample), "tokens": n, "chars_per_token": c}, open(os.path.join(outdir, "calib.json"), "w"))
    return "calib: %d chars = %d tokens, %.3f chars/token" % (len(sample), n, c)


def target_chars(L, c, qtok=110):   # question + 4 inserted facts + header ~ 110 tokens
    return int((L - qtok) * c)


def phase_ladder(api, arts, outdir, L, c):
    seed = DOC_SEED["ladder"] + L
    doc = build_doc(arts, target_chars(L, c), seed)
    rows = []
    with open(os.path.join(outdir, "ladder-%d.jsonl" % L), "a") as out:
        for state, k, reset in ladder_plan():
            tries = api.reset(reset == "all") if reset else 0
            row = {"phase": "ladder", "length": L, "state": state, "q": k + 1, "reset": reset, "reset_tries": tries}
            r = one(api, out, row, doc + question(k))
            r["valid"] = r.get("error") is None and valid(r.get("prompt_tokens"), L)
            r["correct"] = answer_ok(r.get("answer"), seed, k)
            rows.append(r)
    with open(os.path.join(outdir, "ladder-%d.jsonl" % L), "w") as out:
        for r in rows:
            out.write(json.dumps(r, sort_keys=True) + "\n")
    g = {r["state"]: r for r in rows}
    return "ladder %d: cold1 %.2fs reload1 %.2fs gpu1 %.2fs cold2 %.2fs; reload load %s GB/s; valid %d/%d; correct %d/%d" % (
        L, g["cold1"].get("ttft_s") or -1, g["reload1"].get("ttft_s") or -1, g["gpu1"].get("ttft_s") or -1, g["cold2"].get("ttft_s") or -1,
        g["reload1"].get("load_gbps"), sum(r["valid"] for r in rows), len(rows), sum(r["correct"] for r in rows), len(rows))


def phase_rr(api, arts, outdir, L, c, sessions, rounds):
    docs = [(DOC_SEED["rr"] + 17 * s, build_doc(arts, target_chars(L, c), DOC_SEED["rr"] + 17 * s)) for s in range(sessions)]
    api.reset(True)
    n_ok = 0
    with open(os.path.join(outdir, "rr-%d.jsonl" % L), "w") as out:
        for rd in range(rounds):
            for s, (seed, doc) in enumerate(docs):
                k = rd % 4
                row = {"phase": "rr", "length": L, "round": rd + 1, "session": s + 1, "q": k + 1}
                r = one(api, out, row, doc + question(k))
                ok = r.get("error") is None and valid(r.get("prompt_tokens"), L)
                n_ok += ok
    return "rr %d x %d tokens x %d rounds: %d valid" % (sessions, L, rounds, n_ok)


def phase_decode(api, arts, outdir, L, c):
    seed = DOC_SEED["decode"] + L
    doc = build_doc(arts, target_chars(L, c), seed)
    res = []
    with open(os.path.join(outdir, "decode-%d.jsonl" % L), "w") as out:
        one(api, out, {"phase": "decode", "length": L, "state": "prime"}, doc + question(0))
        for i in range(2):
            r = one(api, out, {"phase": "decode", "length": L, "state": "run%d" % (i + 1)}, doc + question(1 + i),
                    max_tokens=512, ignore_eos=True)
            ct, ds = r.get("completion_tokens"), r.get("decode_s")
            res.append(round((ct - 1) / ds, 2) if ct and ds else None)
    return "decode at %d: %s tok/s (C1, GPU-resident KV)" % (L, res)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("api"), ap.add_argument("model"), ap.add_argument("fit"), ap.add_argument("outdir")
    ap.add_argument("phase", choices=("calib", "ladder", "rr", "decode"))
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    api = Api(a.api, a.model, a.timeout)
    arts = corpus(a.fit)
    if a.phase == "calib":
        print(phase_calib(api, arts, a.outdir))
        return 0
    c = cpt(a.outdir)
    if not c:
        print("NO CALIB (run the calib phase first)")
        return 2
    L = max(512, int(a.length * a.scale))
    if a.phase == "ladder":
        print(phase_ladder(api, arts, a.outdir, L, c))
    elif a.phase == "rr":
        print(phase_rr(api, arts, a.outdir, L, c, a.sessions, a.rounds))
    else:
        print(phase_decode(api, arts, a.outdir, L, c))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
