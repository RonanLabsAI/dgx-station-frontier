#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# RonanLabs 2026-10-10 (Stage 0 amendment C2). Stdlib only.
"""C2 long-context accuracy eval (LC-1008) for DS-V4.1-Flash: run one arm, score it, and compute the registered
paired non-inferiority verdict.

  lc_eval.py verify <items.jsonl.gz> <decompressed sha256>                  -> exit 0 iff the item file is the pinned one
  lc_eval.py run <label> <out.jsonl> --items <gz> [--port 8009] [--model M] [--limit N]
  lc_eval.py score <out.jsonl>                                              -> "LC <label> k/n = p% ..." + per task/length
  lc_eval.py verdict <A0a.jsonl> <U300.jsonl> [<A0R.jsonl>] [--margin 2.0]  -> "C2 NON-INFERIOR|INFERIOR|INCONCLUSIVE|
                                                                               NOT INTERPRETABLE: ..." (+ JSON)
Request hygiene (E2 convention for identical-input Q3 prompts): every item gets its own cache_salt
"<run nonce>-<item id>" (vLLM salts the block hash from block 0, i.e. a nonce at the START, without changing the
tokens, so every arm sees byte-identical prompts); the prefix-cache hit share of each request is read from /metrics
around it; > 2% -> one retry on a fresh salt -> still > 2% = CONTAMINATED (the item is dropped from the pair for every
arm, never scored). Greedy (temperature 0), thinking off, max_tokens 160, one request at a time, fixed item order.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_score import score  # noqa: E402

Z95 = 1.6448536269514722   # one-sided 95%


def load_items(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def verify(a):
    h = hashlib.sha256()
    with gzip.open(a.items, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ok = h.hexdigest() == a.sha
    print(f"LC items {'OK' if ok else 'MISMATCH'}: decompressed sha256 {h.hexdigest()} (pinned {a.sha})")
    raise SystemExit(0 if ok else 1)


def _hits(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    v = [0.0, 0.0]
    for l in txt.splitlines():
        if l.startswith("vllm:prefix_cache_hits_total"):
            v[0] += float(l.split()[-1])
        elif l.startswith("vllm:prefix_cache_queries_total"):
            v[1] += float(l.split()[-1])
    return v


def post(base, body, salt, timeout=900):
    """-> (response or None, hit %, contaminated, error, salt used)."""
    for attempt in (1, 2):
        s = salt if attempt == 1 else f"{salt}-retry-{time.time_ns()}"
        try:
            h0 = _hits(base)
            req = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps({**body, "cache_salt": s}).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            h1 = _hits(base)
        except (urllib.error.URLError, OSError, ValueError) as ex:
            return None, None, False, f"{ex.__class__.__name__}: {str(ex)[:200]}", s
        hit = 100 * (h1[0] - h0[0]) / (h1[1] - h0[1]) if h1[1] > h0[1] else 0.0
        if hit <= 2.0:
            return r, hit, False, None, s
    return r, hit, True, None, s


def run(a):
    items = load_items(a.items)
    if a.limit:
        items = items[:a.limit]
    base = f"http://127.0.0.1:{a.port}"
    nonce = f"lc-{a.label}-{time.time_ns()}"
    t_start = time.time()
    n_ok = 0
    with open(a.out, "w") as f:
        f.write(json.dumps({"header": True, "label": a.label, "nonce": nonce, "items": len(items), "model": a.model,
                            "max_tokens": a.max_tokens, "t0": t_start}) + "\n")
        for it in items:
            t0 = time.time()
            r, hit, dirty, err, salt = post(base, {
                "model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
                "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": it["content"]}]}, f"{nonce}-{it['id']}")
            row = {"k": it["k"], "id": it["id"], "task": it["task"], "L": it["L"], "salt": salt,
                   "s": round(time.time() - t0, 3), "prefix_hit_pct": hit, "contaminated": dirty, "error": err}
            if r is not None:
                ch = r["choices"][0]
                msg = ch.get("message") or {}
                reply = msg.get("content") or ""
                ok, rec = score(it, reply)
                row.update({"reply": reply[:800], "finish": ch.get("finish_reason"),
                            "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                            "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                            "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
                            "pass": bool(ok), "recall": rec})
                n_ok += bool(ok) and not dirty
            f.write(json.dumps(row) + "\n")
            f.flush()
            if it["k"] % 50 == 0:
                print(f"{a.label} {it['k'] + 1}/{len(items)} correct so far {n_ok} ({time.time() - t_start:.0f} s)",
                      flush=True)
    print(f"{a.label} done: {len(items)} items in {time.time() - t_start:.0f} s; wrote {a.out}")


def load_run(path):
    rows, hdr = {}, None
    with open(path) as f:
        for l in f:
            r = json.loads(l)
            if r.get("header"):
                hdr = r
            else:
                rows[r["id"]] = r
    return hdr, rows


def valid(r):
    return r is not None and not r.get("contaminated") and not r.get("error") and "pass" in r


def summarize(rows):
    v = [r for r in rows.values() if valid(r)]
    by = {}
    for r in v:
        for key in (r["task"], f"{r['L'] // 1024}K"):
            by.setdefault(key, [0, 0])
            by[key][0] += r["pass"]
            by[key][1] += 1
    return sum(r["pass"] for r in v), len(v), by


def score_cmd(a):
    hdr, rows = load_run(a.x)
    k, n, by = summarize(rows)
    bad = len(rows) - n
    print(f"LC {hdr['label'] if hdr else '?'} {k}/{n} = {100 * k / max(1, n):.2f}% (invalid {bad}; "
          f"finish=length {sum(1 for r in rows.values() if r.get('finish') == 'length')})")
    print("  " + "  ".join(f"{key} {c}/{t}" for key, (c, t) in sorted(by.items())))


def paired(A, X, ids):
    """U = X - A over the common valid items. Agresti-Min (2005) adjusted Wald for a paired difference: 0.5 added to
    each cell of the 2x2 table. Returns dict with points in percentage points."""
    n11 = sum(1 for i in ids if A[i]["pass"] and X[i]["pass"])
    b = sum(1 for i in ids if A[i]["pass"] and not X[i]["pass"])      # A right, X wrong
    c = sum(1 for i in ids if not A[i]["pass"] and X[i]["pass"])      # A wrong, X right
    n = len(ids)
    bb, cc, nn = b + 0.5, c + 0.5, n + 2.0
    d = (cc - bb) / nn
    se = math.sqrt(max(0.0, (bb + cc) - (cc - bb) ** 2 / nn)) / nn
    return {"n": n, "acc_ref": 100 * (n11 + b) / max(1, n), "acc_arm": 100 * (n11 + c) / max(1, n),
            "diff_raw": 100 * (c - b) / max(1, n), "b_ref_only": b, "c_arm_only": c, "diff_adj": 100 * d,
            "se": 100 * se, "lb95": 100 * (d - Z95 * se), "ub95": 100 * (d + Z95 * se)}


def verdict(a):
    runs = [load_run(p) for p in [a.a0a, a.x] + ([a.a0r] if a.a0r else [])]
    (ha, A), (hx, X) = runs[0], runs[1]
    R = runs[2][1] if a.a0r else None
    n_items = (ha or {}).get("items") or len(A)
    notes, ok = [], True
    for lab, rows in (("A0a", A), ("U300", X)) + ((("A0R", R),) if R is not None else ()):
        nv = sum(1 for r in rows.values() if valid(r))
        if nv < 0.98 * n_items:
            ok = False
            notes.append(f"{lab} has {nv}/{n_items} valid items (< 98%)")
    ids = sorted(i for i in A if valid(A[i]) and valid(X.get(i)) and (R is None or valid(R.get(i))))
    P = paired(A, X, ids)
    out = {"margin_pts": a.margin, "primary_U300_vs_A0a": P}
    if P["acc_ref"] < 20.0:
        ok = False
        notes.append(f"A0a accuracy {P['acc_ref']:.1f}% < 20% (eval broken)")
    if R is not None:
        N = paired(A, R, ids)
        out["noise_A0R_vs_A0a"] = N
        if abs(N["diff_raw"]) >= a.margin:
            ok = False
            notes.append(f"A0R-vs-A0a {N['diff_raw']:+.2f} pts: same-recipe noise alone reaches the margin")
    per = {}
    for key in sorted({X[i]["task"] for i in ids} | {f"{X[i]['L'] // 1024}K" for i in ids}):
        sub = [i for i in ids if X[i]["task"] == key or f"{X[i]['L'] // 1024}K" == key]
        per[key] = paired(A, X, sub)
    out["per_cell_information"] = per
    if not ok:
        v = "NOT INTERPRETABLE"
    elif P["lb95"] > -a.margin:
        v = "NON-INFERIOR"
    elif P["ub95"] < -a.margin:
        v = "INFERIOR"
    else:
        v = "INCONCLUSIVE"
    out["verdict"] = v
    out["notes"] = notes
    line = (f"C2 {v}: U300 {P['acc_arm']:.2f}% vs A0a {P['acc_ref']:.2f}% on {P['n']} paired items; U300-A0a "
            f"{P['diff_raw']:+.2f} pts (adj {P['diff_adj']:+.2f}, one-sided 95% [{P['lb95']:.2f}, {P['ub95']:.2f}], "
            f"margin -{a.margin}); disagreeing A0a-only {P['b_ref_only']} / U300-only {P['c_arm_only']}")
    if R is not None:
        N = out["noise_A0R_vs_A0a"]
        line += (f"; noise A0R-A0a {N['diff_raw']:+.2f} pts [{N['lb95']:.2f}, {N['ub95']:.2f}], disagreeing "
                 f"{N['b_ref_only']}+{N['c_arm_only']}")
    if notes:
        line += "; " + "; ".join(notes)
    print(line)
    for key, q in per.items():
        print(f"  {key:5s} U300 {q['acc_arm']:6.2f}% A0a {q['acc_ref']:6.2f}% diff {q['diff_raw']:+6.2f} "
              f"(n {q['n']}, disagreeing {q['b_ref_only']}/{q['c_arm_only']})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    raise SystemExit(0 if v == "NON-INFERIOR" else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("verify")
    v.add_argument("items")
    v.add_argument("sha")
    r = sp.add_parser("run")
    r.add_argument("label")
    r.add_argument("out")
    r.add_argument("--items", required=True)
    r.add_argument("--port", default="8009")
    r.add_argument("--model", default="deepseek-v4.1-flash")
    r.add_argument("--max-tokens", type=int, default=160)
    r.add_argument("--limit", type=int, default=0)
    s = sp.add_parser("score")
    s.add_argument("x")
    d = sp.add_parser("verdict")
    d.add_argument("a0a")
    d.add_argument("x")
    d.add_argument("a0r", nargs="?")
    d.add_argument("--margin", type=float, default=2.0)
    d.add_argument("--json")
    a = ap.parse_args()
    {"verify": verify, "run": run, "score": score_cmd, "verdict": verdict}[a.cmd](a)
