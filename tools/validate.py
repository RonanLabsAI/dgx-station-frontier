#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Validate results.jsonl against its log excerpts, and every README number against results.jsonl.

usage: python3 tools/validate.py            (exit 0 = PASS)

results.jsonl
  * ids unique; required fields present for each kind:
      measured   metric value unit config date_pt versions image image_digest log log_match
      computed   same, minus log_match, plus op + inputs (strings found in the log) or rows_in (other row ids)
      withdrawn  log log_match, notes starting with WITHDRAWN
      public-reference  url source   (numbers published by others; quoted for comparison only)
  * every log file exists under logs/, every log_match / inputs string occurs in it;
  * computed rows are recomputed from their inputs (sum, mean, ratio, pct_change, difference, minutes_between, ratio_rows);
  * image_digest pins a sha256 digest; recipe folders exist.

README files (README.md, recipes/*/README.md, atlas/*/README.md)
  Fenced code, inline code spans, HTML comments and link URLs are ignored (only link text is read).
  * Inside <!-- results --> ... <!-- /results --> blocks, EVERY number must match a results.jsonl row value at the
    precision written (1,856 matches 1855.6; 49.8K matches 49788.5; signs are ignored), and the same table line must
    link that row's receipt: its log excerpt (ours) or its URL (public references).
  * Everywhere else, every number written with a performance unit (tok/s, GB/s, Gb/s, us, TFLOPS, W, %, x) must
    match a row. Numbers without such a unit (sizes, counts, dates, shapes) are not claims and are not checked.
  * Every relative link resolves to a file or folder in the repo.
"""
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERR = []


def err(msg):
    ERR.append(msg)


def num(s):
    return float(re.findall(r"\d[\d,]*(?:\.\d+)?", s)[-1].replace(",", ""))


# ------------------------------------------------------------------------------------------------ results.jsonl
rows = []
for i, line in enumerate(open(os.path.join(ROOT, "results.jsonl")), 1):
    if line.strip():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            err(f"results.jsonl:{i}: bad JSON ({e})")
byid = {}
for r in rows:
    if r.get("id") in byid:
        err(f"duplicate id {r.get('id')}")
    byid[r.get("id")] = r

REQ = {
    "measured": ["metric", "value", "unit", "config", "date_pt", "versions", "image", "image_digest", "log", "log_match"],
    "computed": ["metric", "value", "unit", "config", "date_pt", "versions", "image", "image_digest", "log", "op"],
    "withdrawn": ["metric", "value", "unit", "log", "log_match", "notes"],
    "public-reference": ["metric", "value", "unit", "url", "source"],
}


def logs_of(r):
    lg = r.get("log")
    return [lg] if isinstance(lg, str) else list(lg or [])


def close(a, b, rel=0.002, ab=0.06):
    return abs(a - b) <= max(ab, rel * abs(b))


for r in rows:
    rid, kind = r.get("id"), r.get("kind")
    if kind not in REQ:
        err(f"{rid}: unknown kind {kind!r}")
        continue
    for f in REQ[kind]:
        if r.get(f) in (None, "", []):
            err(f"{rid}: missing field {f}")
    if kind == "withdrawn" and not str(r.get("notes", "")).startswith("WITHDRAWN"):
        err(f"{rid}: withdrawn row notes must start with WITHDRAWN")
    if kind in ("measured", "computed") and "@sha256:" not in str(r.get("image_digest", "")):
        err(f"{rid}: image_digest must pin a repo@sha256 digest")
    if r.get("recipe") and not os.path.isdir(os.path.join(ROOT, r["recipe"])):
        err(f"{rid}: recipe folder {r['recipe']} missing")
    text = ""
    for lg in logs_of(r):
        p = os.path.join(ROOT, lg)
        if not lg.startswith("logs/") or not os.path.isfile(p):
            err(f"{rid}: log {lg} missing (must live under logs/)")
        else:
            text += open(p, errors="replace").read()
    for m in r.get("log_match", []) + r.get("inputs", []):
        if m not in text:
            err(f"{rid}: {m!r} not found in {logs_of(r)}")
    if kind == "computed":
        op, ins, v = r.get("op"), [num(x) for x in r.get("inputs", []) if re.search(r"\d", x)], float(r.get("value", "nan"))
        try:
            if op == "sum":
                got = sum(ins)
            elif op == "mean":
                got = sum(ins) / len(ins)
            elif op == "ratio":
                got = ins[0] / ins[1]
            elif op == "pct_change":
                if r["id"].startswith("dsv4pro-e3-vs-e2"):   # ratio of the 3-rep means of the two named rows
                    a, b = (byid[x]["value"] for x in (
                        ("dsv4pro-e2-real-c16", "dsv4pro-e3a16-real-c16") if "real" in r["id"] else ("dsv4pro-e2-catid-c16", "dsv4pro-e3a16-catid-c16")))
                    got = (b / a - 1) * 100
                else:
                    got = (ins[1] / ins[0] - 1) * 100
            elif op == "difference":
                got = ins[1] - ins[0]
            elif op == "minutes_between":
                h0, m0, s0 = map(int, re.findall(r"(\d\d):(\d\d):(\d\d)", r["inputs"][0])[0])
                h1, m1, s1 = map(int, re.findall(r"(\d\d):(\d\d):(\d\d)", r["inputs"][1])[0])
                got = (((h1 - 7) * 3600 + m1 * 60 + s1 - (h0 * 3600 + m0 * 60 + s0)) % 86400) / 60   # 2nd stamp UTC; PDT = UTC-7
            elif op == "ratio_rows":
                a, b = (byid[x] for x in r["rows_in"])
                got = a["value"] / b["value"]
            else:
                raise ValueError(op)
            if not close(got, v):
                err(f"{rid}: recomputed {op} = {got:.4f}, row says {v}")
        except Exception as e:  # noqa: BLE001
            err(f"{rid}: cannot recompute ({e!r})")

# ------------------------------------------------------------------------------------------------ README numbers
NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
UNIT = re.compile(r"^\s?(tok/s|GB/s|Gb/s|us\b|\u00b5s|TFLOPS|W\b|%|x\b)")


def tokens(line):
    """Yield (value, tolerance, text, has_performance_unit) for each standalone number in a line."""
    for m in NUM.finditer(line):
        s, e = m.start(), m.end()
        prev = line[s - 1] if s else " "
        if prev in "-\u2212+" and s >= 2 and line[s - 2].isdigit():
            prev = " "                                   # range such as 6.9-7.6x
        elif prev in "-\u2212+":
            prev = line[s - 2] if s >= 2 else " "       # sign
        if prev.isalnum() or prev in "_.@#/\\:":
            continue
        nxt, nxt2 = line[e:e + 1], line[e + 1:e + 2]
        k = False
        if nxt == "K" and not nxt2.isalpha():
            k = True
        elif nxt == "x" and not nxt2.isalnum():
            pass
        elif nxt.isalnum() or nxt in "_:" or (nxt in "./" and nxt2.isdigit()):
            continue
        txt = m.group(0)
        whole, _, frac = txt.replace(",", "").partition(".")
        val = float(txt.replace(",", ""))
        prec = len(frac)
        tol = 0.5 * 10 ** (-prec)
        if k:
            val, tol = val * 1000, tol * 1000
        after = line[e + (1 if k else 0):]
        rng = re.match(r"^[-\u2013\u2212][\d.,]+K?x?", after)       # first half of a range inherits the unit
        unit = bool(UNIT.match(after)) or nxt == "x" or bool(rng and (UNIT.match(after[rng.end():]) or rng.group(0).endswith("x")))
        yield val, tol, txt + ("K" if k else ""), unit


def clean(md):
    md = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), md, flags=re.S)
    md = re.sub(r"<!--(?! /?results -->).*?-->", "", md, flags=re.S)
    return md


NAMES = re.compile(r"RTX PRO \d+|ConnectX-\d+|CX-\d+")   # product names that contain standalone numbers


def strip_line(line):
    line = re.sub(r"`[^`]*`", "", line)
    line = NAMES.sub("", line)
    return re.sub(r"\]\([^)]*\)", "]", line)


links_seen = set()
readmes = [p for p in ["README.md"] + glob.glob("recipes/*/README.md", root_dir=ROOT) + glob.glob("atlas/*/README.md", root_dir=ROOT)]
n_checked = 0
for rel in readmes:
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        err(f"{rel}: missing")
        continue
    md = clean(open(path).read())
    inblock = False
    for ln, raw in enumerate(md.splitlines(), 1):
        if "<!-- results -->" in raw:
            inblock = True
            continue
        if "<!-- /results -->" in raw:
            inblock = False
            continue
        targets = re.findall(r"\]\(([^)\s]+)\)", re.sub(r"`[^`]*`", "", raw))
        for t in targets:
            if re.match(r"(https?:|mailto:|#)", t):
                continue
            tp = os.path.normpath(os.path.join(os.path.dirname(path), t.split("#")[0]))
            links_seen.add(os.path.relpath(tp, ROOT))
            if not os.path.exists(tp):
                err(f"{rel}:{ln}: broken link {t}")
        line = strip_line(raw)
        linked = {os.path.relpath(os.path.normpath(os.path.join(os.path.dirname(path), t)), ROOT) for t in targets
                  if not re.match(r"(https?:|#)", t)}
        urls = {t for t in targets if t.startswith("http")}
        for val, tol, txt, has_unit in tokens(line):
            if not inblock and not has_unit:
                continue
            n_checked += 1
            cands = [r for r in rows if isinstance(r.get("value"), (int, float))
                     and abs(abs(r["value"]) - val) <= tol + 1e-9]
            if not cands:
                err(f"{rel}:{ln}: number {txt!r} has no results.jsonl row")
                continue
            if inblock:
                ok = any((r["kind"] == "public-reference" and r.get("url") in urls)
                         or (r["kind"] != "public-reference" and set(logs_of(r)) & linked)
                         or (r.get("op") == "ratio_rows" and all(set(logs_of(byid[x])) & linked or byid[x].get("url") in urls
                                                                  for x in r["rows_in"]))
                         for r in cands)
                if not ok:
                    err(f"{rel}:{ln}: {txt!r} matches {[r['id'] for r in cands][:4]} but none has its receipt linked on this line")
        if not inblock and "<!-- results" in raw:
            pass
    if inblock:
        err(f"{rel}: unterminated results block")

for lg in sorted(glob.glob("logs/**/*.log", root_dir=ROOT, recursive=True)):
    if not any(lg in logs_of(r) for r in rows):
        err(f"{lg}: log excerpt not referenced by any results.jsonl row")

print(f"results.jsonl: {len(rows)} rows; README numbers checked: {n_checked}; readmes: {len(readmes)}")
if ERR:
    print("FAIL")
    for e in ERR:
        print("  " + e)
    sys.exit(1)
print("PASS")
