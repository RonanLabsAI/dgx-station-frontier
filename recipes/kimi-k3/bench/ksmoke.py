#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# Kimi K3 quality smoke against an OpenAI-compatible server (llama-server or vLLM) on 127.0.0.1:PORT, model kimi-k3:
# three greedy canaries, GSM8K first N test rows (temperature 0, CONC parallel, "Answer: <n>" extraction), five tool calls.
# usage: ksmoke.py PORT N CONC [gsm8k_test.jsonl]   env MAXTOK (GSM8K max_tokens incl. reasoning; default 768,
# 2048 for the one-Station GSM8K-200 row). gsm8k_test.jsonl: openai/grade-school-math test set (MIT), not vendored:
#   curl -L -o gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
import json, os, re, sys, threading, urllib.request
port, N, CONC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
def chat(msgs, max_tokens=768, temp=0.0, **kw):
    body = {"model": "kimi-k3", "messages": msgs, "max_tokens": max_tokens, "temperature": temp, **kw}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600).read())
    return r["choices"][0]["message"]
# canaries
can = [("What is 4738 * 2917? Reply with just the number.", str(4738 * 2917)),
       ("What is the capital of Australia? Reply with one word.", "canberra"),
       ("Write a Python function is_prime(n) in under 10 lines. Code only.", "def is_prime")]
for q, want in can:
    m = chat([{"role": "user", "content": q}], 256)
    txt = (m.get("content") or "")
    print(f"CANARY ok={want.lower() in txt.lower().replace(',', '')} q={q[:40]!r} got={txt[:160]!r} reasoning_len={len(m.get('reasoning_content') or '')}", flush=True)
# GSM8K
rows = [json.loads(l) for l in open(sys.argv[4] if len(sys.argv) > 4 else "gsm8k_test.jsonl")][:N]
res = [None] * N; lock = threading.Lock(); nxt = [0]
def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", s) or re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None
def work():
    while True:
        with lock:
            if nxt[0] >= N: return
            i = nxt[0]; nxt[0] += 1
        r = rows[i]; gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            m = chat([{"role": "user", "content": r["question"] + "\nSolve it step by step, briefly. End with a final line 'Answer: <number>'."}], int(os.environ.get("MAXTOK", "768")))
            got = num(m.get("content") or "")
        except Exception as e:
            got = f"ERR {e}"
        ok = got is not None and not str(got).startswith("ERR") and abs(float(got) - float(gold)) < 1e-6
        res[i] = (ok, gold, got); print(f"GSM {i} ok={ok} gold={gold} got={got}", flush=True)
th = [threading.Thread(target=work) for _ in range(CONC)]; [t.start() for t in th]; [t.join() for t in th]
print(f"RESULT gsm8k_{N} {sum(r[0] for r in res)}/{N} = {100*sum(r[0] for r in res)/N:.1f}%", flush=True)
# tool calls
tools = [{"type": "function", "function": {"name": "get_metar", "description": "Get the METAR for an airport",
          "parameters": {"type": "object", "properties": {"icao": {"type": "string"}}, "required": ["icao"]}}}]
qs = [("What's the weather at KSFO right now?", "KSFO"), ("Fetch the METAR for KOAK.", "KOAK"), ("Is it VFR at KLAX?", "KLAX"),
      ("Get the current observation for KJFK please.", "KJFK"), ("Check the METAR at KSEA.", "KSEA")]
ok = 0
for q, icao in qs:
    try:
        m = chat([{"role": "user", "content": q}], 512, tools=tools)
        tc = m.get("tool_calls") or []
        good = bool(tc) and tc[0]["function"]["name"] == "get_metar" and json.loads(tc[0]["function"]["arguments"]).get("icao", "").upper() == icao
    except Exception as e:
        good = False; tc = str(e)
    ok += good; print(f"TOOL ok={good} {q!r} -> {str(tc)[:160]}", flush=True)
print(f"RESULT toolcall {ok}/5", flush=True)
