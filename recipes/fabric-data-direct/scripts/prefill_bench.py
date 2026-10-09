#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Prefill bench: unique-prefix prompts of ~8K / ~64K tokens, max_tokens=1, thinking off. Reports TTFT-equivalent
(full request latency with 1 output token) and prefill tok/s = prompt_tokens / latency. Also concurrent C4 at 64K.
env BASE_URL MODEL TAG"""
import json, os, time, uuid, threading, urllib.request
BASE=os.getenv("BASE_URL","http://127.0.0.1:8004/v1"); MODEL=os.getenv("MODEL","glm-5.3"); TAG=os.getenv("TAG","x")
H={"Content-Type":"application/json"}
WORDS=("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo "
       "sierra tango uniform victor whiskey xray yankee zulu").split()
def prompt(ntok):
    import random; rnd=random.Random(uuid.uuid4().int)
    body=" ".join(rnd.choice(WORDS)+str(rnd.randint(0,99)) for _ in range(int(ntok/2.6)))
    return f"[{uuid.uuid4().hex}] Read the following log and reply with OK.\n{body}\nReply OK."
def req(ntok):
    p={"model":MODEL,"messages":[{"role":"user","content":prompt(ntok)}],"max_tokens":1,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":False}}
    t0=time.monotonic(); r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/chat/completions",data=json.dumps(p).encode(),headers=H),timeout=1800))
    return r["usage"]["prompt_tokens"], time.monotonic()-t0
def conc(ntok,c):
    res=[None]*c
    def w(i): res[i]=req(ntok)
    th=[threading.Thread(target=w,args=(i,)) for i in range(c)]; t0=time.monotonic(); [t.start() for t in th]; [t.join() for t in th]
    wall=time.monotonic()-t0; return sum(r[0] for r in res), wall
if __name__=="__main__":
    req(2000)  # warm
    for n in (8192, 65536):
        req(n)  # warm at shape (discarded)
        rs=[req(n) for _ in range(3)]
        for pt,dt in rs: print(f"[{TAG}] PREFILL target {n}: prompt_tokens {pt} latency {dt:.3f}s -> {pt/dt:,.0f} tok/s",flush=True)
        m=sorted(pt/dt for pt,dt in rs)[1]; print(f"[{TAG}] PREFILL {n} median {m:,.0f} tok/s",flush=True)
    for n,c in ((65536,4),(8192,16)):
        tok,wall=conc(n,c); print(f"[{TAG}] PREFILL-CONC {n}x{c}: {tok} tok in {wall:.2f}s -> {tok/wall:,.0f} tok/s agg",flush=True)
