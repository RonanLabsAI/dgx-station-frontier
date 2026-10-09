# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""routes_calib.py <url> <model> <out.json> <prompts.json>... -- E2 calibration. Sends each prompt (list of
{"id","messages"}) with the server's --enable-return-routed-experts on, and aggregates per-(layer, expert) routing counts
per prompt FILE (one split per file), separately for prefill and decode. Aggregate only: no per-request routing is kept.
env: THINK=1 EFFORT=high MAXTOK=3000 CONC=16 NEXPERTS=384
"""
import base64, io, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

URL, MODEL, OUT = sys.argv[1:4]
FILES = sys.argv[4:]
THINK = os.environ.get('THINK', '1') == '1'; EFFORT = os.environ.get('EFFORT', 'high')
MAXTOK = int(os.environ.get('MAXTOK', '3000')); CONC = int(os.environ.get('CONC', '16'))
E = int(os.environ.get('NEXPERTS', '384'))
lock = threading.Lock()
acc = {}  # split -> phase -> int64[L,E]
meta = {}


def add(split, b64, n_prompt):
    a = np.load(io.BytesIO(base64.b64decode(b64))).astype(np.int64)  # [T, L, k]
    with lock:
        for ph, sl in (('prefill', a[:n_prompt]), ('decode', a[n_prompt:])):
            if sl.size == 0:
                continue
            L = sl.shape[1]
            c = acc.setdefault(split, {}).setdefault(ph, np.zeros((L, E), dtype=np.int64))
            for l in range(L):
                c[l] += np.bincount(sl[:, l, :].ravel(), minlength=E)[:E]


def ask(job):
    split, p = job
    kw = {'thinking': THINK}
    if THINK:
        kw['reasoning_effort'] = EFFORT
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': MAXTOK, 'chat_template_kwargs': kw, 'messages': p['messages']}
    t = time.time()
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        ch = d['choices'][0]
        if ch.get('routed_experts'):
            add(split, ch['routed_experts'], d['usage']['prompt_tokens'])
        return dict(split=split, id=p['id'], ok=True, ptoks=d['usage']['prompt_tokens'], ctoks=d['usage']['completion_tokens'],
                    finish=ch.get('finish_reason'), sec=round(time.time() - t, 1), routes=bool(ch.get('routed_experts')))
    except Exception as e:  # noqa: BLE001
        return dict(split=split, id=p['id'], ok=False, error=str(e)[:300])


jobs = []
for f in FILES:
    split = os.path.splitext(os.path.basename(f))[0]
    for p in json.load(open(f)):
        jobs.append((split, p))
t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(ask, jobs))
out = {'format': 'routing-counts/v1', 'model': MODEL, 'source': 'vllm --enable-return-routed-experts', 'time': time.time(),
       'think': THINK, 'effort': EFFORT, 'max_tokens': MAXTOK, 'wall_s': round(time.time() - t0),
       'splits': {s: {ph: {str(l): c[l].tolist() for l in range(c.shape[0])} for ph, c in phs.items()} for s, phs in acc.items()},
       'requests': [{k: v for k, v in r.items() if k != 'id'} for r in res]}
json.dump(out, open(OUT, 'w'))
for s, phs in acc.items():
    print(s, {ph: int(c.sum()) // (c.shape[0] * 6) for ph, c in phs.items()}, 'tokens')
print('errors', sum(not r['ok'] for r in res), 'no-routes', sum(r.get('ok') and not r.get('routes') for r in res), 'wall', out['wall_s'])
