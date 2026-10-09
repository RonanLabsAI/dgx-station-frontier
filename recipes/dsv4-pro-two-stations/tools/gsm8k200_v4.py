# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""GSM8K-200 for DeepSeek-V4 (2026-10-07). Adapted from recipes/dsv41-flash-sidecar/bench/gsm8k.py (first 200 rows of openai/grade-school-math
test.jsonl, same scorer). V4 tokenizer: thinking on (default), reasoning_effort is a STRING ("high" default).
Sampling: temperature 0.0 (greedy), max_tokens 16000. Runaway = finish_reason length.
If the server runs --enable-return-routed-experts, each choice carries `routed_experts` (base64 np.save int32[T, L, k]);
with ROUTES=<path.json> they are aggregated into per-(layer, expert) counts, split prompt/decode (aggregate only, no
per-request data is stored).

usage: gsm8k200_v4.py <url> <model> <out.jsonl> [conc] [n]     env: EFFORT=high THINK=1 ROUTES=/path/counts.json
"""
import base64, io, json, os, re, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 16
NROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 200
EFFORT = os.environ.get('EFFORT', 'high'); THINK = os.environ.get('THINK', '1') == '1'
ROUTES = os.environ.get('ROUTES')
HERE = __file__.rsplit('/', 1)[0]
rows = [json.loads(l) for l in open(f'{HERE}/gsm8k_test.jsonl')][:NROWS]
NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
lock = threading.Lock()
counts = {}  # phase -> np.int64[L, E]


def norm(s):
    s = s.replace(',', '').rstrip('.')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def pred(text):
    text = text or ''
    m = re.search(r'####\s*(.*)', text)
    seg = m.group(1) if m else (re.findall(r'\\boxed\{([^}]*)\}', text) or [text])[-1]
    nums = NUM.findall(seg.replace('$', ''))
    return norm(nums[0] if m and nums else (nums[-1] if nums else ''))


def add_routes(b64, n_prompt):
    import numpy as np
    a = np.load(io.BytesIO(base64.b64decode(b64)))  # int32[T, L, k]
    E = int(os.environ.get('NEXPERTS', '384'))
    with lock:
        for phase, sl in (('prefill', a[:n_prompt]), ('decode', a[n_prompt:])):
            if sl.size == 0:
                continue
            L = sl.shape[1]
            c = counts.setdefault(phase, np.zeros((L, E), dtype=np.int64))
            for l in range(L):
                ids = sl[:, l, :].ravel()
                ids = ids[ids >= 0]
                c[l] += np.bincount(ids, minlength=E)[:E]


def ask(i):
    r = rows[i]
    gold = norm(r['answer'].split('####')[-1].strip())
    kw = {'thinking': THINK}
    if THINK:
        kw['reasoning_effort'] = EFFORT
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 16000, 'chat_template_kwargs': kw,
            'messages': [{'role': 'user', 'content': r['question'] +
                          "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."}]}
    t = time.time()
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        ch = d['choices'][0]
        content = ch['message'].get('content') or ''
        if ROUTES and ch.get('routed_experts'):
            add_routes(ch['routed_experts'], d['usage']['prompt_tokens'])
        p = pred(content)
        return dict(i=i, gold=gold, pred=p, ok=p == gold, finish=ch.get('finish_reason'),
                    ctoks=d['usage']['completion_tokens'], sec=round(time.time() - t, 1), tail=content[-200:])
    except Exception as e:  # noqa: BLE001
        return dict(i=i, gold=gold, pred=None, ok=False, finish='error', error=str(e)[:300], sec=round(time.time() - t, 1))


t0 = time.time()
with ThreadPoolExecutor(CONC) as ex, open(OUT, 'w') as f:
    res = []
    for r in ex.map(ask, range(len(rows))):
        res.append(r)
        f.write(json.dumps(r) + '\n')
        f.flush()
ok = sum(r['ok'] for r in res)
run = sum(r['finish'] == 'length' for r in res)
err = sum(r['finish'] == 'error' for r in res)
print(json.dumps(dict(n=len(res), correct=ok, acc=round(100 * ok / len(res), 2), runaways=run, errors=err,
                      wall_s=round(time.time() - t0), median_ctoks=sorted(r.get('ctoks', 0) for r in res)[len(res) // 2],
                      effort=EFFORT if THINK else 'nothink')))
if ROUTES and counts:
    out = {'format': 'routing-counts/v1', 'model': MODEL, 'source': 'vllm --enable-return-routed-experts (GSM8K-200 traffic)',
           'time': time.time(), 'phases': {ph: {str(l): c[l].tolist() for l in range(c.shape[0])} for ph, c in counts.items()},
           'routes_total': {ph: int(c.sum()) for ph, c in counts.items()}}
    json.dump(out, open(ROUTES, 'w'))
    print('routes written', ROUTES, out['routes_total'])
