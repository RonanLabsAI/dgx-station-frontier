# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""GSM8K-200 gate for DS-V4.1-Flash (2026-10-07). First 200 rows of openai/grade-school-math test.jsonl
(sha256 3730d312...). Sampling: thinking on, reasoning_effort 50, temperature 0.3, max_tokens 24000.
Scores the last number in the answer (after '####' if present) against the gold '#### N'. Runaway = finish_reason length.

usage: gsm8k200.py <url> <model> <out.jsonl> [conc]
"""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 16
HERE = __file__.rsplit('/', 1)[0]
rows = [json.loads(l) for l in open(f'{HERE}/gsm8k_test.jsonl')][:int(__import__("os").environ.get("GSM_N","200"))]
NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


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


def ask(i):
    r = rows[i]
    gold = norm(r['answer'].split('####')[-1].strip())
    body = {'model': MODEL, 'temperature': 0.3, 'max_tokens': 24000,
            'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 50},
            'messages': [{'role': 'user', 'content': r['question'] +
                          "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."}]}
    t = time.time()
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        ch = d['choices'][0]
        content = ch['message'].get('content') or ''
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
                      wall_s=round(time.time() - t0), median_ctoks=sorted(r.get('ctoks', 0) for r in res)[len(res) // 2])))
