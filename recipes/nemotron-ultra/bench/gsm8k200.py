#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""GSM8K-200 for Nemotron 3 Ultra-class models: the first 200 rows of openai/grade-school-math test.jsonl
(gsm8k_test.jsonl next to this script; not vendored, see the README), same rows and scorer as our other GSM8K-200 rows.
Reasoning on via chat_template_kwargs.enable_thinking, greedy (temperature 0), max_tokens 16000.
Runaway = finish_reason "length".

usage: gsm8k200.py <base_url> <model> <out.jsonl> [conc=32] [n=200]
"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, MODEL, OUT = sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
NROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 200
HERE = __file__.rsplit('/', 1)[0]
rows = [json.loads(line) for line in open(f'{HERE}/gsm8k_test.jsonl')][:NROWS]
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
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 16000,
            'chat_template_kwargs': {'enable_thinking': True},
            'messages': [{'role': 'user', 'content': r['question'] +
                          "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."}]}
    t = time.time()
    try:
        req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        ch = d['choices'][0]
        msg = ch['message']
        p = pred(msg.get('content'))
        return {'i': i, 'gold': gold, 'pred': p, 'ok': p == gold, 'finish': ch.get('finish_reason'),
                'ctoks': d.get('usage', {}).get('completion_tokens'), 'sec': round(time.time() - t, 1),
                'content_tail': (msg.get('content') or '')[-300:]}
    except Exception as e:  # noqa: BLE001
        return {'i': i, 'gold': gold, 'pred': None, 'ok': False, 'finish': 'error', 'err': str(e)[:300],
                'sec': round(time.time() - t, 1)}


t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(ask, range(len(rows))))
with open(OUT, 'w') as f:
    for r in res:
        f.write(json.dumps(r) + '\n')
ok = sum(r['ok'] for r in res)
run = sum(r['finish'] == 'length' for r in res)
err = sum(r['finish'] == 'error' for r in res)
toks = [r['ctoks'] for r in res if r.get('ctoks')]
print(f"GSM8K-{len(rows)}: {ok}/{len(rows)} = {100*ok/len(rows):.1f}%  runaway {run}  errors {err}  "
      f"mean completion tokens {sum(toks)/max(1, len(toks)):.0f}  wall {time.time()-t0:.0f}s  conc {CONC}")
