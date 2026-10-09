#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""Select 20 stdin problems from LiveCodeBench code_generation_lite test6.jsonl (rev 0fe84c39) for the code20 check.
Private tests are base64(zlib(pickle(json-str))). They are decoded with an Unpickler that refuses every global, so a
pickle can only yield plain str/bytes/list/dict and cannot execute code. Output: code20.jsonl (one problem per line:
id, title, difficulty, date, statement, tests=[{input, output}] public+private, capped).
usage: python3 -I lcb_select.py <test6.jsonl> <out.jsonl>
"""
import base64, io, json, pickle, sys, zlib


class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'global {module}.{name} refused')


def private(s):
    try:
        return json.loads(s)
    except Exception:
        raw = NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load()
        return json.loads(raw)


src, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
cand = [r for r in rows if r['platform'] in ('atcoder', 'codeforces') and not r['starter_code']
        and all(t['testtype'] == 'stdin' for t in json.loads(r['public_test_cases']))]
cand.sort(key=lambda r: (r['contest_date'], r['question_id']))
want = {'easy': 6, 'medium': 8, 'hard': 6}
pick = []
for d, n in want.items():
    pool = [r for r in cand if r['difficulty'] == d]
    step = max(1, len(pool) // n)
    pick += pool[::step][:n]
with open(out, 'w') as f:
    for r in pick:
        tests = json.loads(r['public_test_cases']) + private(r['private_test_cases'])
        tests = [{'input': t['input'], 'output': t['output']} for t in tests if t.get('testtype', 'stdin') == 'stdin'][:30]
        f.write(json.dumps({'id': r['question_id'], 'title': r['question_title'], 'difficulty': r['difficulty'],
                            'date': r['contest_date'], 'platform': r['platform'], 'statement': r['question_content'],
                            'tests': tests}) + '\n')
print(len(rows), 'rows;', len(cand), 'stdin candidates;', len(pick), 'picked')
for r in pick:
    print(r['question_id'], r['difficulty'], r['contest_date'][:10], r['question_title'][:50])
