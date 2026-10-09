#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""code20.py -- small executed coding check (2026-10-08): 20 stdin/stdout problems from LiveCodeBench code_generation_lite
test6.jsonl (rev 0fe84c39; AtCoder ABC/ARC 2025-01..03: 6 easy / 8 medium / 6 hard; public + private tests, <=30 per problem),
pre-extracted to code20.jsonl by lcb_select.py (private tests decoded with a no-globals Unpickler).
Contamination caveat: the problems predate Nemotron 3 Ultra's release, so this is a sanity check, not a leaderboard.

Generation: reasoning ON (enable_thinking True), temperature 1.0 / top_p 0.95 (both model cards), seed 1 per problem,
max_tokens MAXTOK (default 24000), one sample (pass@1). The model may answer in C++ or Python; the LAST fenced code block is
judged. Judging runs in a throwaway container with no network, 1 CPU, 2 GB, read-only code mount: g++ -O2 -std=gnu++20 or
python3, per-test wall limit TL seconds (default 6 s; AtCoder's is 2 s on native hardware), whitespace-normalised compare.
A problem passes only if every test passes.

usage: code20.py <base_url> <model> <outdir> [conc=20]      env: MAXTOK=24000 TL=6 JUDGE_IMAGE=<image with g++ + python3>
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, MODEL, OUTD = (sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('', '', os.environ.get('OUTD', '/tmp/code20'))
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 20
MAXTOK = int(os.environ.get('MAXTOK', '24000')); TL = float(os.environ.get('TL', '6'))
IMG = os.environ.get('JUDGE_IMAGE', 'vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b')
HERE = __file__.rsplit('/', 1)[0]
probs = [json.loads(l) for l in open(f'{HERE}/code20.jsonl')]
os.makedirs(OUTD, exist_ok=True)
PROMPT = ("Solve the following competitive programming problem. Read from standard input and write to standard output.\n"
          "Give your final solution as ONE complete program in a single fenced code block (```cpp or ```python).\n\n{st}")
FENCE = re.compile(r'```(cpp|c\+\+|python|py)?\s*\n(.*?)```', re.S)


def gen(p):
    body = {'model': MODEL, 'temperature': 1.0, 'top_p': 0.95, 'seed': 1, 'max_tokens': MAXTOK,
            'chat_template_kwargs': {'enable_thinking': True},
            'messages': [{'role': 'user', 'content': PROMPT.format(st=p['statement'])}]}
    t = time.time()
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    ch = d['choices'][0]
    return ch['message'].get('content') or '', ch.get('finish_reason'), d.get('usage', {}).get('completion_tokens'), time.time() - t


def judge(p, lang, code):
    """Compile + run all tests inside one no-network container. Returns (passed, total, note)."""
    with tempfile.TemporaryDirectory(dir=OUTD) as w:
        src = 'main.cpp' if lang == 'cpp' else 'main.py'
        open(f'{w}/{src}', 'w').write(code)
        for k, t in enumerate(p['tests']):
            open(f'{w}/in{k}.txt', 'w').write(t['input'])
        run = ('g++ -O2 -std=gnu++20 -o /tmp/a /w/main.cpp 2>/tmp/ce || { echo CE; head -c 400 /tmp/ce; exit 0; }; X=/tmp/a'
               if lang == 'cpp' else 'X="python3 /w/main.py"')
        run += (f'; for f in /w/in*.txt; do k=${{f#/w/in}}; k=${{k%.txt}}; timeout {TL} $X < $f > /tmp/out$k 2>/dev/null; '
                'echo "RC $k $?"; cp /tmp/out$k /out/out$k 2>/dev/null; done')
        os.makedirs(f'{w}/out', exist_ok=True)
        r = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--cpus', '1', '--memory', '2g', '--pids-limit', '256',
                            '-v', f'{w}:/w:ro', '-v', f'{w}/out:/out', '--entrypoint', 'bash', IMG, '-c', run],
                           capture_output=True, text=True, timeout=60 + TL * len(p['tests']) * 1.5)
        if r.stdout.startswith('CE'):
            return 0, len(p['tests']), 'compile error: ' + r.stdout[2:200]
        rcs = dict(l.split()[1:3] for l in r.stdout.splitlines() if l.startswith('RC '))
        ok = 0; note = ''
        for k, t in enumerate(p['tests']):
            got = open(f'{w}/out/out{k}').read() if os.path.exists(f'{w}/out/out{k}') else ''
            if rcs.get(str(k)) == '0' and got.split() == t['output'].split():
                ok += 1
            elif not note:
                note = f"test {k}: rc {rcs.get(str(k))}" + (' (timeout)' if rcs.get(str(k)) == '124' else ' (wrong answer)')
        return ok, len(p['tests']), note


def one(p):
    try:
        text, fin, ctoks, sec = gen(p)
    except Exception as e:  # noqa: BLE001
        return {'id': p['id'], 'difficulty': p['difficulty'], 'pass': False, 'note': f'gen error {str(e)[:200]}'}
    open(f"{OUTD}/{p['id']}.md", 'w').write(text)
    blocks = FENCE.findall(text)
    res = {'id': p['id'], 'difficulty': p['difficulty'], 'finish': fin, 'ctoks': ctoks, 'gen_sec': round(sec, 1)}
    if not blocks:
        return {**res, 'pass': False, 'note': 'no code block' + (' (length)' if fin == 'length' else '')}
    tag, code = blocks[-1]
    lang = 'python' if (tag or '').startswith('py') or (not tag and 'def ' in code and '#include' not in code) else 'cpp'
    ok, tot, note = judge(p, lang, code)
    return {**res, 'lang': lang, 'tests_ok': ok, 'tests': tot, 'pass': ok == tot, 'note': note}


if __name__ == '__main__':
    t0 = time.time()
    with ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(one, probs))
    with open(f'{OUTD}/code20.jsonl', 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    by = {}
    for r in res:
        by.setdefault(r['difficulty'], []).append(r['pass'])
    tot = sum(r['pass'] for r in res)
    print(f"code20 pass@1: {tot}/{len(res)}  " + '  '.join(f"{d} {sum(v)}/{len(v)}" for d, v in by.items()) +
          f"  length-capped {sum(r.get('finish') == 'length' for r in res)}  mean ctoks "
          f"{sum(r.get('ctoks') or 0 for r in res)/len(res):.0f}  wall {time.time()-t0:.0f}s")
