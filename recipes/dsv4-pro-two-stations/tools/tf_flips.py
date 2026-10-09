# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""tf_flips.py -- E2/E3 numerics gate (teacher-forced top-1 flips, J-M style).
  gen <base> <model> <ref.json>          : on the REFERENCE server (E1): 32 chat prompts -> /tokenize (chat template, thinking
                                           off) -> greedy 160-token continuation -> teacher-forced prompt_logprobs top-1 over
                                           prompt+continuation. Saves token ids + per-position top-1 ids.
  tf  <base> <model> <ref.json> <out.json>: on the CANDIDATE server: same token sequences, teacher-forced top-1; reports the
                                           share of positions whose top-1 differs (all positions, and continuation-only),
                                           plus free-running greedy prefix agreement on the same 32 prompts.
<base> = http://127.0.0.1:8010
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "What is the capital of France? Answer in one word.", "What is 17 * 23? Answer with the number only.",
    "Spell the word 'airplane' backwards.", "Name the largest planet in the solar system.",
    "Write a Python function that returns the square of a number.", "Explain in two sentences why the sky is blue.",
    "List three primary colors.", "A pilot flies at 120 knots for 2.5 hours. How many nautical miles?",
    "What does the acronym VFR stand for in aviation?", "Who wrote 'Pride and Prejudice'?", "Give a haiku about mountains.",
    "Summarize the plot of Romeo and Juliet in one sentence.", "Translate 'good morning, how are you?' into German.",
    "Write a bash one-liner that counts lines in all .py files under the current directory.",
    "What is the derivative of x^3 * sin(x)?", "Explain the difference between TCP and UDP in three bullet points.",
    "Explain hypoxia types (hypoxic, hypemic, stagnant, histotoxic) for a private pilot student.",
    "What are the symptoms of carbon monoxide poisoning in the cockpit and what should a pilot do?",
    "Describe the IMSAFE checklist.", "What is spatial disorientation and how does the 'leans' illusion occur?",
    "Explain the four left-turning tendencies of a propeller airplane.", "What does a METAR 'BKN008' mean for a VFR pilot?",
    "List the required documents on board an aircraft (ARROW).", "Explain density altitude and why it matters on a hot day.",
    "Write a JSON object describing a flight plan with fields origin, destination, altitude_ft, and alternates (a list).",
    "Refactor this into a list comprehension: out=[]\nfor x in xs:\n    if x%2==0:\n        out.append(x*x)",
    "Give a SQL query that returns the 5 most recent rows from table events ordered by created_at.",
    "What is the regulatory alcohol limit for pilots under 14 CFR 91.17?",
    "Explain why an aircraft stalls at a critical angle of attack rather than a particular airspeed.",
    "Write a short professional email declining a meeting invitation.",
    "In one paragraph, compare MoE and dense transformer models.",
    "What is the Pythagorean theorem? Give a worked example.",
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def top1_seq(base, model, ids):
    d = post(base + '/v1/completions', {'model': model, 'prompt': ids, 'max_tokens': 1, 'temperature': 0.0,
                                        'prompt_logprobs': 1})
    pl = d['choices'][0]['prompt_logprobs']
    out = []
    for pos in pl:
        if pos is None:
            out.append(None)
            continue
        best = min(pos.items(), key=lambda kv: kv[1].get('rank', 99))
        out.append(int(best[0]))
    return out


def gen_one(base, model, q):
    tok = post(base + '/tokenize', {'model': model, 'messages': [{'role': 'user', 'content': q}],
                                    'add_generation_prompt': True, 'chat_template_kwargs': {'thinking': False}})['tokens']
    d = post(base + '/v1/completions', {'model': model, 'prompt': tok, 'max_tokens': 160, 'temperature': 0.0,
                                        'return_token_ids': True})
    cont = d['choices'][0].get('token_ids') or []
    return tok, cont


def main():
    mode, base, model = sys.argv[1:4]
    if mode == 'gen':
        def one(q):
            tok, cont = gen_one(base, model, q)
            return {'q': q, 'prompt_ids': tok, 'cont_ids': cont, 'top1': top1_seq(base, model, tok + cont)}
        with ThreadPoolExecutor(int(__import__("os").environ.get("CONC", "1"))) as ex:
            ref = list(ex.map(one, PROMPTS))
        json.dump(ref, open(sys.argv[4], 'w'))
        print('ref saved', len(ref), 'seqs', sum(len(r['top1']) for r in ref), 'positions')
        return
    ref = json.load(open(sys.argv[4]))

    def one(r):
        t = top1_seq(base, model, r['prompt_ids'] + r['cont_ids'])
        _, cont2 = gen_one(base, model, r['q'])
        np_ = len(r['prompt_ids'])
        pairs = [(a, b, i >= np_) for i, (a, b) in enumerate(zip(r['top1'], t)) if a is not None and b is not None]
        pref = 0
        for a, b in zip(r['cont_ids'], cont2):
            if a != b:
                break
            pref += 1
        return {'n': len(pairs), 'flips': sum(a != b for a, b, _ in pairs), 'n_cont': sum(c for *_, c in pairs),
                'flips_cont': sum(a != b for a, b, c in pairs if c), 'free_prefix': pref, 'free_len': len(r['cont_ids']),
                'free_identical': cont2 == r['cont_ids']}
    with ThreadPoolExecutor(int(__import__("os").environ.get("CONC", "1"))) as ex:
        res = list(ex.map(one, ref))
    n = sum(r['n'] for r in res); f = sum(r['flips'] for r in res)
    nc = sum(r['n_cont'] for r in res); fc = sum(r['flips_cont'] for r in res)
    summ = {'positions': n, 'flips': f, 'flip_pct': round(100 * f / max(n, 1), 3), 'cont_positions': nc, 'cont_flips': fc,
            'cont_flip_pct': round(100 * fc / max(nc, 1), 3), 'free_identical': sum(r['free_identical'] for r in res),
            'n_prompts': len(res)}
    json.dump({'summary': summ, 'per_prompt': res}, open(sys.argv[5], 'w'))
    print(json.dumps(summ))


main()
