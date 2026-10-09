# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""coherence16.py <url> <model> <out.jsonl> -- E1: 16 greedy prompts (temperature 0, thinking off, 200 tokens, all 16
in flight). Prints each answer head; a human reads them for coherence. Exact-answer items are auto-checked."""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL, MODEL, OUT = sys.argv[1:4]
P = [("What is the capital of France? Answer in one word.", "paris"),
     ("What is 17 * 23? Answer with the number only.", "391"),
     ("Spell the word 'airplane' backwards.", "enalpria"),
     ("What gas do plants absorb from the air for photosynthesis? One word.", "carbon"),
     ("Name the largest planet in the solar system. One word.", "jupiter"),
     ("Translate 'thank you' into Spanish.", "gracias"),
     ("What is the boiling point of water at sea level in degrees Celsius? Number only.", "100"),
     ("Write a Python function that returns the square of a number.", "def"),
     ("Explain in two sentences why the sky is blue.", None),
     ("List three primary colors.", None),
     ("A pilot flies at 120 knots for 2.5 hours. How many nautical miles? Number only.", "300"),
     ("What does the acronym VFR stand for in aviation?", "visual"),
     ("Who wrote 'Pride and Prejudice'?", "austen"),
     ("Give a haiku about mountains.", None),
     ("What is the chemical symbol for gold?", "au"),
     ("Summarize the plot of Romeo and Juliet in one sentence.", None)]


def ask(i):
    q, exp = P[i]
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 200, 'chat_template_kwargs': {'thinking': False},
            'messages': [{'role': 'user', 'content': q}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=900).read())
        a = d['choices'][0]['message'].get('content') or ''
    except Exception as e:  # noqa: BLE001
        a = 'ERROR ' + str(e)[:200]
    ok = None if exp is None else (exp in a.lower())
    return dict(i=i, q=q, ok=ok, a=a)


with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(ask, range(len(P))))
with open(OUT, 'w') as f:
    for r in res:
        f.write(json.dumps(r) + '\n')
        print(r['i'], r['ok'], repr(r['a'][:140]))
chk = [r for r in res if r['ok'] is not None]
print('auto-checked %d/%d correct' % (sum(r['ok'] for r in chk), len(chk)))
