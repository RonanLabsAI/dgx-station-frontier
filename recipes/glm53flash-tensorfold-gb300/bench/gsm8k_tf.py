#!/usr/bin/env python3
"""GSM8K-N for a TensorFold server (B1). First N rows of openai/grade-school-math test.jsonl, same scorer as our
gsm8k200_v4.py (#### answer, else last \\boxed{}, else last number). Greedy, non-streamed. Stdlib only.

usage: gsm8k_tf.py URL TEST.jsonl OUT.jsonl [--n 50] [--conc 4] [--think 1] [--max-tokens 16000]
Prompt: question + "\\nThink step by step, then give the final answer on its own line as '#### <number>'."
Pass bar (PLAN.md B1): >= 95%. Runaway = finish_reason length (counted wrong, reported).
"""
import argparse
import json
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
SUFFIX = "\nThink step by step, then give the final answer on its own line as '#### <number>'."


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"), ap.add_argument("test"), ap.add_argument("out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--model", default="glm")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.test)][:a.n]
    out, lock = [], threading.Lock()

    def ask(i):
        r = rows[i]
        body = {"model": a.model, "messages": [{"role": "user", "content": r["question"] + SUFFIX}],
                "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 20261010 + i,
                "enable_thinking": bool(a.think), "chat_template_kwargs": {"enable_thinking": bool(a.think)}}
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=3600))
            ch = resp["choices"][0]
            text = ch["message"].get("content") or ""
            fin = ch.get("finish_reason")
            usage = resp.get("usage", {})
        except Exception as e:  # noqa: BLE001
            text, fin, usage = f"ERROR {e}", "error", {}
        gold = norm(r["answer"].split("####")[-1].strip())
        p = pred(text)
        rec = {"i": i, "gold": gold, "pred": p, "ok": p == gold, "finish": fin, "usage": usage, "tail": text[-300:]}
        with lock:
            out.append(rec)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(ask, range(len(rows))))
    out.sort(key=lambda r: r["i"])
    ok = sum(r["ok"] for r in out)
    summ = {"n": len(out), "correct": ok, "acc": ok / len(out), "runaways": sum(r["finish"] == "length" for r in out),
            "errors": sum(r["finish"] == "error" for r in out), "think": a.think, "pass_95": ok / len(out) >= 0.95}
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summ}) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
