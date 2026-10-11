#!/usr/bin/env python3
"""Read-by-eye smoke set for B1: 3 coding prompts + 2 tool-call prompts (public, written here). Greedy, thinking off.
usage: smoke_tf.py URL OUT.json      Writes every full response; prints a short digest. Stdlib only.
"""
import json
import sys
import urllib.request

CODE = [
    "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals and returns them "
    "sorted. Include three asserts that test edge cases.",
    "In C, write a function `int is_balanced(const char *s)` that returns 1 if the brackets ()[]{} in s are balanced "
    "and 0 otherwise. Use a fixed-size stack and explain its limit in one sentence.",
    "Here is a buggy JavaScript function. Find the bug and give the fixed code only.\n\n"
    "function sum(arr) {\n  let total;\n  for (let i = 0; i <= arr.length; i++) total += arr[i];\n  return total;\n}",
]
TOOLS = [{
    "type": "function",
    "function": {"name": "get_weather", "description": "Current weather for a city.",
                 "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                 "unit": {"type": "string", "enum": ["c", "f"]}},
                                "required": ["city"]}},
}, {
    "type": "function",
    "function": {"name": "convert_currency", "description": "Convert an amount between ISO currencies.",
                 "parameters": {"type": "object", "properties": {"amount": {"type": "number"},
                                                                 "from": {"type": "string"},
                                                                 "to": {"type": "string"}},
                                "required": ["amount", "from", "to"]}},
}]
TOOL_PROMPTS = [
    "What's the weather in Reykjavik right now, in Celsius?",
    "How much is 250 US dollars in euros? Then tell me the weather in Lisbon.",
]


def post(url, body):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def main():
    url, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "glm"
    res = []
    base = {"model": model, "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    for p in CODE:
        r = post(url, {**base, "messages": [{"role": "user", "content": p}]})
        res.append({"kind": "code", "prompt": p, "response": r})
    for p in TOOL_PROMPTS:
        r = post(url, {**base, "messages": [{"role": "user", "content": p}], "tools": TOOLS, "tool_choice": "auto"})
        res.append({"kind": "tool", "prompt": p, "response": r})
    json.dump(res, open(out, "w"), indent=1)
    for x in res:
        m = x["response"]["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        print(f"--- {x['kind']}: {x['prompt'][:60]!r}")
        print(f"    finish={x['response']['choices'][0].get('finish_reason')} tool_calls={calls}")
        print("    " + (m.get("content") or "")[:300].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
