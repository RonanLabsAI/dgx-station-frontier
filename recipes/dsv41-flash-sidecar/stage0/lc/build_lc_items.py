#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# RonanLabs 2026-10-10 (Stage 0 amendment C2). Runs on any host with pandas + tokenizers; the Station only reads the output.
"""Build LC-1008: a deterministic RULER-style long-context accuracy set for DS-V4.1-Flash (16K / 32K / 64K).

  build_lc_items.py <CNN test parquet> <tokenizer.json> <rt/pool-manifest.json> <longctx-cont.json> <s0-qa.json>
                    <out.jsonl.gz> [--seed 20261011] [--n 56]

To rebuild the exact LC-1008 used in the runs (decompressed sha256 5ec29a90...2812846), pass
../../bench/realtext/manifest.json as the pool manifest and lc-exclude.json (the article ids of the six long-context
continuation prompts and of the 20 QA20 prompts; only their "src" fields are read) as BOTH the longctx-cont and the
s0-qa arguments. The parquet is abisee/cnn_dailymail 3.0.0 test.parquet and tokenizer.json is DeepSeek-V4.1-Flash's;
both are sha256-checked against the manifest.

Task definitions follow RULER (Hsieh et al. 2024, "RULER: What's the Real Context Size of Your Long-Context Language
Models?"; github.com/NVIDIA/RULER, Apache-2.0, design reference @c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a). No RULER
code or data is copied: this is a re-implementation with a public, pinned haystack and chat-format questions.

Haystack: abisee/cnn_dailymail @96df5e68 3.0.0 test (Apache-2.0), sha256 checked against rt/pool-manifest.json;
articles used by the RT pool, by E2's Q3 continuation prompts or by QA20 are excluded. Whole-article order, sentence
level insertion. Vocabulary (keys, CWE words) = lowercase [a-z]{4,10} words of the same pool with corpus count >= 30,
minus the 300 most frequent, sorted.

Tasks (6) x lengths (16384, 32768, 65536 tokens of content; tokenized with the model's tokenizer.json, pinned) x n:
  mk    niah multikey: 16 needles "One of the special magic numbers for {key} is: {7 digits}." in CNN text; 8 of the
        15 distractor keys share a word with the target key. Target depth stratified (i + 0.5) / n.   rule first_num
  mv    niah multivalue: the target key has 4 values (4 needles) + 8 distractor needles.              rule num_set
  mq    niah multiquery: 12 needles, 4 keys asked in one question, answers in the asked order.        rule num_seq
  uuid  niah multikey-3: the haystack is ONLY "One of the special magic uuids for {uuid} is: {uuid}." lines;
        one key asked; depth stratified.                                                               rule first_uuid
  vt    variable tracking: 4 chains x 5 variables (VAR ABCDE = 12345; VAR FGHIJ = VAR ABCDE; ...) in CNN text,
        each chain in order; all 5 names of one chain asked.                                           rule name_set
  cwe   common words extraction: numbered word list, 10 words x 30 occurrences, the rest x 3; the 10 asked. rule word_set
Every item: thinking off, greedy, max_tokens 160, its own cache_salt (lc_eval.py). Item order = one seeded shuffle,
identical for every arm. Output: gzip (mtime 0) JSONL; the pinned hash is of the DECOMPRESSED bytes (lc_eval.py
verify).
"""
import argparse
import gzip
import hashlib
import io
import json
import random
import re
from collections import Counter

import pandas as pd
from tokenizers import Tokenizer

LENS = (16384, 32768, 65536)
TASKS = ("mk", "mv", "mq", "uuid", "vt", "cwe")
RESERVE = 160          # room for the chat template + the question
LOWBAND = 0.94         # content must land in [0.94 L, L - 96] tokens
INTRO_NUM = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz "
             "you about the numbers afterwards.\n\n")
INTRO_UUID = ("Some special magic uuids are hidden within the following text. Make sure to memorize it. I will quiz you "
              "about the uuids afterwards.\n\n")
INTRO_VT = "Memorize and track the chains of variable assignment hidden in the following text.\n\n"
INTRO_CWE = ("Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones "
             "that appear most often.\n\n")
Q_MK = ("\n\nQuestion: What is the special magic number for {k} mentioned in the provided text? Answer with the number "
        "only.")
Q_MV = ("\n\nQuestion: What are all the special magic numbers for {k} mentioned in the provided text? Answer with the "
        "numbers only, separated by commas.")
Q_MQ = ("\n\nQuestion: What are the special magic numbers for {k0}, {k1}, {k2} and {k3} mentioned in the provided text? "
        "Answer with the four numbers only, in this order, separated by commas.")
Q_UUID = ("\n\nQuestion: What is the special magic uuid for {k} mentioned in the provided text? Answer with the uuid "
          "only.")
Q_VT = ("\n\nQuestion: Find all variables that are assigned the value {v} in the text above, directly or through "
        "other variables. Answer with the variable names only, separated by commas.")
Q_CWE = ("\n\nQuestion: What are the 10 most common words in the above list? Answer with the 10 words only, separated "
         "by commas.")
SENT = re.compile(r"(?<=[.!?])\s+")


class Ctx:
    def __init__(self, a):
        man = json.load(open(a.manifest))
        want = man["sources"]["sha256"]
        for p, k in ((a.cnn, "test.parquet"), (a.tok, "tokenizer.json")):
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if got != want[k]:
                raise SystemExit(f"REFUSED: {k} sha256 {got} != manifest {want[k]}")
        used = {s.split(":")[0] for f in man["files"].values() for src in f.get("src", []) for s in src}
        used |= {str(s).split(":")[0] for s in man.get("flush_src", [])}
        for p in json.load(open(a.cont))["prompts"]:
            used |= set(p["src"])
        for it in json.load(open(a.qa))["items"]:
            used |= set(it["src"])
        self.tok = Tokenizer.from_file(a.tok)
        df = pd.read_parquet(a.cnn)
        arts = [(str(i), " ".join(str(t).split())) for i, t in zip(df["id"], df["article"]) if str(i) not in used]
        self.n_excluded = len(df) - len(arts)
        self.arts = []
        for _, body in arts:
            ss = [s for s in SENT.split(body) if s]
            if ss:
                self.arts.append(ss)
        flat = [s for ss in self.arts for s in ss]
        counts = self.tok.encode_batch(flat, add_special_tokens=False)
        it = iter(c.ids for c in counts)
        self.art_tok = [[len(next(it)) for _ in ss] for ss in self.arts]
        wc = Counter(w for ss in self.arts for s in ss for w in re.findall(r"(?<![A-Za-z])[a-z]{4,10}(?![A-Za-z])", s))
        common = {w for w, _ in wc.most_common(300)}
        self.vocab = sorted(w for w, c in wc.items() if c >= 30 and w not in common)

    def ntok(self, s):
        return len(self.tok.encode(s, add_special_tokens=False).ids)


def render(hay):
    """hay: list of (sentence, starts_paragraph)."""
    out = []
    for i, (s, par) in enumerate(hay):
        out.append(("\n\n" if par else " ") + s if i else s)
    return "".join(out)


def cnn_hay(C, rng, budget):
    order = list(range(len(C.arts)))
    rng.shuffle(order)
    hay, t = [], 0
    for ai in order:
        for j, (s, n) in enumerate(zip(C.arts[ai], C.art_tok[ai])):
            if t + n + 1 > budget:
                return hay
            hay.append((s, j == 0))
            t += n + 1
    return hay


def fit(C, build, L):
    """build(budget) -> (content, meta); shrink the budget until the content is <= L - 96 tokens."""
    budget = L - RESERVE
    for _ in range(8):
        content, meta = build(budget)
        n = C.ntok(content)
        if n <= L - 96:
            if n < LOWBAND * L:
                raise SystemExit(f"REFUSED: content {n} tokens < {LOWBAND} x {L}")
            return content, meta, n
        budget -= (n - (L - 96)) + 32
    raise SystemExit("REFUSED: could not fit the content")


def insert(hay, needles, positions):
    """needles inserted as their own sentences before hay index positions[i] (positions on the original hay)."""
    out, by = [], {}
    for nd, p in zip(needles, positions):
        by.setdefault(p, []).append(nd)
    for i, h in enumerate(hay):
        for nd in by.get(i, []):
            out.append((nd, False))
        out.append(h)
    for nd in by.get(len(hay), []):
        out.append((nd, False))
    return out


def num7(rng, taken):
    while True:
        v = str(rng.randint(1000000, 9999999))
        if v not in taken:
            taken.add(v)
            return v


def keys(C, rng, k, share_with=None, n_share=0):
    out, seen = [], set()
    while len(out) < k:
        if share_with and len(out) < n_share:
            a, b = share_with.split("-")
            w = rng.choice(C.vocab)
            key = f"{a}-{w}" if rng.random() < 0.5 else f"{w}-{b}"
        else:
            key = f"{rng.choice(C.vocab)}-{rng.choice(C.vocab)}"
        if key not in seen and key != share_with and len(set(key.split("-"))) == 2:
            seen.add(key)
            out.append(key)
    return out


def needle(k, v):
    return f"One of the special magic numbers for {k} is: {v}."


def depth_pos(hay_len, d):
    return min(hay_len, max(0, int(round(d * hay_len))))


class Collision(Exception):
    pass


def need(cond, what):
    if not cond:
        raise Collision(what)


def gen_niah(C, task, L, i, n, seed, att):
    S = f"{seed}-{task}-{L}-{i}" + (f"-a{att}" if att else "")
    d_t = (i + 0.5) / n
    tk = keys(C, random.Random(S + "-tk"), 1)[0]

    def build(budget):
        r = random.Random(S + "-hay")
        taken = set()
        if task == "mk":
            ks = [tk] + keys(C, random.Random(S + "-k"), 15, share_with=tk, n_share=8)
        elif task == "mv":
            ks = [tk] * 4 + keys(C, random.Random(S + "-k"), 8)
        else:
            ks = [tk] + keys(C, random.Random(S + "-k"), 11)
        rv = random.Random(S + "-v")
        vs = [num7(rv, taken) for _ in ks]
        nds = [needle(k, v) for k, v in zip(ks, vs)]
        if task == "mk":
            gold, q = [vs[0]], Q_MK.format(k=tk)
        elif task == "mv":
            gold, q = vs[:4], Q_MV.format(k=tk)
        else:
            gold, q = vs[:4], Q_MQ.format(k0=ks[0], k1=ks[1], k2=ks[2], k3=ks[3])
        hay = cnn_hay(C, r, budget - sum(C.ntok(x) + 1 for x in nds) - C.ntok(INTRO_NUM + q))
        # the first needle (the target, or the first asked key) at the stratified depth; the rest uniform
        pos = [depth_pos(len(hay), d_t)] + [r.randint(0, len(hay)) for _ in nds[1:]]
        return INTRO_NUM + render(insert(hay, nds, pos)) + q, {"gold": gold, "asked": ks[:4] if task == "mq" else [tk],
                                                                "ks": ks, "needles": nds}

    content, meta, n_tok = fit(C, build, L)
    for nd in meta["needles"]:
        need(content.count(nd) == 1, "needle not unique")
    for v in meta["gold"]:
        need(len(re.findall(rf"(?<![0-9]){v}(?![0-9])", content)) == 1, "gold value not unique")
    for k in set(meta["ks"]):
        want = meta["ks"].count(k) + (1 if k in meta["asked"] else 0)
        need(len(re.findall(rf"(?<![A-Za-z-]){re.escape(k)}(?![A-Za-z-])", content)) == want, f"key {k} count")
    rule = {"mk": "first_num", "mv": "num_set", "mq": "num_seq"}[task]
    return {"task": task, "L": L, "i": i, "depth": round(d_t, 4), "rule": rule, "gold": meta["gold"],
            "content_tokens": n_tok, "content": content}


def uuid4s(rng):
    h = "%032x" % rng.getrandbits(128)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def gen_uuid(C, L, i, n, seed, att):
    S = f"{seed}-uuid-{L}-{i}" + (f"-a{att}" if att else "")
    d_t = (i + 0.5) / n

    def build(budget):
        r = random.Random(S)
        t = C.ntok(INTRO_UUID + Q_UUID.format(k=uuid4s(r)))
        lines = []
        while True:   # exact per-line token counts, in batches of 64 lines
            batch = [f"One of the special magic uuids for {uuid4s(r)} is: {uuid4s(r)}." for _ in range(64)]
            stop = False
            for ln, e in zip(batch, C.tok.encode_batch(batch, add_special_tokens=False)):
                if t + len(e.ids) + 1 > budget:
                    stop = True
                    break
                lines.append(ln)
                t += len(e.ids) + 1
            if stop:
                break
        ti = min(len(lines) - 1, int(d_t * len(lines)))
        k, v = re.match(r"One of the special magic uuids for (\S+) is: (\S+)\.$", lines[ti]).groups()
        return INTRO_UUID + "\n".join(lines) + Q_UUID.format(k=k), {"gold": [v], "key": k}

    content, meta, n_tok = fit(C, build, L)
    need(content.count(meta["key"]) == 2 and content.count(meta["gold"][0]) == 1, "uuid not unique")
    return {"task": "uuid", "L": L, "i": i, "depth": round(d_t, 4), "rule": "first_uuid", "gold": meta["gold"],
            "content_tokens": n_tok, "content": content}


def gen_vt(C, L, i, n, seed, att):
    S = f"{seed}-vt-{L}-{i}" + (f"-a{att}" if att else "")

    def build(budget):
        r = random.Random(S)
        names, vals = [], []
        while len(names) < 20:
            nm = "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
            if nm not in names:
                names.append(nm)
        while len(vals) < 4:
            v = str(r.randint(10000, 99999))
            if v not in vals:
                vals.append(v)
        chains = [names[5 * c:5 * c + 5] for c in range(4)]
        stmts = [[f"VAR {ch[0]} = {vals[c]}."] + [f"VAR {ch[j]} = VAR {ch[j - 1]}." for j in range(1, 5)]
                 for c, ch in enumerate(chains)]
        q = Q_VT.format(v=vals[0])
        hay = cnn_hay(C, r, budget - sum(C.ntok(s) + 1 for ch in stmts for s in ch) - C.ntok(INTRO_VT + q))
        nds, pos = [], []
        for ch in stmts:
            nds += ch
            pos += sorted(r.randint(0, len(hay)) for _ in ch)
        return INTRO_VT + render(insert(hay, nds, pos)) + q, {"gold": chains[0], "names": names, "vals": vals,
                                                               "d": sum(pos[:5]) / 5 / max(1, len(hay))}

    content, meta, n_tok = fit(C, build, L)
    for c in range(4):
        for j, nm in enumerate(meta["names"][5 * c:5 * c + 5]):
            need(len(re.findall(rf"(?<![A-Za-z]){nm}(?![A-Za-z])", content)) == (1 if j == 4 else 2), f"name {nm}")
    for j, v in enumerate(meta["vals"]):
        need(len(re.findall(rf"(?<![0-9]){v}(?![0-9])", content)) == (2 if j == 0 else 1), f"value {v}")
    return {"task": "vt", "L": L, "i": i, "depth": round(meta["d"], 4), "rule": "name_set", "gold": meta["gold"],
            "names": meta["names"], "content_tokens": n_tok, "content": content}


def gen_cwe(C, L, i, n, seed, att):
    S = f"{seed}-cwe-{L}-{i}" + (f"-a{att}" if att else "")

    def build(budget):
        r = random.Random(S)
        words = r.sample(C.vocab, len(C.vocab))
        cw = words[:10]
        def make(n_ucw):
            if 10 + n_ucw > len(words):
                raise SystemExit(f"REFUSED: vocabulary too small ({len(C.vocab)}) for {L}")
            seq = [w for w in cw for _ in range(30)] + [w for w in words[10:10 + n_ucw] for _ in range(3)]
            random.Random(S + "-order").shuffle(seq)
            return INTRO_CWE + "\n".join(f"{k + 1}. {w}" for k, w in enumerate(seq)) + Q_CWE

        n_ucw = 200
        for _ in range(6):   # secant steps on the measured token count (3 entries per uncommon word)
            t = C.ntok(make(n_ucw))
            per3 = t / (300 + 3 * n_ucw) * 3
            step = int((budget - t) / per3)
            if 0 <= step < 3:
                break
            n_ucw = max(1, n_ucw + step - (1 if step < 0 else 0))
        while C.ntok(make(n_ucw)) > budget:
            n_ucw -= 2
        return make(n_ucw), {"gold": sorted(cw), "n_ucw": n_ucw}

    content, meta, n_tok = fit(C, build, L)
    return {"task": "cwe", "L": L, "i": i, "depth": None, "rule": "word_set", "gold": meta["gold"],
            "content_tokens": n_tok, "content": content}


GEN = {"mk": lambda *x: gen_niah(x[0], "mk", *x[1:]), "mv": lambda *x: gen_niah(x[0], "mv", *x[1:]),
       "mq": lambda *x: gen_niah(x[0], "mq", *x[1:]), "uuid": gen_uuid, "vt": gen_vt, "cwe": gen_cwe}


def gen(C, task, L, i, n, seed):
    """A draw that collides with the haystack (needle/key/value/name not unique) is redrawn with an attempt salt;
    deterministic, and the attempt number is recorded."""
    for att in range(10):
        try:
            it = GEN[task](C, L, i, n, seed, att)
            it["attempt"] = att
            return it
        except Collision:
            continue
    raise SystemExit(f"REFUSED: {task} {L} {i}: no collision-free draw in 10 attempts")


def main():
    ap = argparse.ArgumentParser()
    for k in ("cnn", "tok", "manifest", "cont", "qa", "out"):
        ap.add_argument(k)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--n", type=int, default=56)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--lens", default=",".join(map(str, LENS)))
    a = ap.parse_args()
    C = Ctx(a)
    print(f"pool {len(C.arts)} articles ({C.n_excluded} excluded), vocab {len(C.vocab)}", flush=True)
    items = []
    for task in a.tasks.split(","):
        for L in map(int, a.lens.split(",")):
            for i in range(a.n):
                it = gen(C, task, L, i, a.n, a.seed)
                it["id"] = f"{task}-{L // 1024}k-{i:02d}"
                items.append(it)
            print(f"{task} {L}: {a.n} items, content tokens {min(x['content_tokens'] for x in items[-a.n:])}-"
                  f"{max(x['content_tokens'] for x in items[-a.n:])}", flush=True)
    random.Random(f"{a.seed}-order").shuffle(items)
    buf = io.BytesIO()
    for k, it in enumerate(items):
        it["k"] = k
        buf.write((json.dumps(it, sort_keys=True, ensure_ascii=False) + "\n").encode())
    raw = buf.getvalue()
    with open(a.out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as g:
        g.write(raw)
    meta = {"n_items": len(items), "seed": a.seed, "n_per_cell": a.n, "tasks": a.tasks, "lens": a.lens,
            "decompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "gz_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
            "content_tokens_total": sum(x["content_tokens"] for x in items),
            "source": "abisee/cnn_dailymail @96df5e686bee6baa90b8bee7c28b81fa3fa6223d 3.0.0 test (Apache-2.0) sha256 "
                      "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
            "tokenizer_sha256": hashlib.sha256(open(a.tok, "rb").read()).hexdigest(),
            "design_ref": "NVIDIA/RULER @c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a (task definitions; no code copied)",
            "pool_articles": len(C.arts), "excluded_articles": C.n_excluded, "vocab": len(C.vocab)}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
