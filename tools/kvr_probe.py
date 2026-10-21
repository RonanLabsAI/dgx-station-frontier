#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""kvr_probe.py BASE MODEL FIT_JSONL RUNDIR PHASE [--k K] [--pass P] [--length L] [--nonce X] [--timeout S] [--scale F (tests only)]
results/kv-offload.md, test 3 (DeepSeek, second host-tier run): one measurement phase against one booted arm (kvr-off / kvr-on). REPORT ONLY. Writes
RUNDIR/<phase-tag>.json and prints a one-line JSON summary last. Real text only (fit.jsonl public news); no prompt text is written.

Reused UNCHANGED (imports of byte copies): tools/kvo_probe.py (test 2: one(), body(), the /metrics parser and per-request
deltas, Guard, strip, the multi-turn session messages, the --scale shrink) and tools/lc_probe_2hop_ds.py (corpus, needle plan, build,
stream, calibrate). Every request: temperature 0, chat_template_kwargs {"thinking": false}, C1.

EVICTION is deterministic: POST /reset_prefix_cache (GPU prefix cache only) or ?reset_external=true (GPU + host tier), vLLM af7f9488
dev endpoint (VLLM_SERVER_DEV_MODE=1; the arm listens on 127.0.0.1 only). It returns {"success": false} while blocks are held (e.g. an
async store still in flight): retried once a second, up to RESET_TRIES; a reset that never succeeds is an error row (no request sent).
With no connector (arm OFF) ?reset_external=true is a no-op success (Scheduler.reset_connector_cache).

Phases (the runner calls them in the pre-registered order, identical in both arms):
  turn  --k 1..5   8 sessions x 1 turn over one ~250K-token system prefix P (kvo_probe's sessions, unchanged sizes: +~8K tokens of
                   tool output per turn, max_tokens 96). Turn 1: reset(all) first. Turns 2-5: reset(GPU) before session 1 (the "first
                   toucher"); sessions 2-8 ("followers") follow with no reset. ON: the first toucher's P + history come from the host
                   tier; OFF: they are recomputed. Eviction is VERIFIED per turn from the first toucher's /metrics deltas.
  corr  --pass 1|2|3   2 documents (seeds 1016, 1017) x 250K tokens x 10 questions (test 2's items, same wording),
                   max_tokens 64, answers kept (<= 400 chars). Pass 1 and pass 3: reset(all) once at the start (each document's item 1
                   recomputed, items 2-10 GPU hits). Pass 2: reset(GPU) before EVERY item (ON: every item's prefix from the host tier; OFF: every
                   item recomputed). Passes 2 and 3 re-send pass 1's prompts byte-identical (nonce + chars/token in state-corr.json).
  ladder --length L  one L-token document (4 single needles, seed 1019), rows (question in brackets):
                   cold1 [q1] reset(all) | reload1 [q1] reset(GPU) | reload2 [q2] reset(GPU) | gpu1 [q1] | gpu2 [q2] | cold2 [q1] reset(all)
                   ON: reload rows = host reload; OFF: reload rows = recompute (the OFF counterpart of a reload).
  reset            one reset(all); prints {"reset_tries": n} (the runner's boot check that the dev endpoint answers).
Per request: prompt_tokens, TTFT, wall, finish, the answer, and /metrics deltas (GPU prefix hit tokens, connector (host) hit tokens,
offload load / store bytes and seconds). Errors are rows; 2 consecutive timeouts or a dead server end the phase.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvo_probe as K  # noqa: E402
import lc_probe_2hop_ds as B  # noqa: E402

RESET_TRIES = int(os.environ.get("KVR_RESET_TRIES", "60"))   # tests shorten it; 60 x 1 s on the Station
CORR_SEEDS = (1016, 1017)
LADDER_SEED = 1019
LADDER_ROWS = (("cold1", 0, "all"), ("reload1", 0, "gpu"), ("reload2", 1, "gpu"), ("gpu1", 0, None), ("gpu2", 1, None), ("cold2", 0, "all"))
LADDER_MAX = 64
SCALE = 1.0


def reset(base, external, tries=RESET_TRIES, sleep=1.0):
    """POST /reset_prefix_cache[?reset_external=true] until {"success": true} -> number of tries; RuntimeError after `tries`."""
    path = base + "/reset_prefix_cache" + ("?reset_external=true" if external else "")
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(path, data=b"", method="POST")
            r = json.load(urllib.request.urlopen(req, timeout=60))
            if r.get("success") is True:
                return i + 1
            last = r
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
        time.sleep(sleep)
    raise RuntimeError("reset_prefix_cache(external=%s) never succeeded in %d tries (last %s)" % (external, tries, last))


def do_reset(base, kind, rec):
    """kind None|'gpu'|'all' -> True if the request may be sent (rec gets reset / reset_tries or an error)."""
    rec["reset"] = kind
    if not kind:
        return True
    try:
        rec["reset_tries"] = reset(base, kind == "all")
        return True
    except RuntimeError as e:
        rec["error"] = "reset failed: %s" % e
        return False


MX = ("load_bytes", "load_time", "store_bytes", "store_time", "ext_hit_tok", "gpu_hit_tok")


def one_x(base, b, timeout):
    """kvo_probe.one() (unchanged) plus UNROUNDED counter deltas "mx" from scrapes just outside it: kvo_probe.delta() rounds to 3
    decimals, and a 0.5-2 GiB reload over C2C takes only ~3-12 ms of connector load time (engine GB/s needs the full precision)."""
    m0 = K.scrape(base)
    rec = K.one(base, b, timeout)
    m1 = K.scrape(base)
    rec["mx"] = {k: m1[k] - m0[k] for k in MX} if m0 and m1 else None
    return rec


def ladder_plan():
    return [{"state": s, "qi": qi, "reset": r} for s, qi, r in LADDER_ROWS]


def corr_reset_kind(pass_, si, qi):
    """pass 1 / 3: reset(all) once, before the pass's first item (document 2 is cold anyway: no shared prefix), so the host tier
    keeps BOTH documents for pass 2; pass 2: reset(GPU) before every item (pure; tested)."""
    if pass_ == 2:
        return "gpu"
    return "all" if si == 0 and qi == 0 else None


# ---------------------------------------------------------------- phases
def phase_turn(a, text, cpt):
    stp = os.path.join(a.rundir, "state-turns.json")
    st = K.load_state(stp)
    if a.k == 1:
        st = {"nonce": a.nonce, "cpt": cpt, "hist": {str(s): [] for s in range(1, K.SESSIONS + 1)}, "done": 0}
    elif st.get("done") != a.k - 1:
        raise SystemExit("turn %d needs turn %d done first (state says %s)" % (a.k, a.k - 1, st.get("done")))
    nonce, cpt = st["nonce"], st["cpt"]
    a.nonce = nonce   # the prompts carry turn 1's nonce; record that one
    P = "[kvo %s-shared] Shared reference material for all agents follows.\n" % nonce + K.text_at(text, 0, int(K.TURN_P_L * cpt))
    res = {"phase": "turn", "k": a.k, "p_len": K.TURN_P_L, "tool_len": K.TURN_TOOL_L, "req": []}
    g = K.Guard(a.base)
    for s in range(1, K.SESSIONS + 1):
        h = st["hist"][str(s)]
        h.append({"role": "user", "content": K.session_turn_msg(text, cpt, s, a.k)})
        rec = {"session": s, "turn": a.k, "role": "first" if s == 1 else "follower"}
        kind = ("all" if a.k == 1 else "gpu") if s == 1 else None
        if g.skip:
            rec["error"] = "skipped: " + g.skip
        elif do_reset(a.base, kind, rec):
            rec.update(one_x(a.base, K.body(a.model, [{"role": "system", "content": P}] + h, K.TURN_MAX), a.timeout))
            g.after(rec)
        if "error" in rec:
            h.pop()
        else:
            h.append({"role": "assistant", "content": rec["content"]})
        res["req"].append(K.strip(rec))
        print(json.dumps(res["req"][-1]), flush=True)
    st["done"] = a.k if not any("error" in r for r in res["req"]) else -1
    K.save_state(stp, st)
    lo = K.TURN_P_L + (a.k - 1) * K.TURN_TOOL_L
    res["valid"] = all("error" not in r for r in res["req"]) and all(
        K.PT_LO * lo <= (r.get("ptoks") or 0) <= K.PT_HI * (lo + K.TURN_TOOL_L) + K.TURNS * K.TURN_MAX for r in res["req"])
    return res


def phase_corr(a, text, cpt):
    stp = os.path.join(a.rundir, "state-corr.json")
    st = K.load_state(stp)
    if a.pass_ == 1:
        st = {"nonce": a.nonce, "cpt": cpt}
        K.save_state(stp, st)
    elif not st:
        raise SystemExit("corr pass %d needs state-corr.json from pass 1" % a.pass_)
    nonce, cpt = st["nonce"], st["cpt"]
    a.nonce = nonce   # the prompts carry pass 1's nonce; record that one
    res = {"phase": "corr", "pass": a.pass_, "length": K.CORR_L, "req": []}
    g = K.Guard(a.base)
    for si, seed in enumerate(CORR_SEEDS):
        needles, qs, codes = B.plan(seed, K.CORR_L)
        doc = "[kvo %s-corr-%d]\n" % (nonce, seed) + B.build(text, int(K.CORR_L * cpt), needles)
        for qi, qq in enumerate(qs):
            rec = {"seed": seed, "i": qi, "kind": qq["kind"]}
            if g.skip:
                rec["error"] = "skipped: " + g.skip
            elif do_reset(a.base, corr_reset_kind(a.pass_, si, qi), rec):
                rec.update(one_x(a.base, K.body(a.model, [{"role": "user", "content": doc + "\n\nQuestion: " + qq["q"] +
                                                            qq.get("fmt", " Answer with the 6-digit code only.")}], K.CORR_MAX), a.timeout))
                g.after(rec)
                if "error" not in rec:
                    rec["ok"] = qq["want"] in rec["content"]
            res["req"].append(K.strip(rec, keep_content=400))
            print(json.dumps(res["req"][-1]), flush=True)
    res["valid"] = K.valid_len(res["req"], K.CORR_L)
    res["correct"] = sum(1 for r in res["req"] if r.get("ok"))
    res["n"] = len(res["req"])
    return res


def phase_ladder(a, text, cpt):
    L = max(200, int(a.length * SCALE))
    needles, qs, codes = B.plan(LADDER_SEED, L)
    singles = [x for x in qs if x["kind"] == "single"]
    doc = "[kvr %s-ladder-%d]\n" % (a.nonce, L) + B.build(text, int(L * cpt * K.DOC_FILL), needles)
    res = {"phase": "ladder", "length": L, "length_prereg": a.length, "req": []}
    g = K.Guard(a.base)
    for row in ladder_plan():
        qq = singles[row["qi"]]
        rec = {"state": row["state"], "q": row["qi"] + 1}
        if g.skip:
            rec["error"] = "skipped: " + g.skip
        elif do_reset(a.base, row["reset"], rec):
            rec.update(one_x(a.base, K.body(a.model, [{"role": "user", "content": doc + "\n\nQuestion: " + qq["q"] +
                                                        " Answer with the 6-digit code only."}], LADDER_MAX), a.timeout))
            g.after(rec)
            if "error" not in rec:
                rec["ok"] = qq["want"] in rec["content"]
        res["req"].append(K.strip(rec, keep_content=400))
        print(json.dumps(res["req"][-1]), flush=True)
    res["valid"] = K.valid_len(res["req"], L)
    res["correct"] = sum(1 for r in res["req"] if r.get("ok"))
    res["n"] = len(res["req"])
    return res


PHASES = {"turn": phase_turn, "corr": phase_corr, "ladder": phase_ladder}


def tag_of(a):
    return {"turn": "turn-%d" % a.k, "corr": "corr-p%d" % a.pass_, "ladder": "ladder-%d" % a.length}[a.phase]


def main():
    global SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("fit"); ap.add_argument("rundir")
    ap.add_argument("phase", choices=sorted(PHASES) + ["reset"])
    ap.add_argument("--k", type=int, default=0); ap.add_argument("--pass", dest="pass_", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--length", type=int, default=0); ap.add_argument("--nonce", default="")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--scale", type=float, default=1.0, help="tests only: shrink the pre-registered token lengths")
    a = ap.parse_args()
    a.base = a.base.rstrip("/")
    if a.phase == "reset":
        print(json.dumps({"reset_tries": reset(a.base, True)}), flush=True)
        return 0
    if a.scale != 1.0:
        K.apply_scale(a.scale)
        SCALE = a.scale
    a.nonce = a.nonce or str(int(time.time()) % 1000000)
    if a.phase == "turn" and not 1 <= a.k <= K.TURNS:
        raise SystemExit("--k 1..%d" % K.TURNS)
    if a.phase == "ladder" and a.length <= 0:
        raise SystemExit("ladder needs --length")
    text = B.corpus(a.fit)
    cpt = B.calibrate(a.base, a.model, text)
    res = PHASES[a.phase](a, text, cpt)
    res.update(chars_per_token=round(cpt, 3), nonce=a.nonce, scale=a.scale, t_end=round(time.time(), 1))
    out = os.path.join(a.rundir, tag_of(a) + ".json")
    json.dump(res, open(out, "w"), indent=1)
    sm = {k: res[k] for k in ("phase", "valid", "correct", "n") if k in res}
    sm["out"] = out
    print(json.dumps(sm), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
