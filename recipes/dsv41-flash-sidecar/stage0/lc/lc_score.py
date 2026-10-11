#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# RonanLabs 2026-10-10 (Stage 0 amendment C2). Stdlib only.
"""Rule-based scoring for the C2 long-context accuracy eval (LC-1008). One function per registered rule.

Every item carries "rule" and "gold" (list of strings). score(item, reply) -> (correct: bool, recall: float).
  first_num   first 7-digit number in the reply == gold[0]                          (niah multikey, CNN haystack)
  num_set     set of 7-digit numbers in the reply == set(gold)                      (niah multivalue)
  num_seq     7-digit numbers in the reply, in order, == gold (the asked key order) (niah multiquery)
  first_uuid  first UUID in the reply (case-insensitive) == gold[0]                 (niah multikey-3, all-UUID haystack)
  name_set    the item's variable names that occur in the reply == set(gold)        (variable tracking)
  word_set    the list words that occur in the reply == set(gold)                   (common words extraction)
recall = |gold found| / |gold| (RULER-style partial credit; information only). Correct is exact: extra wrong answers
fail the item (stricter than RULER's string_match_all, which ignores extras).
"""
import re

NUM7 = re.compile(r"(?<![0-9])[0-9]{7}(?![0-9])")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
NAME5 = re.compile(r"(?<![A-Za-z])[A-Z]{5}(?![A-Za-z])")
WORD = re.compile(r"[a-z]+")
LISTLINE = re.compile(r"^[0-9]+\. ([a-z]+)$", re.M)
THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(reply):
    return THINK.sub(" ", reply or "")


def list_words(content):
    return set(LISTLINE.findall(content))


def score(item, reply, content=None):
    r, g, rule = clean(reply), item["gold"], item["rule"]
    if rule == "first_num":
        f = NUM7.findall(r)
        return bool(f) and f[0] == g[0], float(g[0] in f)
    if rule == "num_set":
        f = NUM7.findall(r)
        return set(f) == set(g), sum(x in f for x in g) / len(g)
    if rule == "num_seq":
        f = NUM7.findall(r)
        return f == g, sum(x in f for x in g) / len(g)
    if rule == "first_uuid":
        f = [u.lower() for u in UUID.findall(r)]
        return bool(f) and f[0] == g[0].lower(), float(g[0].lower() in f)
    if rule == "name_set":
        f = set(NAME5.findall(r)) & set(item["names"])
        return f == set(g), sum(x in f for x in g) / len(g)
    if rule == "word_set":
        vocab = list_words(content if content is not None else item["content"])
        f = set(WORD.findall(r.lower())) & vocab
        return f == set(g), sum(x in f for x in g) / len(g)
    raise ValueError(f"unknown rule {rule}")
