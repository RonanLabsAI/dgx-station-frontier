# Debug memo — Astra positive control (2026-11-03, operator analysis)

Source: authors' `logs-ref/exp8_math/README.md` (cloned logs), our kernel audit
and eval outputs. Written while ctrl-1 sealed eval and ctrl-2 session run.

## Fidelity gaps vs the original Exp8 (ranked by likely impact)

1. **Editor dose.** Original runs: 74–158 turns, 8.5–20.3 h wall clock, up to
   300 Python kernel runs, forward passes of up to 512 practice problems with
   paired original-vs-edited comparison (256 each), plus teacher-forced reads
   (hidden states / MLP activations / attention). Our ctrl sessions: one Codex
   session each, ~1 h, practice forwards of n=2–40, no activation reads, no
   paired mode. Our control is a far weaker dose of the same editor.

2. **Refused edit classes not enforced here.** Original `save_edit` refused
   edits to the final norm, embedding/output layer, and whole-tensor constant
   multiplies. Our kernel accepts them. Ctrl-1's sealed edit touched ONLY
   `model.norm.weight` (1,024 elements) — an edit the original harness would
   have refused. Any future EDIT arm in this study should enforce the same
   refusals for comparability. (Dumb norm09/norm11 baselines also edit the
   norm; they are operator controls, not subject edits — keep, but label.)

3. **Baseline loop-rate mismatch (unexplained).** Authors' unedited MATH-500:
   7.9% acc, 93.0% of answers cut at 32,768 tokens. Ours (first-100 subset x2,
   same stated sampling temp 0.6 / top_p 0.95 / top_k 20 / presence 1.5):
   41.5% acc, 6.0% hit-limit. The failure mode the published edit fixes is
   largely absent in our stack, compressing measurable gain: the authors' own
   edited checkpoint scored 51.0% under our harness (+9.5pp over our pristine,
   vs +40.2pp in theirs). Candidate causes, not yet isolated:
   subset (first 100 vs full 500 x4 samples), grader (ours numeric/string vs
   math-verify symbolic), prompt suffix wording, vLLM build differences
   (presence-penalty strength, chat-template/thinking defaults).
   Debug probe when Spark 1 is free: serve pristine, dump raw generations on
   8 MATH-500 items, check think-block presence and length distribution.

4. **Practice-pass size.** Original: one pass = up to 512 generations at full
   32,768 max tokens ("85 of its first 96 practice answers ran to the limit").
   Our subject forwards used --max 1024 on n<=40, so the loop phenomenon was
   barely observable to our editor during search.

## Consequence for the A1 gate

A ctrl failure under A1 (+15pp MATH-500 subset, tier2 drop <=3pp) cannot
distinguish "phenomenon does not replicate" from "editor dose and eval-stack
loop-rate too low to show it". If both ctrl sessions miss the band, use the
protocol's debug cycles on items 1/3/4 (bigger paired passes at 32k,
loop-rate probe) before concluding non-replication; a faithful-dose control
(one ~20 h multi-conversation run) is out of overnight scope and would need
Stephen's call.
