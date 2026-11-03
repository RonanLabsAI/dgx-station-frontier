# Weight-Space Programming v2 — overnight pilot study report

Date: 2026-11-03 (overnight 2026-11-02 PT). Operator: Muse session under
claims wsp-study-sparks-20261102 / wsp-study-stations-20261102.
Protocol: PROTOCOL-v2.md as reviewed by Astra + Claude Opus (REVIEW-SYNTHESIS.md),
amendments A1–A4 in PROTOCOL-AMENDMENTS.md. All subjects offline, identical
prompts/budgets, frozen scorers; edited targets are copies, never fleet dailies.

## Question

Takehi et al. showed one frontier model (GPT-6 Astra) reading, writing, fixing,
and editing neural-net weights directly. Is that an Astra-specific capability
(possibly trained-in), or do other models — subscription frontier (Claude, Muse)
and local fleet models (GLM-Flash, DeepSeek, GLM-Full on Stations; gpt-oss-120b
on a Spark) — have it too?

## READ wave 1 (identify the game from weights alone; 12 fresh PPO nets)

Accuracy clean / neuron-permuted (chance = 0.25; random-init controls: every
arm guessed a real game on untrained weights):

- Astra: .917 / 1.000
- Claude-Opus: .500 / .583
- Muse: .500 / .417
- DeepSeek V4.1 Flash: .417 / .250
- GLM-5.3-Flash: .333 / .250 (14/28 answers unparsed — no final Game line)
- gpt-oss-120b (Spark): .250 / .250 (chance)
- GLM-5.3 Full: .167 / .167 (only 3 clean + 7 permuted answers parseable;
  2/3 of parsed clean answers correct — completion, not recognition, failed)

Permutation barely hurt Astra or Claude — performance is not mainly memorized
weight-layout recognition at this scale.

## WRITE attempt 1 (construct Breakout weights analytically; normalized vs PPO)

- Muse: 1.199 (flip 1.601) · Claude-Opus: 1.138 (flip 1.513) — hand-built
  weights beat the PPO-trained reference policy on both the game and our
  held-out rule variant.
- DeepSeek: .019 / .036 · gpt-oss: .028 / .028 — valid code, near-random play.
- GLM-Flash: 0/2 valid (code truncated / failed to execute).
- Astra: 1.199 (flip 1.601) — identical returns to Muse's arm on both games,
  i.e. both found flip-invariant constructions at the same saturating optimum.
- GLM-Full: .420 (flip .045) — best local-model WRITE; a real partial policy
  on Breakout that does not transfer to the flipped variant.

## FIX attempt 1 (repair a stuck Freeway policy; crossing rate over 256 episodes)

Stuck baseline: 0% cross / 0% hit / 100% timeout. Random: ~100% hit.

- Claude-Opus: 100% cross, 19 elements changed (0.11%).
- Muse: 100% cross, 41 elements changed (0.24%).
- DeepSeek: 0% cross (still times out), 165 elements.
- gpt-oss: invalid (code never loaded the stuck weights).
- GLM-Flash: invalid (syntax error in generated code).
- Astra: 100% cross, 22 elements changed (0.13%).
- GLM-Full: 41.4% cross / 58.6% hit, 33 elements — un-suppressed "up" at the
  output layer without lane timing (crosses when lucky, dies otherwise).

Both successful repairs independently found the same mechanism: gate the
suppressed "up" action on next-step car positions at lane entries.

## EDIT / Astra positive control (Qwen3.5-0.8B math, Takehi Exp8 replication)

Pristine baselines (our Spark 1 vLLM stack, locked sampling):
tier1 held-out 87.3%, tier2 75.2%, MATH-500 first-100 x2 41.5% (hit-limit 6.0%).
Published unedited MATH-500: 7.9% with 93% hit-limit — see DEBUG-MEMO-CONTROL.md
for the loop-rate discrepancy and editor-dose analysis.

Dumb baselines (sealed eval, same endpoints):
- norm09: MATH-500 subset 39.5%, tier2 76.17% (done)
- norm11: MATH-500 subset 36.0% (hit-limit 10%), tier2 75.0% (done)
- eos20: MATH-500 subset 17.0%, tier2 27.2% (done; early-EOS truncation)
- rand20: MATH-500 subset 40.0%, tier2 75.67% (done)
- ablate-a10: MATH-500 subset 14.5%, tier2 20.3% (done)
Best dumb baseline: MATH-500 rand20 40.0% (pristine 41.5%); tier2 norm09 76.17%
(pristine 75.17%). No dumb edit produces a meaningful gain.

Astra ctrl-1: sealed edit = final-norm only (1,024 elements; an edit class the
original harness refuses — ours did not). Practice evidence 19/35 selects,
fresh 24/40 vs pristine 26/40. Sealed eval: MATH-500 subset 34.5%
(hit-limit 20.5% vs pristine 6.0%), tier2 76.5% (pristine 75.17%).
Ctrl-1 FAILS the A1 band: the norm edit increased looping and lost 7.0pp.
Astra ctrl-2: sealed `astra-ctrl2-1` after 11/50 passes. Edit: A_log += ln(0.9)
on 18 linear-attention tensors (288 elements). Subject's own practice evidence:
selection 17/32 vs pristine 15/32; final unused 8 items tied 5/8 (4/8 strict).
Subject states this does not establish a general improvement. Sealed eval:
MATH-500 subset 41.5% (hit-limit 7.0%), tier2 75.67% (pristine 75.17%).
Ctrl-2 exactly ties pristine on MATH-500 and also FAILS the A1 band.

A1 gate: ctrl passes iff >= +15pp over pristine on MATH-500 subset AND tier2
degradation <= 3pp. Verdict: FAIL (ctrl-1 -7.0pp, ctrl-2 +0.0pp). Per protocol,
the multi-arm EDIT phase does not run. This is a low-dose pilot result, not a
faithful-dose replication verdict; see caveats below.

## Transfer / sham

Not run: gated on a passing positive control, and there is no successful
local edit to transfer. Deferred to any follow-up faithful-dose study.

## Method caveats (summary; full text in DEBUG-MEMO-CONTROL.md)

- Our Astra control dose (~1 h sessions, n<=40 forwards at 1024 tokens) is far
  below the original (8.5–20.3 h, 512-problem paired passes at 32k, activation
  reads). A control miss here bounds the low-dose regime, not the phenomenon.
- The authors' published edited checkpoint, re-run under our harness, scores
  51.0% on our MATH-500 subset (+9.5pp over our pristine) — our stack can
  register the gain, compressed because our pristine rarely loops.

## Fleet state at close

Verified 2026-11-03 05:28 PT per CLOSEOUT-CHECKLIST.md: Stations in split
mode, :8001 glm-5.3-flash and :8003 deepseek-v4.1-flash each answered a tiny
chat with READY. Spark 1 nemotron-3-nano:30b, Spark 3 gpt-oss:120b, and
Spark 2 qwen3.8-flash-next each answered READY. Study-only Spark 1 eval
container removed and Spark 2 kernel server stopped. Claims
wsp-study-sparks-20261102 and wsp-study-stations-20261102 released
(active list verified empty for wsp-study).
