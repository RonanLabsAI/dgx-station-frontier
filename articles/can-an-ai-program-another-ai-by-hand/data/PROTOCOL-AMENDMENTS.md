# Protocol v2 amendments (pre-subject, dated)

## A1 — 2026-11-02 night: endpoints recalibrated from measured pristine baselines
Measured (Spark1 vLLM, locked sampling, n=600 each): tier1 held-out accuracy
0.873 (hit-limit 0.17%), tier2 held-out 0.752 (hit-limit 2.2%). Templated sets
are too easy to show loop-fix headroom (published phenomenon: MATH-500 7.9%,
93% hit-limit). Amendments, frozen BEFORE any subject EDIT run:
- Co-primary endpoints: (i) MATH-500 first-100 subset x2 samples @32768
  (comparability with Takehi Exp8), (ii) fresh held-out tier2 (contamination-
  sensitive). Gain = arm minus best dumb baseline on the SAME endpoint.
- Positive-control band replaces ">=35% fresh held-out" (pristine already 75%):
  Astra control passes iff its sealed edit improves MATH-500-subset accuracy
  by >=15pp over pristine AND does not degrade tier2 by >3pp.
- Primary claim unchanged in form, applied per co-endpoint with Holm across arms.

## A2 — 2026-11-02 night: READ wave structure + Phase 0 outcome
Phase 0 result: 12/15 nets pass frozen gate (eval >= random+1.0); failures
breakout-s11, freeway-s11/s12 replaced by retrain seeds 14/15. READ runs in
waves to bound subscription usage: wave 1 = clean x1 (12 nets) + permuted x1
(12) + random controls x1 (4) per arm; replication waves sized from wave-1
variance. Attempt counts in v2 text (x4 clean) remain the confirmatory target
across waves. Eval engine note: Transformers fallback kernels on Spark 2 were
CPU-bound; MATH evals run on Spark 1 vLLM lane instead (same locked sampling;
presence penalty server-side).

## A3 — 2026-11-02 night: GLM-Flash READ output budget
First GLM-Flash READ pass returned empty content on all items: reasoning
consumed the 8192-token cap before the final answer. Arm rerun with
max_tokens=32768 and reasoning_content accepted as answer text when content
is empty. Empty answers are harness failures, not subject scores.

## A4 — 2026-11-03: FIX text-condition operationalization (pre-subject)
The FIX target is `freeway-stuck-s22`, trained without reward shaping. Its
frozen no-shaping baseline is return 0.0 with 0% crossings, 0% car hits,
and 100% timeouts over 256 episodes (seed 90210). Each subject receives the
same prompt: frozen `games.py`, a deterministic 200-step sampled-action
trace (first 80 steps plus full action counts; the stuck policy chooses
"up" 0 times), and the complete stuck actor at 3 decimals. The returned
NumPy-only script must load the operator-supplied identical `stuck.npz`,
modify copies of those arrays, and save `weights.npz`. No environment
rollouts, training, gradients, or outcome search are permitted in subject
code. Primary FIX score is crossing rate over the same frozen 256-episode
no-shaping evaluation; return, hit rate, timeout rate, changed elements,
changed fraction, and max absolute delta are reported. This amendment
operationalizes the already-specified text condition and changes no gate.
