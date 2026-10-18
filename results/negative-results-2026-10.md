# Negative results, October 2026

Things we tried on the Stations that did not clear the bar we set for them before the run. Each section says what the
idea was, the pre-registered bar, what we measured and why it missed. Everything here ran on public traffic only:
random token ids and the public CNN/DailyMail pool of [`recipes/kimi-k3/bench/build_pool_k3.py`](../recipes/kimi-k3/bench/build_pool_k3.py)
(where an expert map was used, it was fitted to random ids and held-out fit slices of that same public pool).

Every number below is a row in [`results.jsonl`](../results.jsonl) with a scrubbed log excerpt under
[`logs/negatives-2026-10/`](../logs/negatives-2026-10/). decode tok/s = C x 1000 / mean TPOT throughout. Random-id rows
are labelled as such; they are a control, not real-text speed.

## 1. DeepSeek-V4-Pro: NVFP4 experts through FlashInfer's b12x kernel do not run on the GB300

**Idea ("Route B").** Our DS-V4-Pro recipe ([`recipes/dsv4-pro-two-stations/`](../recipes/dsv4-pro-two-stations/)) runs
its routed experts on the GB300 with marlin MXFP4 W4A16. Serving them as NVFP4 (W4A4) through FlashInfer's b12x MoE
kernel might be faster. Before converting any weights, a model-free microbench on one GB300 asked two questions in
order: does b12x admit and execute this model's MoE geometry on SM103 at all, and if so is it at least `1.26x` faster
than marlin at one to eight tokens per call? Pre-registered: an admission failure is terminal (no fallback backend, no
version change, no retry).

**Result: infeasible.** b12x compiles its micro-kernel for the GB300 and then rejects the architecture
(FlashInfer 0.7.0, image `af7f9488`, geometry read from the checkpoint: hidden `7168`, expert intermediate `3072`,
`384` routed experts, `192` per rank at EP2, top-k `6`):

```
expects arch to be one of [base_dsl.Arch.sm_120a, base_dsl.Arch.sm_120f, base_dsl.Arch.sm_121a, base_dsl.Arch.sm_121f], but got sm_103a
  --> .../flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py:537
      mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
```

<!-- results -->
| Measurement | Result | Receipt |
|---|---|---|
| FlashInfer b12x NVFP4 MoE executes on the GB300 (sm_103a); `1` = yes | 0 | [log](../logs/negatives-2026-10/dsv4pro-fp4moe-route-b.log) |
<!-- /results -->

The b12x MoE path is written for the SM120/SM121 warp-level MMA (RTX PRO 6000 and GB10-class parts), not for the
GB300's tensor cores. No timing was taken. On the GB300 the NVFP4 direction for DS-V4-Pro ends here; b12x remains the
right kernel for the RTX PRO 6000 sidecar, where we already use it.

## 2. Kimi K3: Humming tile fix, cold-expert staging and a split-layer size hint (E-A)

**Idea.** Kimi K3 at C16 is bound by Grace: most of its int3 experts (`408` GiB per rank) are read over C2C, and long
prefills stall the decode steps they interleave with ([`recipes/kimi-k3/`](../recipes/kimi-k3/), section C). Three
independent switches, all acting only on large eager (prefill) MoE calls, so decode CUDA graphs are unchanged:

- **TILE:** Humming's tuning table picks a slow 64-row tile for MoE calls above a size threshold (a lookup quirk in its
  last table interval); move those calls to the nearest interval that picks 128-row tiles.
- **STAGE:** copy the next layer's cold expert rows from pinned Grace memory into a two-slot HBM ring on a
  high-priority side stream while the current layer computes.
- **HINT:** on a boot that splits layers into hot (HBM) and cold (Grace) subsets, give Humming the subsets' true
  per-expert load instead of an under-stated one. The split map was fitted half to random-id and half to public
  real-text route counts (fit slices that are never benched).

Arms, one session on 2026-10-13: `kr` (control, no drafter, every switch off), `kr-tile`, `kr-stage` (TILE + STAGE) on
one boot; `ke2-hint` (split map + HINT) on a second boot. Two more arms, HINT + TILE with and without STAGE, never ran:
their numerical gate did not fire (below). Pre-registered bar per arm: C16 at least `1.20x` the same-session control on
both real text and random ids, and C1 at least `0.97x`.

<!-- results -->
| Arm | Real text C1, tok/s (mean of three) | Real text C16, tok/s | C16 / control | Random ids C1, tok/s (mean of three) | Random ids C16, tok/s | C16 / control | Real text C16 median TTFT, s | Receipt |
|---|---|---|---|---|---|---|---|---|
| kr (control) | 18.08 | 44.46 | | 18.1 | 56.64 | | 23.59 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
| kr-tile | 18.08 | 45.08 | 1.0139 | 18.09 | 54.12 | 0.9555 | | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
| kr-stage | 18.09 | 46.88 | 1.0544 | 18.11 | 59.61 | 1.0524 | 16.5 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
| ke2-hint | 18.71 | 44.99 | 1.0119 | 21.59 | 57.06 | 1.0074 | 37.02 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
<!-- /results -->

<!-- results -->
| Also measured | Value | Receipt |
|---|---|---|
| ke2-hint over control, C1: real text / random ids | 1.0348 / 1.1928 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
| GSM8K, first `200` test rows, control arm, % | 98.5 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
| HINT + TILE arms: numerical-gate records from the hot subset, rank `0` (at least `3` required) | 0 | [log](../logs/negatives-2026-10/kimi-k3-ea-tile-stage-hint.log) |
<!-- /results -->

- **No arm passes.** The best, kr-stage, lifts C16 by 1.054 (real text) and 1.052 (random ids) of the control, far
  under 1.20, though it does cut the median C16 time to first token on real text from 23.59 to 16.5 s, and the
  `3.6K`-token prefill from `2.78` to `1.78` s.
- **TILE alone is flat** (1.014 and 0.956 of the control at C16).
- **The split map with HINT** gains on random ids at C1 (1.193 of the control) but barely on real text (1.035), is flat
  at C16, and makes real-text time to first token worse (37.02 s).
- **The two composed arms are a kit problem, not a speed result.** With HINT and TILE both on, the hot-subset calls
  never reached the checked path on either rank, so the pre-registered numerical gate could not fire and the runner
  did not measure them.
- Every arm kept quality: coherence 11 of 12 auto-checked on every measured arm, GSM8K-200 98.5 on the control, zero
  prefix-cache hits on every reported run.

## 3. Kimi K3 + DSpark: two draft tokens instead of three (accept-k)

**Idea.** The published K3 drafter schedule verifies three draft tokens at one user. A shorter draft costs less per
verify step and loses less when the third token is rejected; a model of the trade-off predicted a dead heat (`27.1`
against `27.2` tok/s). Arm A: `[[1,1,3],[2,3,3],[4,16,0]]` (three drafts at one to three users); arm B:
`[[1,1,2],[2,3,2],[4,16,0]]`. Pre-registered: adopt B if its real-text C1 is at least `1.03x` A with the flips and
GSM8K gates met; close the idea below `0.97x`; in between keep three (the incumbent wins ties). A third arm (two drafts
at one user only) was planned and not run.

<!-- results -->
| Measurement | A: three drafts | B: two drafts | B / A | Receipt |
|---|---|---|---|---|
| Real text C1 decode tok/s per user, mean of three | 26.28 | 26.42 | 1.0053 | [log](../logs/negatives-2026-10/kimi-k3-accept-k.log) |
| Real text C2 decode tok/s, aggregate (one run) | 30.42 | 30.88 | 1.0151 | [log](../logs/negatives-2026-10/kimi-k3-accept-k.log) |
| Real text C1 mean accepted length, mean of three | 2.743 | 2.382 | | [log](../logs/negatives-2026-10/kimi-k3-accept-k.log) |
| Random ids (greedy), C1 decode tok/s per user, mean of three | 33.87 | 28.66 | | [log](../logs/negatives-2026-10/kimi-k3-accept-k.log) |
| Teacher-forced top-1 flips against A's reference, % (A against itself: the floor) | 0.0 | 0.0 | | [log](../logs/negatives-2026-10/kimi-k3-accept-k.log) |
<!-- /results -->

**Result: a tie, so three drafts stay.** B is within half a percent of A at one user and one and a half percent at two,
inside the band where the pre-registration keeps the incumbent. The shorter draft accepts fewer tokens per step and
makes up for it in step time, as predicted. GSM8K was not rerun in this window. On greedy random ids B looks worse,
but those runs swing with how repetitive each random prompt's continuation is (A: 32.69-35.17, B: 20.8-33.9 decode
tok/s per repetition) and are not the bar.

## 4. Kimi K3 + DSpark: prefetching the verify step's experts (prefetch P0): invalid

**Idea.** At one user with the drafter, every verify step runs four tokens (the accepted prefix plus three drafts)
through every layer. If the set of experts those four tokens touch in a layer were small, it could be copied from
Grace to HBM on a side stream ahead of the layer and the copy hidden behind compute. A probe boot measured the union
per layer on both ranks (and wrapped the step in NVTX ranges for an Nsight capture). Pre-registered: build the
prefetch hook only if the mean union is at most `10` experts per layer, the copy takes at most half the verify-phase
compute, and the probe changes C1 by under `2%`; close the idea if the union is above `14`, the copy is at least as long
as the compute, or the capture cannot attribute verify-phase time (measurement invalid).

<!-- results -->
| Measurement | Value | Receipt |
|---|---|---|
| Real text C1 decode tok/s per user, mean of three: base boot / probe boot | 26.1 / 25.87 | [log](../logs/negatives-2026-10/kimi-k3-prefetch-p0.log) |
| Probe's effect on C1, % | -0.89 | [log](../logs/negatives-2026-10/kimi-k3-prefetch-p0.log) |
| Mean union of experts touched by one verify step, per layer, per rank | 22.28 | [log](../logs/negatives-2026-10/kimi-k3-prefetch-p0.log) |
| Estimated copy time for that union, ms per step | 96.28 | [log](../logs/negatives-2026-10/kimi-k3-prefetch-p0.log) |
<!-- /results -->

**Result: invalid by its own rule.** No Nsight report came back from the probe boot, so verify-phase compute could not
be measured, which the pre-registration defines as an invalid measurement. What was measured points the same way a
valid run would have: the union averages about `22` experts per layer per rank, above the line of `14` at which the idea
closes, and copying it at the best measured side-stream rate (`266.4` GB/s) would take about `96` ms per step against a
whole verify step of about `105` ms, so there is nothing left to hide it behind. The probe itself cost under one
percent of C1. No valid rerun is in this release.

## 5. Kimi K3 + DSpark: RTX PRO 6000 warm-expert sidecars (sidecar P0): KILL

**Idea.** The sidecar that lifted DS-V4-Pro's C16 ([sidecar A/B](sidecar-ab.md)), ported to K3: `4,900` warm expert rows
per rank move from Grace to the RTX PRO 6000 of each Station, held there as lossless 4-bit (`uint4b8`) copies of the
int3 weights and run with marlin, in a separate process per GPU, with BF16 activations over shared memory. The map is
fitted half to random-id and half to public real-text route counts. A kernel gate (P0) first checked the RTX PRO 6000
kernel against the Grace reads it replaces; then two boots in one session: the published DSpark recipe (control) and
the same with both sidecars. Pre-registered: pass at real-text C16 `>= 1.5x` the control with every quality gate
(GSM8K-200 within one point of `98.5`, teacher-forced flips at most one point over the control's own floor, coherence
at least 11 of 12, zero sidecar timeouts); KILL below `1.2x` or on any gate miss.

<!-- results -->
| Measurement | Control | Sidecars | Sidecars / control | Receipt |
|---|---|---|---|---|
| Real text C1 decode tok/s per user (one run) | 25.93 | 28.49 | 1.0987 | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| Real text C4 decode tok/s, aggregate (one run) | 30.15 | 32.66 | 1.0833 | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| Real text C16 decode tok/s, aggregate (mean of two) | 42.95 | 48.33 | 1.1251 | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| Random ids (greedy), C16 decode tok/s, aggregate (mean of two) | 54.87 | 60.39 | 1.1005 | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| Teacher-forced top-1 flips against the control's reference, % (control: its own floor) | 0.0 | 2.888 | | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| GSM8K, first `200` test rows, % | | 97.5 | | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| Sidecar timeouts, whole session, rank `0` | | 0 | | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
| P0: sidecar kernel time / Grace read time it replaces, one call (bar `<= 0.5`) | | 0.328 | | [log](../logs/negatives-2026-10/kimi-k3-sidecar-p0.log) |
<!-- /results -->

**Result: KILL, twice over.** The flips gate failed: 2.888% of teacher-forced positions changed their top-1 token
against a floor of 0.0, nearly three points where the bar allowed one. And the speed missed too: 1.125 of the control
at real-text C16 is under the `1.2x` KILL line, let alone the `1.5x` pass. GSM8K (97.5), coherence (11 of 12) and zero
timeouts held, and the kernel itself passed P0 comfortably. We did not diagnose where the flips come from.

## 6. Not reported

- **The original split-map run that the E-A `ke2` boot reproduces** (K-E2, 2026-10-10): its raw logs are not in our
  archive, so it is not reported. The `ke2-hint` rows above come from a boot of the same configuration.

## Credits

- **b12x / Local Inference Lab**, [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0), as packaged in
  FlashInfer 0.7.0: the kernel tested in section 1.
- **original-el8 (Jason Cook)**, [dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research)
  (Apache-2.0): the separate-process expert sidecar design that the K3 sidecar in section 5 follows.
- **Moonshot AI** (Kimi K3), **Vellum** (the 3-bit quantisation) and **Inferact** (the Kimi-K3-DSpark drafter); no
  weights are included.
- **catid (Christopher Taylor)**, [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks): the
  random-id benchmark shape.
