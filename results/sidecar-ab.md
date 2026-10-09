# How much does an RTX PRO 6000 expert sidecar add to a DGX Station GB300?

Each DGX Station GB300 has a second GPU on PCIe: an RTX PRO 6000 Blackwell Max-Q with 96 GB of its own memory. For a
mixture-of-experts model that does not fit in the GB300's HBM, the experts that spill out are normally read from Grace
memory over C2C. A sidecar puts some of those experts on the RTX PRO 6000 instead and computes them there, in parallel
with the GB300. This page measures what that buys, as matched A/B pairs: the same model, the same placement in HBM, the
same harness, on the same day. The only difference is whether the sidecar is on.

Every number below is a row in [`results.jsonl`](../results.jsonl) with a scrubbed log excerpt under [`logs/`](../logs/).

## 1. DeepSeek-V4-Pro across two Stations: E2 only vs E3-A16

**Model and placement.** DeepSeek-V4-Pro-0813 (1.6T parameters) served by stock vLLM multi-node across both Stations,
TP2 + EP2 over one 400G rail, marlin MXFP4 experts. Full recipe, launchers and hooks:
[`recipes/dsv4-pro-two-stations/`](../recipes/dsv4-pro-two-stations/).

| | A: E2 only (no sidecars) | B: E3-A16 (both sidecars on) |
|---|---|---|
| GB300 HBM | dense weights, KV, the hottest experts of every layer (`5,568` rows per rank, pinned) | same |
| RTX PRO 6000 | idle | the `2,400` warmest remaining experts per rank (about 84 GB of weights), W4A16 on b12x with [our W4A16 fix](../recipes/dsv4-pro-two-stations/patches/b12x-w4a16-skip-dropped-routes.patch) |
| Grace memory | everything else | everything else, minus the warm set |
| Session | 2026-10-08 08:38-09:14 PT | 2026-10-08 07:49-08:17 PT |
| Receipt | [e2-pin-hot.log](../logs/dsv4-pro-two-stations/e2-pin-hot.log) | [e3-a16-sidecar.log](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |

Both arms: same container image (vLLM nightly `af7f9488`, pinned by digest), same weights revision, same fabric settings,
same hot-expert map, same benchmark script (`tools/bench_e3.sh`), three repetitions per cell, each on its own seed.
Decode tok/s = C x 1000 / mean TPOT.

- **Random ids** (reproducible): 8,192 random-token prompt, 1,024 forced output tokens, greedy. This is catid's
  published Station-pair shape.
- **Real text** (numbers only): 36 short prompts from a private workload, thinking off, 1,024 forced output tokens. The
  prompts are not published, and the expert maps were fitted to the same workload (on held-out topics), so these rows
  cannot be reproduced from this repo. They show what a usage-aware map buys on real traffic; random ids route almost
  uniformly and understate it.

<!-- results -->
| Measurement | A: E2 only | B: E3-A16 | Change | Receipts |
|---|---|---|---|---|
| Real text, C1, tok/s per user | 40.5 | 43.9 | +8% | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Real text, C16, tok/s aggregate | 152.0 | 230.8 | +52% | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Random ids, C1, tok/s per user | 35.3 | 37.6 | +6% | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Random ids, C16, tok/s aggregate | 118.3 | 164.6 | +39% | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| GSM8K-200 accuracy, % | 97.5 | 97.5 | | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Teacher-forced top-1 flips vs stock vLLM, % (stock against itself: 3.22) | 3.88 | 3.91 | | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log), [stock](../logs/dsv4-pro-two-stations/e1-stock.log) |
| Prefix-cache hit, every reported run, % | 0.00 | 0.00 | | [E2](../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Sidecar vs marlin on the same experts, minimum per-layer cosine on real prompts | | 0.999992 | | [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Sidecar timeouts (rank 0, whole session) | | 0 | | [E3-A16](../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
<!-- /results -->

Notes on the table:

- The GSM8K-200 figure for E2 comes from its own boot the evening before (same map, same launcher); the throughput rows
  are the matched rerun.
- One E3-A16 random-id C16 repetition measured a `3.03%` prefix-cache hit. It was discarded and replaced on a new seed;
  both are visible in the log.
- Stock vLLM on this path is not run-to-run deterministic, so flips are judged against its own floor. Neither arm adds
  a full point over it.

### Why C1 gains little

The sidecar removes most of the Grace reads, and at C16 that matters: each step touches many distinct experts per
layer, so Grace reads grow with the batch while the fixed per-step costs are shared by 16 streams. At C1 a step touches
few experts, and most of the step is something else. An Nsight Systems trace of one E3-A16 real-text C1 request (both
ranks, 22 decode steps each; [receipt](../logs/dsv4-pro-two-stations/e3-a16-c1-step-nsys.log)) breaks the step down:

| Component of the 22.6 ms C1 decode step | ms per step | Source |
|---|---|---|
| Small latency-bound kernels: dense FP8 x FP4 GEMMs (396 per step, each too small to reach bandwidth), hyper-connection mixing, norm / quantisation / elementwise | 10.6 | measured, rank 0 kernel busy time |
| Grace-resident expert reads, critical path | 5.5 | estimate from the trace: per rank only 3.3-3.5, but a cold expert on either rank stalls both at the next all-reduce |
| HBM-resident expert compute, critical path | 3.2 | estimate from the trace (per rank 2.8) |
| Cross-Station all-reduce, pure transfer (123 per step) | 2.3 | estimate: 123 calls at the fastest per-call time seen on either rank; the kernel total of 4.5-5.0 includes waiting on the other rank |
| Sidecar: exposed wait plus pack / publish / scatter-add | 0.75 | measured (0.76 rank 0, 0.73 rank 1); the round trip itself runs under the GB300's own expert calls |
| GPU idle: launch gaps and host work, CUDA graphs on | 0.95 | measured |
| Also on the step: attention and sparse indexer about 1.1, MoE routing about 1.0, other kernels about 1.1 | | measured |

Kernels on different streams overlap, and the critical-path rows take the slower rank layer by layer, so the column
sums to more than the step. The shape of it is what matters:

- **About half of a C1 step is many small kernels** that are bound by launch latency, not by memory or bandwidth. No
  expert tier touches them.
- **The sidecar itself is cheap.** Its round trip hides under the GB300's own expert calls; what is left exposed is
  under a millisecond per step.
- **The Grace residual still costs about a quarter of the step,** not because there is much of it but because cold hits
  are uncorrelated between the two ranks, so almost every step waits for one of them somewhere.
- **The cross-Station collectives are a smaller lever than they look.** Their tail latency is mostly the Grace skew
  above; the transfer itself is about 2.3 ms per step.

So at C1 the levers are kernel fusion and rank-balanced placement, not more expert memory.

## 2. DeepSeek-V4.1-Flash on one Station: sidecar vs Grace only

**Status: [pending same-session run 2026-10-09].** One Station, run back to back in one session: Recipe A is the
GB300 with its RTX PRO 6000 serving the cold experts through the b12x sidecar
([`recipes/dsv41-flash-sidecar/`](../recipes/dsv41-flash-sidecar/)); Recipe B is the same server with the cold experts
read from Grace memory by the GB300, and the RTX PRO 6000 idle. Same image, same expert map, same server flags (32
sequences, DSpark speculative decoding on, prefix caching on), same bench script, a fresh seed per run and a zero
prefix-cache hit on every reported run.

| Measurement | A: GB300 + RTX PRO 6000 sidecar | B: GB300 + Grace only | Change | Receipts |
|---|---|---|---|---|
| Decode, C1, tok/s per user (mean of 3 reps) | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Decode, C16, tok/s aggregate | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Decode, C32, tok/s aggregate | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Prefill, 8K-token prompt, C1, tok/s | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Prefill, 64K-token prompt, C1, tok/s | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Prefix-cache hit, every reported run | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |
| Peak power draw, GB300 / RTX PRO 6000 | [pending same-session run 2026-10-09] | [pending same-session run 2026-10-09] | | |

## 3. Takeaways

- **The sidecar is a throughput tier.** For a mixture-of-experts model that spills out of HBM, the RTX PRO 6000 as a
  warm-expert tier lifted DS-V4-Pro's C16 aggregate by +52% on real text and +39% on random ids, with no measurable
  quality cost (GSM8K-200 unchanged, flips inside stock vLLM's own noise, zero sidecar timeouts).
- **It does little for a single user.** C1 rose +8% on real text and +6% on random ids. At C1 the step is bound by
  small latency-bound kernels and the cross-Station all-reduces, plus the cross-rank skew of the few Grace reads that
  remain, not by expert memory.
- **The gain depends on traffic.** The warm set is chosen from routing counts, so traffic that routes like the
  calibration set gains more than uniformly random token ids.
- **Section 2 will show the one-Station case,** where the model fits one GB300 plus Grace and the sidecar's job is the
  cold tail rather than a warm middle tier.

## Credits

- **original-el8 (Jason Cook)**, [dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research)
  (Apache-2.0): the graph-safe shared-memory expert sidecar design and hook that both recipes use. Our DS-V4-Pro E3 code
  is a port of it (headers and NOTICE kept).
- **b12x / Local Inference Lab**, [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0): the MoE kernels the
  sidecar runs on the RTX PRO 6000. E3-A16 adds our fix for a W4A16 bug on DS-V4-Pro shapes (offered upstream
  separately).
- **J-M-Recipes (James Meadlock)**, [recipes](https://github.com/J-M-Recipes/recipes) (MIT): the pin-hot-experts hook
  behind the E2 placement that both arms share.
- **catid (Christopher Taylor)**, [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks): the
  random-id benchmark shape.
