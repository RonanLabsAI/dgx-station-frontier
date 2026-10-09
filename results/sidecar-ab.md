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

**Model and placement.** DeepSeek-V4.1-Flash on one Station (the right one), run back to back in one session on
2026-10-09: arm A 00:13-00:21 PT, then arm B 00:37-00:47 PT. Nothing else ran on the Station. Recipe:
[`recipes/dsv41-flash-sidecar/`](../recipes/dsv41-flash-sidecar/).

| | A: GB300 + RTX PRO 6000 sidecar | B: GB300 + Grace only |
|---|---|---|
| GB300 HBM | dense weights, KV, the hot experts (285 of 384 per layer, el8's `rowmap-mix-v1`), DeepGEMM MegaMoE | same |
| Cold experts (99 per layer) | on the RTX PRO 6000, b12x sidecar (`MEGA_PEER=2`) | read by the GB300 from pinned Grace memory in TRT-LLM layout (`MEGA_PEER=0`, `MEGA_COLD_TRT=1`); the RTX PRO 6000 holds nothing |
| Receipt | [sidecar-ab-a.log](../logs/dsv41-flash-sidecar/sidecar-ab-a.log) | [sidecar-ab-b.log](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |

Both arms: the same container image (vLLM nightly `af7f9488`, pinned by digest), weights revision, expert map and
server flags (`32` sequences, context `131072`, DSpark speculative decoding `5/3/3`, prefix caching on). The bench is
`vllm bench serve` with random ids, 8,192 in / 1,024 forced out, temperature 0, `5 x C` requests, a fresh seed for
every invocation, the warm-up on its own seed, and `--num-warmups 0`.

Two throughput metrics, labelled on every line because they differ:

- **decode tok/s** = C x 1000 / mean TPOT: generation speed once the prompt is in (the metric of section 1);
- **output tok/s** = vLLM's output token throughput: output tokens / wall time, so it also pays for prefill (TTFT).
  The DS-V4.1 recipe README reports this one.

<!-- results -->
| Measurement | A: sidecar | B: Grace only | A / B | Receipts |
|---|---|---|---|---|
| C1 decode tok/s per user, mean of 3 reps | 528.68 | 268.73 | 1.97x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| C1 output tok/s per user, mean of the same 3 reps | 474.42 | 247.07 | | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| C16 decode tok/s, aggregate | 2,048.66 | 722.67 | 2.83x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| C16 output tok/s, aggregate | 1,738.98 | 645.97 | 2.69x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| C32 decode tok/s, aggregate | 2,616.52 | 927.54 | 2.82x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| C32 output tok/s, aggregate | 2,311.39 | 848.95 | 2.72x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| Prefill, `8K`-token prompt, C1, tok/s | 36.0K | 24.8K | 1.45x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| Prefill, `64K`-token prompt, C1, tok/s | 38.6K | 26.5K | 1.46x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| Prefill, `8K` / `64K`-token prompts, C16, tok/s (A only) | 46.6K / 45.9K | | | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log) |
| Highest prefix-cache hit of any reported run, % (C32; every other run had none) | 0.62 | 0.62 | | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log) |
| GB300 power in the decode window, mean / peak, W | 495.9 / 878.4 | 397.2 / 694.6 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| RTX PRO 6000 power in the decode window, mean / peak, W | 144.7 / 241.31 | 19.5 / 24.22 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
<!-- /results -->

Notes on the table:

- **C1 is noisy.** DSpark acceptance on random ids moves single repetitions by about a tenth in A and a fifth in B
  (A: 534.76 / 483.09 / 568.18; B: 332.23 / 239.23 / 234.74 decode tok/s, all in the receipts). Read the means.
- **Which C1 to compare with what.** The recipe README's C1 figures (389.3 single shot; 445 / 452 as DP2 means) are
  output tok/s, from other boots, and the DP2 means reuse one prompt set, so they read a few percent high. Their
  like-for-like here is the output row (474.42), not the decode row (528.68).
- **C16 against the recipe's headline.** The same harness gave 1,855.6 output tok/s at C16 on 2026-10-07; this
  session's arm A gives 1,738.98. Both runs are clean; the gap is session-to-session and DSpark variance.
- **The C32 prefix hit** (0.62% in both arms, under the 2% bar) is one shared speculative-decoding prefix block, not
  a measured prompt being reused.
- **Power:** 1 s samples of both GPUs over the decode runs (C1 x3, C16, C32), every sample in the receipt. In B the RTX
  PRO 6000 is empty, so its row is idle draw.
- **The 64K-token prefill at C1 is the clean rerun** of a withdrawn figure. At 38.6K tok/s it stays below el8's
  published [44.5K](https://github.com/original-el8/dgx-station-gb300-research) tok/s, so that withdrawal stands (and the
  prompt shapes differ: el8's is `16K` tokens).

### Energy per token, by concurrency

The power rows above pool all five decode runs plus the warm-ups and gaps between them, so they cannot be divided by a
throughput. Here the same 1 s samples are split by run: each measured run's window is its result file's end stamp
minus its duration (both quoted in the bench receipts), centred on the 1 s stamp; warm-ups and gaps are dropped, and
the three C1 repetitions are pooled. tok/J = tok/s / mean W over the same window.

<!-- results -->
| Measurement | A: sidecar | B: Grace only | A / B | Receipts |
|---|---|---|---|---|
| C1, GB300 mean W | 531.6 | 395.82 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C1, RTX PRO 6000 mean W (B: idle, empty) | 133.77 | 19.6 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C1, GPU total W (B: GB300 only / incl. idle RTX) | 665.37 | 395.82 / 415.42 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C1, decode tok/J (B: GB300 only / incl. idle RTX) | 0.795 | 0.679 / 0.647 | 1.17x / 1.23x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C1, output tok/J (B: GB300 only / incl. idle RTX) | 0.713 | 0.624 / 0.595 | 1.14x / 1.20x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C16, GB300 mean W | 777.17 | 449.06 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C16, RTX PRO 6000 mean W (B: idle, empty) | 212.77 | 18.87 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C16, GPU total W (B: GB300 only / incl. idle RTX) | 989.94 | 449.06 / 467.93 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C16, decode tok/J (B: GB300 only / incl. idle RTX) | 2.07 | 1.61 / 1.54 | 1.29x / 1.34x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C16, output tok/J (B: GB300 only / incl. idle RTX) | 1.76 | 1.44 / 1.38 | 1.22x / 1.27x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C32, GB300 mean W | 788.06 | 480.93 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C32, RTX PRO 6000 mean W (B: idle, empty) | 217.99 | 19.38 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C32, GPU total W (B: GB300 only / incl. idle RTX) | 1,006.05 | 480.93 / 500.31 | | [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C32, decode tok/J (B: GB300 only / incl. idle RTX) | 2.60 | 1.93 / 1.85 | 1.35x / 1.40x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
| C32, output tok/J (B: GB300 only / incl. idle RTX) | 2.30 | 1.77 / 1.70 | 1.30x / 1.35x | [A](../logs/dsv41-flash-sidecar/sidecar-ab-a.log), [B](../logs/dsv41-flash-sidecar/sidecar-ab-b.log), [power](../logs/dsv41-flash-sidecar/sidecar-ab-power.log) |
<!-- /results -->

- **Two readings of B.** "GB300 only" leaves the idle RTX PRO 6000 out, as if the Station had no sidecar card;
  "incl. idle RTX" charges B for the card sitting in the same Station, empty. Every Station GB300 ships with the card,
  so the second is the like-for-like box comparison; the first is the conservative one for the sidecar.
- **The sidecar draws more power and still spends less energy per token,** because it more than doubles throughput:
  at C32 the two GPUs draw about twice B's power for 2.82x the decode rate.
- **GPU board power only** (nvidia-smi `power.draw`). Grace CPU, LPDDR5X, fans and PSU losses are not measured. Arm B
  streams its cold experts from Grace memory over C2C, so its unmeasured share is probably the larger one, and these
  ratios likely understate the sidecar's advantage at the wall.
- **Decode tok/J includes prefill power.** The window is the whole run, prefill included, but decode tok/s excludes
  TTFT; output tok/J is the self-consistent wall-time figure. Moving every window by half a second either way changes
  each mean by under 1.2%.

## 3. Takeaways

- **The sidecar is a throughput tier.** For a mixture-of-experts model that spills out of HBM, the RTX PRO 6000 as a
  warm-expert tier lifted DS-V4-Pro's C16 aggregate by +52% on real text and +39% on random ids, with no measurable
  quality cost (GSM8K-200 unchanged, flips inside stock vLLM's own noise, zero sidecar timeouts).
- **It does little for a single user.** C1 rose +8% on real text and +6% on random ids. At C1 the step is bound by
  small latency-bound kernels and the cross-Station all-reduces, plus the cross-rank skew of the few Grace reads that
  remain, not by expert memory.
- **The gain depends on traffic.** The warm set is chosen from routing counts, so traffic that routes like the
  calibration set gains more than uniformly random token ids.
- **On one Station the sidecar is worth more, and at every concurrency.** When the model fits one GB300 plus Grace,
  the RTX PRO 6000 serves the cold tail instead of a warm middle tier, and that tail is read on every step: C1 decode
  doubles (1.97x) and C16 / C32 aggregate decode rises 2.83x / 2.82x over reading the same experts from Grace.
- **Prefill gains less** (1.45x at 8K, 1.46x at 64K): long prompts batch the expert reads, so Grace bandwidth hurts
  them less than it hurts decode.
- **The power cost is modest.** In the decode window the GB300 averages 495.9 W with the sidecar vs 397.2 W without,
  and the RTX PRO 6000 adds 144.7 W, for 2.8x the C32 decode throughput. Split by run, GPU energy per decode token
  falls at every concurrency: 1.35x the C32 decode tok/J of Grace only (1.40x counting the idle card in B).

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
