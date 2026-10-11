# DeepSeek-V4.1-Flash: GB300 hot experts + RTX PRO 6000 cold-expert sidecar

One DGX Station serves DeepSeek-V4.1-Flash with the routed experts split three ways:

- **hot experts in GB300 HBM:** 285 of 384 per layer, run by DeepGEMM MegaMoE;
- **cold experts on the RTX PRO 6000:** the other 99 per layer, run by original-el8's graph-safe `/dev/shm` sidecar on
  **public b12x master `2cc7f66a`** (el8's published numbers used an unpublished b12x branch; it is not needed);
- **Engram n-gram tables in Grace memory.**

DSpark speculative decoding is on. The design is original-el8's ("M3", Apache-2.0); this folder adds our launcher,
a bring-up gate that works, the settings for 64 concurrent sequences, and receipts.

**DP2** runs this same recipe on both Stations at once, one independent copy each, and splits requests between them.
There is no cross-Station traffic: **one prompt never spans Stations.** DP2 numbers are the sum of two servers.

## Results

Shape: 8,192 random-token prompt, 1,024 forced output tokens, temperature 0, `C` warm-ups then `5 x C` requests
(`vllm bench serve`). Output tok/s as reported by vLLM.

<!-- results -->
| Configuration | C1 per user | C16 | C32 | C64 | Receipt |
|---|---|---|---|---|---|
| One Station, `max-num-seqs 32`, warm boot, first bench | 389.3 | 1,855.6 | 2,170.5 | 2,274.2 (capped at `32` sequences — not a C64 measurement) | [log](../../logs/dsv41-flash-sidecar/one-station-c1-c64.log) |
| **One Station, uncapped** (`max-num-seqs 64`, CUDA graphs to `256`; DP2 config run alone, other Station idle; left / right) | | 1,807.25 / 1,995.02 | 2,261.28 / 2,382.88 | **3,159.86 / 3,047.45** | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| DP2, left copy (both benched simultaneously), `max-num-seqs 64` | 431.26 | 1,836.68 | 2,186.54 | 2,906.39 | [log](../../logs/dsv41-flash-sidecar/dp2-left.log) |
| DP2, right copy | 417.68 | 1,933.80 | 2,361.30 | 2,867.89 | [log](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| **DP2 total (sum of both copies)** | | **3,770** | **4,548** | **5,774** | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| C1 per user, mean of three warm repetitions (left / right) | 445 / 452 | | | | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
<!-- /results -->

<!-- results -->
| Quality and power | Left | Right | Receipt |
|---|---|---|---|
| GSM8K, first `100` test rows, thinking on (`reasoning_effort 50, T 0.3`), same boot, % | 99.0 | 98.0 | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| Bring-up gate: sidecar vs TRT-LLM cold path, minimum cosine on the real prompt (`2,342` tokens) | 0.99977 | 0.99976 | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| Peak power in the DP2 window, GB300 / RTX PRO 6000, W (limits `1,300` / `300`) | 926 / 257 | 917 / 254 | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
<!-- /results -->

Reading the numbers:

- **Solo and simultaneous agree** within run-to-run noise, so the two copies share nothing that matters: DP2 scales at
  about two times one Station.
- **Public comparisons** (see the top-level README): el8's one-Station C16 with the same design is
  [1,628.6](https://github.com/original-el8/dgx-station-gb300-research) tok/s; catid's best two-Station C16 is
  [1,677.7](https://github.com/catid/dgx_station_benchmarks) tok/s; catid's one-model TP2 across two Stations reaches
  [2,117](https://github.com/catid/dgx_station_benchmarks) tok/s at C32 and
  [3,401.6](https://github.com/catid/dgx_station_benchmarks) tok/s at C64. DP2 is a different topology (two servers,
  both RTX PRO 6000s in use) and only comparable on aggregate throughput.
- **C1 is noisy.** DSpark acceptance varies run to run, and single-shot C1 tracks it closely. Quote C1 as a mean of at
  least three repetitions. The means above reuse one prompt set across repetitions (TTFT is prefix-warm), so they read
  a few percent high; the cold single runs on the same boots are the 431.26 / 417.68 tok/s in the first table.
- **The first row's C64 is capped:** that server ran `--max-num-seqs 32`, so at C64 half the requests queue. It is
  not a C64 measurement. The uncapped one-Station C64 is the second row (3,160 / 3,047 tok/s), from the same config
  as DP2 with the other Station idle (fresh seed offset, prefix cache clean between phases).
- **One Station vs catid's two-Station TP2.** At C16 one Station leads catid's one model across two GB300s by
  41.7% / 56.4% (left / right; catid TP2 C16 is [1,275.5](https://github.com/catid/dgx_station_benchmarks) tok/s, and
  its best two-Station C16, PP2, is [1,677.7](https://github.com/catid/dgx_station_benchmarks) tok/s). At C64 it trails
  catid's TP2 ([3,401.6](https://github.com/catid/dgx_station_benchmarks) tok/s) by 7.1% / 10.4%. Our reading, not a
  profiled result: at C16 a decode step is bound by reading weights and expert rows, and moving 99 of 384 experts per
  layer to the RTX PRO 6000 (its own memory bandwidth) shortens that read, while one Station pays no cross-Station
  all-reduce per layer. At C64 the step is compute- and batch-bound: the expert GEMMs grow with the batch, and two full
  GB300s carry twice the tensor throughput of one GB300 plus a sidecar that only holds the cold experts.
- **Equal hardware (two Stations vs two Stations):**

<!-- results -->
| Concurrency | catid, one model across two Stations, tok/s | DP2, one copy per Station, total tok/s | DP2 / catid | Receipt |
|---|---|---|---|---|
| C16 | 1,678 (best two-Station C16, PP2) | 3,770 | 2.25x | [catid](https://github.com/catid/dgx_station_benchmarks), [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| C32 | 2,117 (TP2) | 4,548 | 2.15x | [catid](https://github.com/catid/dgx_station_benchmarks), [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| C64 | 3,402 (TP2) | 5,774 | 1.7x | [catid](https://github.com/catid/dgx_station_benchmarks), [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
<!-- /results -->

  catid numbers are cited from [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks) @`ff8a496e`
  (`deepseek-v4.1-flash/notes/README.md`); nothing from that repo is copied here. DP2 also uses both RTX PRO 6000s,
  which catid's runs do not.

- The one-Station receipt predates our bench-hygiene rules; the audit found its decode windows clean (the only
  prefix-cache hits were warm-up copies hitting each other). Its prefill figures are **not** published here: see the
  withdrawn claims in the top-level README.
- **Clean same-session rerun (2026-10-09), sidecar vs Grace only:** see
  [`results/sidecar-ab.md`](../../results/sidecar-ab.md) section 2, which adds clean prefill and power. It reports both
  output tok/s (this README's metric) and decode tok/s (C x 1000 / mean TPOT, higher because it excludes TTFT); compare
  the C1 means above with its output row, not its decode row.

## Prefill-biased expert slots (Stage 0)

**Result:** giving the early MoE layers more HBM expert slots, paid for out of KV cache, speeds up `16K` single-request
prefill by **+7%** against our own same-session recipe, with no decode loss and long-context accuracy non-inferior. Two
sessions, fresh boots, opposite arm orders: 1.070x and 1.075x. Every number below is public-text or random-id, and the
experiments were pre-registered (bars, gates and the accuracy margin were fixed before any run).

### What it is

- With vLLM's SWA bounded-replay overlay (vllm#58132, the `REPLAY_OVERLAY=1` default of our launcher), eager prefill
  steps run layers `21-39` on each request's last `128` tokens only. So prefill sends nearly all of its expert traffic
  through layers `0-20`.
- In this recipe every cold expert runs on the RTX PRO 6000, and prefill waits for it: the GB300 spends about a third of
  `16K` prefill waiting on the sidecar (wait share `0.349` in the A0 arm of run 1, from the sidecar's own counters).
- **U300** gives layers `0-20` 300 hot experts in HBM instead of 285, and leaves layers `21-39` exactly as el8's
  `rowmap-mix-v1` (285). It is nested: in every layer A0's hot set is a subset of U300's, so decode, which uses all 40
  layers, sees el8's map plus 15 extra hot experts per early layer. The `+315` expert slots come out of KV cache:
  `11.81` -> `5.4` GiB, `2,534,609` -> `1,158,544` tokens (still about `8.8` full `131,072`-token contexts).
- **Credits.** The sidecar design, the hook and the base map (`rowmap-mix-v1`, and the calibration counts and scoring
  U300 is built from) are original-el8's. The prefill shape and harness (`bench_prefill.py`, `16,384` / `65,536`
  random token ids, one output token) are catid's. Ours: the nested map, the per-layer sidecar change it needs, the
  gates and the accuracy test.

### The map is public-only

[`stage0/build_s0_maps.py`](stage0/build_s0_maps.py) (scoring imported from
[`stage0/build_e2_maps.py`](stage0/build_e2_maps.py), derived from el8's `build_rowmap.py`) reads only el8's public
calibration counts (`m3/calibration/counts-{A,B,D,E}.json`, sha256-pinned) and scores each expert with
`rowmap-mix-v1`'s own weighting (`0.3 D + 0.3 E + 0.2 A + 0.2 B` of per-layer route shares). It refuses to write unless
its 285/285 rebuild equals `rowmap-mix-v1` on all 40 layers and U300 is nested over it. No counts from our own traffic
are read. The committed [`stage0/rowmap-s0-u300.json`](stage0/rowmap-s0-u300.json) is byte-identical to a rebuild
([`stage0/MAP-HASHES.txt`](stage0/MAP-HASHES.txt) lists every map we served):

```bash
git clone https://github.com/original-el8/dgx-station-gb300-research /tmp/el8 && git -C /tmp/el8 checkout 7699ae54
python3 stage0/build_e2_maps.py --m3 /tmp/el8/deepseek-v4.1-flash/m3 --out /tmp/maps   # A0 copy, X315, X330
python3 stage0/build_s0_maps.py --m3 /tmp/el8/deepseek-v4.1-flash/m3 --out /tmp/maps   # U300, P45 (placebo)
sha256sum /tmp/maps/*.json                                                            # compare with MAP-HASHES.txt
```

### How to run it

- **Not runnable from this repo alone yet.** el8's sidecar builds one b12x weight plan from layer 0's cold-expert
  count and reuses it for every layer, and draws its autotuning routes from layer 0's count too. That is correct for
  any uniform map, but U300 has 84 cold experts in layers `0-20` and 99 in `21-39`. Our sidecar change (one plan per
  cold count, per-layer tuning ids, a scratch-size check on every layer; unchanged behaviour on a uniform map) is not
  published yet. Do not run U300 on the stock sidecar.
- With that change: put `rowmap-s0-u300.json` next to `rowmap-mix-v1.json` in the hook directory, then
  `ROWMAP=rowmap-s0-u300.json SEQS=64 CG_SIZES="1 2 4 ... 256" launch/launch-dsv41-sidecar.sh sidecar` and `... server`
  with the replay overlay on and `GPU_UTIL=0.95`. Run the bring-up gate (`MEGA_PEER_CHECK` / `MEGA_COLD_TRT=1`) first.
  Every arm here booted the same image, hook, sidecar and settings; only the map differed.

### Results

Shapes: prefill = catid `bench_prefill.py`, random token ids, one output token, a fresh seed per invocation (`16K` C1
mean of 5, `64K` C1 mean of 3, `8K` C16 one invocation). Decode = `8,192` random tokens in, `1,024` forced out,
`vllm bench serve`; decode tok/s = C x 1000 / mean TPOT (with DSpark acceptance); step time = vLLM's server-side
inter-token-latency histogram (one observation per request per engine step, so it does not depend on acceptance).
RT = our public real-text pool ([`bench/realtext`](bench/realtext/)): long = CNN/DM, short = ShareGPT first turns.
Every request is unique from its first token; more than two percent prefix-cache hits fails the run.

<!-- results -->
| Metric | Run `1`: A0 | Run `1`: U300 | Confirmation: A0 | Confirmation: U300 | Receipt |
|---|---|---|---|---|---|
| Prefill `16K` C1, tok/s | 44,182.2 | **47,292.2** | 44,177.7 | **47,509.5** | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| U300 / A0, `16K` C1 | | **1.070x** | | **1.075x** | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Prefill `64K` C1, tok/s | 44,970.5 | 47,635.4 | 44,951.8 | 47,639.4 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Prefill `8K` C16, tok/s | 46,770.8 | 49,893.0 | 46,739.0 | 49,813.0 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Decode C16, tok/s (mean of `3`) | 2,151.05 | 2,206.56 | 2,218.09 | 2,267.64 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Decode step time C16, ms (mean of `3`; registered guard) | 27.941 | 27.289 | 27.382 | 26.797 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Decode C1, tok/s (run `1`: mean of `3`; confirmation: mean of `8`) | 526.65 | 435.79 | 495.31 | 526.95 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Output C64, tok/s (`0` preemptions in every arm) | 3,009.23 | 3,081.45 | 2,934.36 | 3,116.39 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| RT long C16 decode, tok/s (mean of `3`) | 1,442.33 | 1,480.59 | 1,440.15 | 1,481.54 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| RT short C16 decode, tok/s (mean of `3`; registered guard) | 1,594.77 | 1,588.88 | 1,600.22 | 1,583.17 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| GSM8K-200, % | 98.5 | 98.5 | 98.5 | 98.0 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| Cosine gate, minimum over `40` layers (real prompt) | 0.99975 | 0.99976 | 0.99969 | 0.99972 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
| KV cache, GiB | 11.81 | 5.4 | 11.81 | 5.4 | [run 1](../../logs/dsv41-stage0/stage0-run1.log), [confirmation](../../logs/dsv41-stage0/stage0-confirm-c1.log) |
<!-- /results -->

Sidecar timeouts were 0 in every arm. Run 1 booted A0, U300, then the placebo P45 (same slot count as A0, 45 hot
experts per early layer swapped); the confirmation booted U300 first, then A0.

**Verdicts, as registered:**

- **Run 1: PASS.** `16K` C1 1.070x (bar `1.05x`), C16 step time 0.977x A0 (guard: at most `1.03x`), RT-short C16 0.996x
  (guard: at least `0.97x`), C64 with no preemption, GSM8K, cosine gate, timeouts and the greedy-continuation band all
  pass.
- **Confirmation: NOT MET.** `16K` C1 1.075x and every speed guard pass (step time 0.979x, RT short 0.989x), but the
  greedy-continuation band failed: U300 diverged from A0 at the first token on 2 of 6 long prompts (it passed in run
  1). That test measures whether greedy outputs stay token-for-token the same, not whether they are right, and A0
  against itself also diverges. So we ran a scored accuracy test, registered before it ran (next section).
- **Single-user decode is acceptance noise.** Run 1 read U300 at 0.83x A0 at C1, while the placebo (same capacity as
  A0) read 558.7 tok/s; DSpark acceptance swings single runs by tens of percent (each log line carries `draft_acc`).
  The confirmation therefore ran C1 eight times per arm: U300 1.064x A0. The registered decode guard is the C16 step
  time, which does not depend on acceptance.

### Long-context accuracy: LC-1008

`1,008` RULER-style items (Hsieh et al. 2024; task definitions re-implemented, no RULER code or data): multikey,
multivalue and multiquery needles, an all-UUID haystack, variable tracking and common-word extraction, each at `16K`,
`32K` and `64K` tokens, 56 items per cell, on a public CNN/DM 3.0.0 haystack. Greedy, thinking off, exact rule-based
scoring (extra wrong answers fail an item). Arms A0, U300, then A0 again on a fresh boot (A0R) to measure how much the
same recipe moves on its own. Paired analysis on the same items; the margin (2.0 points) was fixed before the run.

<!-- results -->
| | A0 | U300 | A0R (A0 again) | Receipt |
|---|---|---|---|---|
| Accuracy, all `1,008` items, % | 82.84 | 82.44 | 82.04 | [log](../../logs/dsv41-stage0/lc-1008.log) |
| `16K` / `32K` / `64K` items, % | 85.42 / 83.63 / 79.46 | 85.42 / 83.33 / 78.57 | | [log](../../logs/dsv41-stage0/lc-1008.log) |
<!-- /results -->

- U300 minus A0: -0.40 points (20 items only A0 got right, 16 only U300 got right); one-sided 95% interval lower bound
  -1.39 points, above the -2.0 margin: **NON-INFERIOR**. A0 against itself moved -0.79 points, so U300's difference
  is within the recipe's own run-to-run noise.
- By task the differences are small and go both ways (common words 97.02% vs 95.83%, multivalue 89.88% vs 91.07%).
  Variable tracking is hard for this model with thinking off (8.33% vs 10.71%); every arm scores it low.
- [`stage0/lc/`](stage0/lc/) holds the item builder, the scorer and the runner with the paired verdict. The item file is
  not committed (37M prompt tokens) but rebuilds byte-identically (decompressed sha256 `5ec29a90...2812846`) from public
  inputs; the builder's docstring gives the command.

### The trade-off: a bigger prefill map costs decode (E2)

Before Stage 0 we tried moving more slots (E2, same Station, same day, also pre-registered): equal total slots, but
layers `0-20` take them from layers `21-39`. Prefill rose a lot. Decode paid for it.

<!-- results -->
| Arm (hot experts, layers `0-20` / `21-39`) | Prefill `16K` C1, tok/s | vs A0 | Prefill `64K` C1, tok/s | Decode C16 vs A0 | RT short C16 vs A0 | Receipt |
|---|---|---|---|---|---|---|
| A0 (`285` / `285`, el8's map) | 44,546.4 | | 44,909.0 | | | [log](../../logs/dsv41-stage0/e2.log) |
| X315 (`315` / `252`) | 52,607.3 | 1.181x | 52,899.4 | 0.910x | 0.986x | [log](../../logs/dsv41-stage0/e2.log) |
| X330 (`330` / `235`) | 54,951.1 | 1.234x | 55,417.6 | 0.961x | 0.933x | [log](../../logs/dsv41-stage0/e2.log) |
| X330 + `16K` prefill chunk | **59,913.1** | **1.345x** | **60,715.4** | 1.021x | 0.919x | [log](../../logs/dsv41-stage0/e2.log) |
<!-- /results -->

- **Verdict, as registered: no arm passed every gate**, so no confirmation was run. X315 failed the C16 decode guard;
  X330 failed both decode guards and the continuation band; X330 with the `16K` chunk failed the RT-short guard and the
  continuation band. GSM8K, the cosine gate and sidecar timeouts passed everywhere. E2's decode readings are single
  runs (acceptance-noisy), but RT short was down in all three arms.
- At `64K`, X330 on one Station (55,417.6 tok/s) is above catid's published two-Station figures for the same prompt
  length ([52,927](https://github.com/catid/dgx_station_benchmarks) tok/s PP2 with DSpark,
  [53,515](https://github.com/catid/dgx_station_benchmarks) tok/s PP2 autoregressive). We do not claim it: the arm failed
  our gates.
- **What Stage 0 learned from this:** prefill and decode want different expert placements. U300 keeps el8's decode
  layout in layers `21-39` and takes its extra slots in `0-20` from KV cache instead, which keeps a free +7% and no
  more. Bigger prefill gains need a separate prefill map. Splitting roles across two Stations (a prefill server on a
  prefill-biased map, a decode server on el8's map, KV over the 400G link) is the obvious design, but it is incremental
  on published prefill/decode-split work, and for DeepSeek-V4.1 it is blocked today by vllm#57661 (P/D KV transfer
  corrupts the sliding-window layers; the fix, vllm#57662, was still open on 2026-10-10).

### How this compares, and what it is not

- The +7% is measured against **our own same-session recipe** (A0 = el8's `rowmap-mix-v1` on the same boot settings),
  in two sessions. It is not a claim against any published number.
- Our A0 today sits level with el8's published one-Station `16K` prefill
  ([44,500](https://github.com/original-el8/dgx-station-gb300-research) tok/s); we do not claim to beat that figure.
  U300 adds +7% on top of the recipe as we run it.
- Also measured while designing this: staging cold experts from Grace into HBM during prefill was a no-go (the copy
  slowed the concurrent expert GEMMs), and greedy-continuation sameness flips from run to run on the same recipe,
  which is why the accuracy question is answered by LC-1008 rather than by output sameness.

## Hardware and software

| Item | Value |
|---|---|
| GPUs | GB300 (`power.limit` 1,300 W) + RTX PRO 6000 Blackwell Max-Q (300 W) in the same Station, each pinned by UUID |
| Image (server and sidecar) | `vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b` = `vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063` |
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a` (FP8 dense + MXFP4 experts) |
| Hook, sidecar, expert map | original-el8 `deepseek-v4.1-flash/m3` @`7699ae54`: `mega_peer_hook.py`, `sitecustomize.py`, `peer_server2.py`, `rowmap-mix-v1.json` (not vendored) |
| Sidecar kernels | b12x master `2cc7f66a` from source on `PYTHONPATH` (the image's cutlass-dsl 4.7.1 already matches b12x's pins) |
| Memory in use | GB300 about 236 GiB (219 weights + 12 KV); RTX about 74 GiB; Grace about 264 GiB (Engram tables) |

## Setup

```bash
W=~/dsv41/m3; mkdir -p $W/{upstream,local} ~/dsv41/cache-a
git clone https://github.com/original-el8/dgx-station-gb300-research /tmp/el8 && git -C /tmp/el8 checkout 7699ae54
cp -r /tmp/el8/deepseek-v4.1-flash/m3/hook $W/upstream/hook          # mega_peer_hook.py, sitecustomize.py, rowmap-mix-v1.json
cp /tmp/el8/deepseek-v4.1-flash/m3/sidecar/peer_server2.py $W/local/peer_server2_local.py
#   edit peer_server2_local.py: replace its hard-coded home-directory paths with paths inside the container
#   (/w for the hook dir, /b12x, /logs); about five lines.
git clone https://github.com/local-inference-lab/b12x ~/dsv41/b12x && git -C ~/dsv41/b12x checkout 2cc7f66a
# vLLM replay overlay (vllm#58132): apply el8's overlay diff to the five vLLM files copied out of the af7f9488 image and
# place them under $W/overlay-58132/tree/vllm/... (see the launcher for the list), or boot without it: REPLAY_OVERLAY=0.
```

The RTX PRO 6000 must be empty: the sidecar holds about 76 GB.

## Launch

```bash
launch/launch-dsv41-sidecar.sh sidecar          # waits for "serving": 40 layers, 560 CUDA graphs on the RTX
# bring-up gate, first boot on a new image or map:
MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 launch/launch-dsv41-sidecar.sh server   # pass = every layer cos >= 0.999 on real prompts
docker rm -f dsv41-a
# production (defaults MEGA_PEER_CHECK=0 MEGA_COLD_TRT=0); for DP2 run on each Station with 64 sequences:
SEQS=64 CG_SIZES="1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 160 192 224 256" \
  launch/launch-dsv41-sidecar.sh server
launch/stop-dsv41.sh                             # stops dsv41-* containers, waits for the GB300 to free its HBM
```

Serves an OpenAI API on port 8009 as `deepseek-v4.1-flash`. Thinking is on by default; per request use
`{"chat_template_kwargs": {"thinking": false}}` or `{"chat_template_kwargs": {"reasoning_effort": 50}}`.
Warm boot about 8-9 min with the FlashInfer JIT cache mounted (`~/dsv41/cache-a/fi`); the first boot about 30 min.

### Flags that matter

| Flag / env | Why |
|---|---|
| `--moe-backend deep_gemm_mega_moe --enable-expert-parallel` | MegaMoE kernel for the hot experts; the hook targets it |
| `MEGA_ROWMAP=rowmap-mix-v1.json` | which experts are hot (HBM) and cold (RTX) per layer |
| `MEGA_PEER=2`, `VLLM_EXP_PEER2_SMALL=64`, `MEGA_FUSED_SEND=2` | sidecar protocol v2 over `/dev/shm/vllm_peer_tier2` |
| `MEGA_PEER_CHECK=N`, `MEGA_PEER_CHECK_MIN_T=65` | per-layer cosine check, sidecar vs the TRT-LLM cold path. **N must be far above 40 x the number of profile forwards:** vLLM's dummy profile run feeds all-zero inputs and consumes small budgets (a budget of 8 checks nothing real) |
| `MEGA_COLD_TRT=1` | keeps a 62 GiB Grace copy of the cold experts so the check has a reference; 0 in production |
| `--speculative-config` DSpark, `num_speculative_tokens_per_batch_size [[1,4,5],[5,8,3],[9,SEQS,3]]` | 5 draft tokens at low batch, 3 above |
| `--cudagraph-capture-sizes ... 256` with `SEQS=64` | 64 sequences x (1 + 3 draft tokens) = 256 tokens per decode step. Without the larger graph sizes, C64 falls out of the graphs and drops by roughly half |
| `--engram-config '{"cpu_offload": true}'` | Engram tables (two of about 94 GiB) live in Grace memory |
| `--enable-prefix-caching` | as published; **benchmarks must use fresh seeds** because of it (see Methodology) |
| FlashInfer JIT cache mount | saves about 15 min per cold boot |
| `numactl --membind=0` | host allocations stay on the Grace node |

## Benchmarks

- `bench/c1c4.sh` runs C1 and C4 four times on one prompt set (rep 0 discarded) and records DSpark acceptance from
  `/metrics`. Use it to average out acceptance noise; its TTFT is prefix-warm by design.
- For throughput, use the hygiene wrapper pattern in the top-level README: a fresh seed per run and per concurrency, a
  separate warm-up seed, and a `/metrics` prefix-hit check. The DP2 runs used a fresh seed offset per phase.
- `bench/gsm8k.py <url>/v1/chat/completions deepseek-v4.1-flash out.jsonl 16` with `GSM_N=100` reproduces the quality
  rows. It reads `gsm8k_test.jsonl` next to it:
  `curl -L -o bench/gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`.
- `bench/psample.sh out.csv` samples power every 2 s.

## Not included

- A usage-aware expert map fitted to our own traffic. We tested one; it did not change decode at C1 or on random
  tokens, and it was fitted to private prompts, so it stays private. The numbers above all use el8's public map.
- A one-model TP2 launcher across both Stations. Ours was modelled on an unlicensed public script and must be rewritten
  before it can be published; it was also never measured.
