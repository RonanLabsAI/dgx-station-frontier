# DeepSeek-V4-Pro-0813 across two DGX Stations (E1, E2, E3)

DeepSeek-V4-Pro-0813 has 1.6T parameters (49B active per token): MXFP4 routed experts plus FP8 dense weights, 892.8 GB
on disk. It does not fit in two GB300s. This recipe serves it on two DGX Stations with stock vLLM multi-node (TP2 + EP2
over one 400G rail) and places the routed experts across three memory tiers per Station:

| Stage | GB300 HBM | Grace memory (read by the GB300 over C2C) | RTX PRO 6000 |
|---|---|---|---|
| **E1**, stock vLLM | dense weights, KV, the routed experts of the later layers | the routed experts of the first layers (`200 GiB` per rank, UVA offload) | idle |
| **E2**, pin-hot split | the most-used experts of every layer (`5,568` rows per rank) | the rest | idle |
| **E3**, warm tier | as E2 | what E2 leaves cold, minus the warm set | the `2,400` warmest E2-cold experts per rank, computed by a sidecar |

E2 ports J-M-Recipes' pin-hot-experts hook (MIT) to vLLM's marlin MXFP4 path with expert parallelism. E3 ports
original-el8's shared-memory sidecar (Apache-2.0) from DS-V4.1's MegaMoE path to the generic marlin path, so the GB300
keeps every weight and simply skips the experts the sidecar serves. We did not find a published DS-V4-Pro run on fewer
than four GB300s (search of 2026-10-06).

## Results

Two benchmarks, three repetitions each, decode tok/s = C x 1000 / mean TPOT:

- **random ids:** 8,192 random-token prompt, 1,024 forced output tokens, greedy (`vllm bench serve`). Reproducible.
- **real text:** 36 short prompts (12 held-out topics x 3 phrasings) from a private workload, thinking off, 1,024
  forced output tokens, streaming TPOT. **Not reproducible from this repo**: the prompts are private, and E2/E3 use expert
  maps fitted to the same private workload (held-out topics were not used to fit them). Random token ids route almost
  uniformly across experts, so they understate what a usage-aware map buys on real traffic; real text shows it.

Every E2, E3 and speculative run used a fresh seed and measured 0.00% prefix-cache hits (one E3-A16 repetition at
`3.03%` was discarded and replaced; it is visible in the log).

<!-- results -->
| Stage | Random ids C1 | Random ids C16 | Real text C1 | Real text C16 | GSM8K-200 % | Top-1 flips vs E1 % | Receipt |
|---|---|---|---|---|---|---|---|
| E1, stock | 37.83 | 109.54 | 37.88 | 114.91 | 98.0 | 3.22 (E1 vs itself: the floor) | [log](../../logs/dsv4-pro-two-stations/e1-stock.log) |
| E2, pin-hot split | 35.32 | 118.3 | 40.5 | 152.02 | 97.5 | 3.88 | [log](../../logs/dsv4-pro-two-stations/e2-pin-hot.log) |
| E3-A8, sidecar W4A8-MX, public b12x | 37.68 | 166.11 | 43.86 | 233.04 | 96.5 | 3.45 | [log](../../logs/dsv4-pro-two-stations/e3-a8-sidecar.log) |
| **E3-A16, sidecar W4A16, b12x + our patch** | **37.58** | **164.61** | **43.93** | **230.75** | **97.5** | **3.91** | [log](../../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| *Superseded, unfixed:* E3-A16 + DSpark drafter, `K = 5` | 21.62 | 72.11 | 77.56 | 270.86 | engine hang: 200 of 200 requests errored | | [log](../../logs/dsv4-pro-two-stations/dspark-experimental.log) |
| E3-A16 + DSpark + rank-symmetric guard + acceptance-gated `K` | 30.07 | 114.57 | 76.51 | 298.25 | 97.5 | 3.68 | [log](../../logs/dsv4-pro-two-stations/dspark-guard-r31b.log) |
| + skip-draft (context-only step when `K = 0`) | 40.68 | 160.93 | 76.57 | 303.59 | 98.0 | 4.62 | [log](../../logs/dsv4-pro-two-stations/dspark-skipdraft-r31c.log) |
| **Final: + mixed expert map (see note)** | **37.12** | **154.29** | **87.72** | **338.41** | **97.5** | **3.66** | [log](../../logs/dsv4-pro-two-stations/dspark-final-v4p.log) |
<!-- /results -->

<!-- results -->
| Detail | Value | Receipt |
|---|---|---|
| E3-A16 over E2, C16 gain, real text / random ids, % | +51.8 / +39.1 | [E2](../../logs/dsv4-pro-two-stations/e2-pin-hot.log), [E3-A16](../../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| Sidecar vs marlin on the same experts, minimum cosine over `61 layers x 2 ranks` on real prompts: A16 / A8 | 0.999992 / 0.998945 | [A16](../../logs/dsv4-pro-two-stations/e3-a16-sidecar.log), [A8](../../logs/dsv4-pro-two-stations/e3-a8-sidecar.log) |
| Sidecar timeouts, E3-A16 session (rank 0) | 0 | [log](../../logs/dsv4-pro-two-stations/e3-a16-sidecar.log) |
| E1: routed experts offloaded to Grace per rank, GiB | 200.81 | [log](../../logs/dsv4-pro-two-stations/e1-stock.log) |
| E1: KV cache, tokens (`131,072`-token context) | 476,607 | [log](../../logs/dsv4-pro-two-stations/e1-stock.log) |
| E1: warm reboot, launch to API ready, min | 5.85 | [log](../../logs/dsv4-pro-two-stations/e1-stock.log) |
<!-- /results -->

Reading the numbers:

- **E1 clears a useful bar** for a 1.6T model on two desktops: 37.8 tok/s for one user, 109.5 tok/s aggregate at C16,
  GSM8K-200 at 98.0%.
- **The tiers help batches, not single users.** E3 lifts real-text C16 by about half over E2 and random-id C16 by about
  two fifths. C1 hardly moves in any stage: at C1 the step is dominated by many small latency-bound kernels (dense GEMM,
  hyper-connection mixing, norms, quantisation) and the cross-Station all-reduces, not by expert reads. A one-shot
  cross-Station all-reduce is a later project.
- **A16 vs A8.** Same throughput within noise. A16 is the only arm that meets a `0.999` per-layer cosine bar, because
  marlin itself is W4A16; A8 sits at the activation-quantisation floor (about `0.9989`). A16 needs our b12x patch
  ([patches/](patches/)); A8 runs on public b12x master.
- **Flips.** Stock vLLM on this path is not run-to-run deterministic: E1 against itself flips 3.22% of teacher-forced
  top-1 tokens. E2 and E3 add well under one point over that floor.
- **DSpark (the bundled drafter) is servable after two fixes.** Unfixed, it hung on thinking-mode GSM8K at C16: an
  out-of-memory error while JIT-loading a drafter kernel on the non-output rank was swallowed, so that rank skipped the
  drafter's collectives and the ranks desynchronised. The guard makes every failure rank-symmetric and attributable
  (failfast, drafter kernel warm-up, KV headroom, `K` decided in the scheduler and broadcast). Skip-draft then runs a
  context-only step when the scheduler sets `K = 0`, so random-id traffic no longer pays for a drafter that predicts
  nothing. The final stack passed a 20-minute mixed soak at C16 with a streaming client: 801 of 801 requests completed,
  0 timeouts, no hang ([log](../../logs/dsv4-pro-two-stations/dspark-final-v4p.log)). A same-session A/B shows
  skip-draft does not change flips (3.66 on, 3.64 off) and is worth +25% on random ids at C1 (37.12 vs 29.65 tok/s).
- **Note on the final row.** Its expert map was re-fitted from route counts over thinking-on and thinking-off traffic
  (private and public text, GSM8K). The fit counts included the C1 real-text prompts measured here, so the 87.72 tok/s
  C1 figure (81.65 with thinking on) is in-sample for the map. Random ids, the C16 run (338.41 tok/s; 330.94 with
  thinking on; 9 of its 12 prompts were not in the fit), GSM8K-200, flips and the soak are not. A clean held-out
  public-text measurement is still owed: it follows in the next section.

## Public-text results (2026-10-09)

The same final stack (E2 + E3-A16 + DSpark `K = 5` + guard + acceptance-gated `K` + skip-draft), re-measured on public
text with every measured prompt held out of the expert map. Two boots in one session, same image: a **public-only map**
(fitted from the CNN/DailyMail and ShareGPT training-slot counts alone, no private text), and the same map with the
**sidecar off** (`E3_ENABLE=0`, warm experts back on Grace), about an hour after the sidecar-on boot.

- **CNN/DM long:** CNN/DailyMail 3.0.0 test articles concatenated to `8,191` tokens, `1,024` forced output tokens.
- **ShareGPT short:** ShareGPT V3 first human turns, natural output (EOS honoured) up to `1,024` tokens.
- Both in the DeepSeek-V4-Pro chat encoding with thinking off and on, from a pool built with seed `20261009` from
  sha-pinned public sources. Every invocation has its own disjoint prompt slice and seed, and measured slots are disjoint
  from the slots the maps were fitted on. C1 = mean of three repetitions, C16 = one run, decode tok/s = C x 1000 / mean
  TPOT. Long C16 runs are KV-capped (about 11 requests resident on average, logged per run).

<!-- results -->
| Bench, decode tok/s | Public-only map, sidecar on: C1 / C16 | Sidecar off: C1 / C16 | Sidecar gain C1 / C16 | Receipt |
|---|---|---|---|---|
| CNN/DM long, thinking off | **94.92** / **282.24** | 73.55 / 183.13 | 1.291x / 1.541x | [on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log) |
| CNN/DM long, thinking on | **86.93** / **274.02** | 67.87 / 178.99 | 1.281x / 1.531x | [on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log) |
| ShareGPT short, thinking off | **93.88** / **288.08** | 74.35 / 179.31 | 1.263x / 1.607x | [on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log) |
| ShareGPT short, thinking on | **83.2** / **312.62** | 66.36 / 202.43 | 1.254x / 1.544x | [on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log) |
| Random ids (control), C1 | 38.28 | 35.62 | 1.075x | [on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log) |
<!-- /results -->

Quality and hygiene, public-only map: GSM8K (thinking on, first fifty rows) 98%; sidecar vs marlin minimum cosine
0.99999 (gate passed); 0 sidecar timeouts on both ranks; 0.0% prefix-cache hits on every public run. Receipts:
[on](../../logs/dsv4-pro-two-stations/v4p-public-map.log), [off](../../logs/dsv4-pro-two-stations/v4p-public-map-sidecar-off.log).

Reading the numbers:

- **Reproducible from public text.** The public-only map is fitted from route counts on public prompts only, and the
  prompts come from public datasets, so the table above can be reproduced without anything private. The map file is not
  shipped here: it is a function only of per-(layer, expert) routing counts on CNN/DailyMail and ShareGPT training
  slots (thinking off and on, equal weight), which are disjoint from the measured slots, so it can be rebuilt from public
  text.
- **The sidecar matters more on real text than on random ids.** On the public-only map the RTX PRO 6000 warm tier adds
  about a quarter for one user and about half or more at C16; random ids, which route almost uniformly, gain little.
  The sidecar-on and sidecar-off boots are separate, about an hour apart.
- C16 is a single run per boot, so treat differences of a few percent as noise.
- ShareGPT C1 varies widely between repetitions (each repetition is three different prompts, and drafter acceptance
  depends on the prompt); the per-repetition rows are in `results.jsonl`.

### Rebuild the public-only map

Scripts: [bench/public/](bench/public/) (pool builder, capture, bench) and `tools/mixfit.py`, `tools/refit.py`,
`tools/analyze_pc.py` (the fit). Run from this recipe folder on the rank-0 Station, with the launcher environment of
"How to run" below (`RANK0_IP`, `RANK1_IP`, `PEER_SSH`).

```bash
# 1. Prompt pool from public data at pinned revisions: abisee/cnn_dailymail @96df5e68 (3.0.0/test-00000-of-00001.parquet),
#    anon8231489123/ShareGPT_Vicuna_unfiltered @192ab218 (ShareGPT_V3_unfiltered_cleaned_split.json), and the model's own
#    tokenizer.json + encoding/encoding_dsv4.py. Seed 20261009; compare the output manifest.json (sha256 of every slot)
#    with bench/public/manifest.json.
W=/models/hf/deepseek-ai__DeepSeek-V4-Pro-0813
python3 bench/public/build_pool_v4p.py test-00000-of-00001.parquet ShareGPT_V3_unfiltered_cleaned_split.json \
  $W/tokenizer.json $W/encoding/encoding_dsv4.py ~/dsv4pro-public/pool

# 2. Capture route counts on the TRAINING slots: E3-A16 with tier counters on, no drafter. Any E2/E3 maps will do
#    (routing does not depend on placement; ours were older maps). Maps: E2 in ~/dsv4pro/prof, E3 in rowmaps/.
E3_MODE=a16 B12X=<b12x checkout with the patch> E3ROWMAP=rowmap-e3.json launch/e3-sidecar.sh 1   # on rank 1
E3_MODE=a16 B12X=<b12x checkout with the patch> E3ROWMAP=rowmap-e3.json launch/e3-sidecar.sh 0   # on rank 0
E3HOOK=1 E3_COUNT=1 E3ROWMAP=rowmap-e3.json PIN_MODE=split ROWMAP=rowmap-e2.json launch/e3-up.sh ~/bench/capture
bash bench/public/capture_public.sh ~/bench/capture 9600     # cnn-off, cnn-on, sg-off, sg-on -> snaps-capture/
launch/e3-down.sh sidecars

# 3. Fit: equal weight over the four public workloads -> rowmap-dsv4pro-v4ppub.json (E2) + rowmap-e3-on-e2-v4ppub.json (E3)
python3 tools/mixfit.py ~/bench/capture/snaps-capture ~/dsv4pro-public/mix --work cnn-off,cnn-on,sg-off,sg-on \
  --tag v4ppub --pcdir tools --old-e2 ~/dsv4pro/prof/rowmap-e2.json --old-e3 rowmaps/rowmap-e3.json
cp ~/dsv4pro-public/mix/rowmap-dsv4pro-v4ppub.json ~/dsv4pro/prof/; cp ~/dsv4pro-public/mix/rowmap-e3-on-e2-v4ppub.json rowmaps/

# 4. Boot on the new maps (sidecars with E3ROWMAP=rowmap-e3-on-e2-v4ppub.json, then e3-up.sh with
#    ROWMAP=rowmap-dsv4pro-v4ppub.json; E3_ENABLE=0 for the sidecar-off arm) and bench the HELD-OUT slots:
python3 bench/public/gauge.py http://127.0.0.1:8010 ~/bench/pub/gauge.tsv &     # measured concurrency (optional)
for ds in long short; do for md in off on; do
  GAUGE=~/bench/pub/gauge.tsv V4P=~/dsv4pro-public bash bench/public/pbench.sh ~/bench/pub/public/$ds-$md $ds $md
done; done
```

`pbench.sh` copies `$V4P/pool` and `$V4P/pylib` (pandas for the `vllm bench serve` client, which the image lacks) into
the container. Two limits: the counters in step 2 need the tier-counter version of `hook/e3_hook.py` (bucketed
`by_bucket` / `tiers` output), which is not in this release yet; and the drafter stack of our rows (DSpark with the
guard, acceptance-gated `K` and skip-draft) is not in this recipe's launcher, so step 4 here boots E3-A16 without a
drafter.

## Hardware and software

| Item | Value |
|---|---|
| Topology | both Stations: GB300 on each (one EP rank per Station), one 400G DAC (RoCEv2) between the ConnectX-8s; RTX PRO 6000 on each for E3 |
| Image | `vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b` = `vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063` (vLLM 0.30.1rc1.dev245, torch 2.13.0+cu130, FlashInfer 0.7.0) |
| Weights | `deepseek-ai/DeepSeek-V4-Pro-0813`, revision `72e1d323`, 66 shards, on local disk on both Stations |
| Fabric | Data Direct overlay + `tuner3.csv` + `NCCL_IB_TC=106` ([fabric recipe](../fabric-data-direct/)) |
| Sidecar | b12x master `2cc7f66a` (A8), or the same plus [patches/b12x-w4a16-skip-dropped-routes.patch](patches/b12x-w4a16-skip-dropped-routes.patch) (A16) |

## How to run

Prerequisites on both Stations: the weights at `/models/hf/deepseek-ai__DeepSeek-V4-Pro-0813`, the image, about
`420 GiB` of free host memory, and the fabric recipe's `~/hostrdma` and `~/nccl-tuner` staged. Set `RANK0_IP` and
`RANK1_IP` to the two rail addresses and `PEER_SSH=user@<rank-1 rail address>` on the rank-0 Station.

```bash
# E1, stock. Rank 1 first (headless), then rank 0 (serves the API on :8010 as deepseek-v4-pro).
launch/launch-dsv4pro.sh 1          # on the rank-1 Station (container name dsv4pro-e3 in every stage)
launch/launch-dsv4pro.sh 0          # on the rank-0 Station; ready in ~6 min warm, ~15 min the first time
# check both server logs for "Total CPU offloaded parameters: 200.81" and the Data Direct line, then:
tools/bench_e3.sh ~/bench/e1 1000   # random-id C1/C16 x3, fresh seeds, prefix-hit check (REALBENCH= for your own real-text bench)
python3 tools/gsm8k200_v4.py http://127.0.0.1:8010/v1/chat/completions deepseek-v4-pro gsm.jsonl 16
python3 tools/tf_flips.py gen http://127.0.0.1:8010 deepseek-v4-pro tf-ref-e1.json      # reference for the flip test
launch/stop-dsv4pro.sh              # rank 0 first, then rank 1; both GB300s must return to ~22 MiB

# E2: collect routing counts on E1 (EXTRA_ARGS=--enable-return-routed-experts), build a map, relaunch with the split.
ROUTES=gsm-routes.json python3 tools/gsm8k200_v4.py ... # or use calibration/routes-e1-gsm8k100.json (ours, GSM8K)
python3 tools/routes_calib.py http://127.0.0.1:8010/v1/chat/completions deepseek-v4-pro dom.json fit.json heldout.json
python3 tools/build_rowmap_e2.py ~/dsv4pro/prof/rowmap-e2.json dom.json calibration/routes-e1-gsm8k100.json
PIN_MODE=split ROWMAP=rowmap-e2.json launch/launch-dsv4pro.sh 1   # then 0 on the rank-0 Station

# E3: warm map from E2's cold rows, sidecars on both RTX PRO 6000s, then the server with the E3 hook.
python3 tools/build_rowmap_e3.py --e2-rowmap ~/dsv4pro/prof/rowmap-e2.json --domain dom.json \
  --gsm calibration/routes-e1-gsm8k100.json --capacity 2400 --out rowmaps/rowmap-e3.json
E3_MODE=a16 B12X=<b12x checkout with the patch> E3ROWMAP=rowmap-e3.json launch/e3-sidecar.sh 1   # on rank 1
E3_MODE=a16 B12X=<b12x checkout with the patch> E3ROWMAP=rowmap-e3.json launch/e3-sidecar.sh 0   # on rank 0
E3HOOK=1 E3_CHECK=3 E3_CHECK_MIN_T=1024 E3ROWMAP=rowmap-e3.json PIN_MODE=split ROWMAP=rowmap-e2.json \
  launch/e3-up.sh ~/bench/e3                           # gate boot: per-layer cosine on eager calls
python3 tools/cos_gate.py send http://127.0.0.1:8010 deepseek-v4-pro tools/gsm8k_test.jsonl 6
python3 tools/cos_gate.py parse ~/bench/e3/server-rank0.log <rank-1 server log> rowmaps/rowmap-e3.json
launch/e3-down.sh sidecars
```

`rowmaps/rowmap-e3-gsm-v1.json` is an example warm map built from GSM8K counts alone for the E1 placement; it was built
but not benchmarked. Name the routing-calibration prompt files `fit.json` and `heldout.json` (lists of
`{"id", "messages"}`): `routes_calib.py` keys each split by file name, and the map builders expect those two splits.
The routing counts are aggregate per-(layer, expert) histograms; no per-request routing is stored.

## Pitfalls we hit (each one cost a boot)

1. **`--cpu-offload-params experts.w13_weight experts.w2_weight` matches nothing** in this vLLM build: the DeepSeek-V4
   experts live under `...ffn.experts.routed_experts.*`, and offload matching works on whole name segments. The first
   boot offloaded 0 GiB. Use `routed_experts.w13_weight routed_experts.w2_weight` and check the
   `Total CPU offloaded parameters` line on both ranks.
2. **Pinned memory rounds up to a power of two** unless `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024`.
   With it, host pinned memory matched the offload size to within half a percent.
3. **Never combine `deep_gemm_mega_moe` with UVA offload**: it re-allocates the expert weights on the GPU. Use marlin.
4. **TP2 + EP2, not PP2.** vLLM does not split decode into micro-batches under pipeline parallelism; in a dress
   rehearsal with a smaller DeepSeek-V4 checkpoint, PP2 was about a quarter slower at C16.
5. **Teardown order:** stop rank 0 first, then rank 1, and confirm both GB300s are back to about 22 MiB before
   relaunching.
6. **b12x W4A16 returned NaN on DS-V4-Pro expert shapes** (hidden 7,168, intermediate 3,072, `fp4_e8m0_k32`): its plain
   top-k sum read rows for dropped (negative-id) routes that were never written. The patch skips them, as b12x's other
   sum paths already do. `tests/b12x_layer_check.py --mode a16` reproduces it on one RTX PRO 6000.
7. **DSpark multi-node** needed `--gpu-memory-utilization 0.99` and a `20,480`-token context to fit the drafter
   (about 20 GiB per rank), and then hung on thinking traffic (above).

## Files

| Path | What | Origin |
|---|---|---|
| `launch/launch-dsv4pro.sh` | one launcher for E1/E2/E3 (`PIN_MODE`, `E3HOOK` switches), Data Direct block, CDI GPU pinning | ours |
| `launch/e3-sidecar.sh`, `e3-up.sh`, `e3-down.sh`, `stop-dsv4pro.sh` | sidecar start, two-Station bring-up and teardown | ours |
| `hook/dsv4pro_pin_hook.py` | E2 hot/cold split on marlin with EP | port of J-M-Recipes (MIT) |
| `hook/e3_hook.py`, `hook/peer_tier_v4.py`, `sidecar/peer_server_v4.py`, `hook/sitecustomize.py` | E3 warm tier | derived from original-el8 (Apache-2.0) |
| `tools/build_rowmap_e2.py`, `tools/build_rowmap_e3.py`, `tools/coverage.py` | map builders and coverage | ours |
| `tools/routes_calib.py`, `tools/gsm8k200_v4.py`, `tools/tf_flips.py`, `tools/cos_gate.py`, `tools/coherence16.py`, `tools/bench_e3.sh` | calibration, quality, numerics gates, benchmark | ours |
| `calibration/routes-e1-gsm8k100.json` | E1 routing counts on GSM8K (first 100 rows, thinking on) | ours |
| `patches/b12x-w4a16-skip-dropped-routes.patch` | b12x W4A16 fix for E3-A16 | ours |
| `tests/` | CPU tests for the map builder and the shared-memory protocol; one-layer b12x check | ours |

Not included: the E2/E3 maps and routing counts we fitted to our private workload, and the real-text prompt set.
The 2026-10-09 public-only map is not included either; it is reproducible from public text (see the public-text results).
