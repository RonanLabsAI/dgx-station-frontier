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
| *Experimental, not servable:* E3-A16 + DSpark drafter, `K = 5` | 21.62 | 72.11 | 77.56 | 270.86 | engine hang: 200 of 200 requests errored | | [log](../../logs/dsv4-pro-two-stations/dspark-experimental.log) |
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
- **DSpark is experimental.** The bundled drafter composes with TP2 + EP2 multi-node, the E2/E3 hooks and CUDA graphs,
  and doubles real-text C1 (acceptance about 2.9 tokens per step). On random ids it predicts almost nothing and halves
  throughput. It also **hung** on GSM8K thinking traffic at C16: the two ranks desynchronised, all 200 requests failed,
  so it has no quality receipt and is not servable. Our diagnosis so far (not yet validated on hardware): an
  out-of-memory error while JIT-loading a kernel on the non-output rank was swallowed, so that rank skipped the
  drafter's collectives.

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
