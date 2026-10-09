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
| One Station, `max-num-seqs 32`, warm boot, first bench | 389.3 | 1,855.6 | 2,170.5 | 2,274.2 (capped: `32` sequences) | [log](../../logs/dsv41-flash-sidecar/one-station-c1-c64.log) |
| DP2, left copy (both benched simultaneously), `max-num-seqs 64` | 431.26 | 1,836.68 | 2,186.54 | 2,906.39 | [log](../../logs/dsv41-flash-sidecar/dp2-left.log) |
| DP2, right copy | 417.68 | 1,933.80 | 2,361.30 | 2,867.89 | [log](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| **DP2 total (sum of both copies)** | | **3,770** | **4,548** | **5,774** | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
| Same config, one copy alone (left / right), other Station idle | | 1,807.25 / 1,995.02 | 2,261.28 / 2,382.88 | 3,159.86 / 3,047.45 | [left](../../logs/dsv41-flash-sidecar/dp2-left.log), [right](../../logs/dsv41-flash-sidecar/dp2-right.log) |
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
- **One-Station C64 in the first row is capped** by `--max-num-seqs 32`. The DP2 rows use 64 sequences.
- The one-Station receipt predates our bench-hygiene rules; the audit found its decode windows clean (the only
  prefix-cache hits were warm-up copies hitting each other). Its prefill figures are **not** published here: see the
  withdrawn claims in the top-level README.

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
