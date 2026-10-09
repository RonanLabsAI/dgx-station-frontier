# Nemotron 3 Ultra 550B-A55B NVFP4 on one and two DGX Stations

NVIDIA's Nemotron 3 Ultra (550B total, 55B active, hybrid Mamba-Transformer MoE) as shipped in NVFP4, served with stock
vLLM on:

- **one Station:** TP1 on the GB300, with 100 GiB of routed-expert weights kept in Grace memory and read by the GPU over
  NVLink-C2C (UVA offload), marlin MoE kernel;
- **two Stations:** TP2 + EP2 across one 400G RoCE link, every weight in HBM, FlashInfer TRT-LLM NVFP4 MoE kernel.

No custom kernels, hooks or patches: two launch modes of one launcher, a hygiene-checked bench, and receipts.

## Models

| Checkpoint | Revision | Size | Licence |
|---|---|---|---|
| [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) (general Ultra) | `02462641` | `352.4 GB`, `113` shards | OpenMDW-1.1 |
| [`nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4) (the IOI model; "Nemotron-3-Ultra-CC" in NVIDIA's blog) | `3ae22c58` | `352.4 GB`, `108` shards | OpenMDW-1.1 |

Background: NVIDIA's [Fine-Tuning Nemotron for IOI and IMO](https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026)
blog. The competitive-coding checkpoint has the same architecture and NVFP4 layout as the general Ultra (its config
differs only in format), so it runs on the same recipe at the same speed; only quality differs. Neither checkpoint is
gated. Weights are not redistributed here.

**Which checkpoint is in which column.** The one-Station recipe was measured with both checkpoints. The two-Station
recipe was measured with the competitive-coding checkpoint only: it stands in for the general Ultra because the
architecture and weight bytes are identical and the one-Station runs show identical speed. The speedups below compare
the same checkpoint on both topologies.

## Results

Shape: 8,192 random-token prompt, 1,024 forced output tokens (`--ignore-eos`), `vllm bench serve`. Decode tok/s =
C x 1000 / mean TPOT, aggregate across streams. C1 is the mean of three repetitions with fresh prompts; C4, C16 and
C32 are one run of 8, 32 and 64 prompts. Prefill = input length / mean TTFT at C1. Prefix caching is off on the
server, so every run has zero cache hits by construction (the `/metrics` delta is still checked per run). MTP
speculative decoding is off in every column.

<!-- results -->
| tok/s | `Nemotron-3-Ultra`, one Station | competitive-coding, one Station | competitive-coding, **two Stations** | Receipt |
|---|---|---|---|---|
| Decode C1 per user (mean of three) | 39.71 | 39.78 | **99.14** | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Decode C4 aggregate | 67.82 | 79.41 | 287.77 | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Decode C16 aggregate | 154.92 | 118.07 | 685.52 | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Decode C32 aggregate | 155.22 | 146.13 | **889.14** (output throughput 750.22) | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Prefill, `8K`-token prompt, C1 | 2,866.6 | 2,864.0 | 13,112.9 | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Prefill, `64K`-token prompt, C1 | 2,904.6 | 2,904.9 | **13,779.7** | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
<!-- /results -->

<!-- results -->
| Two Stations / one Station, same checkpoint | Speedup | Receipt |
|---|---|---|
| Decode C1 per user | 2.49x | [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Decode C32 aggregate | 6.08x | [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Prefill, `64K`-token prompt | 4.74x | [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
<!-- /results -->

<!-- results -->
| Load and power | `Nemotron-3-Ultra`, one Station | competitive-coding, one Station | competitive-coding, two Stations | Receipt |
|---|---|---|---|---|
| Container launch to "Starting vLLM server", minutes | 6.73 | 4.83 | 20.23 (first boot of this MoE backend) | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| Weight load from local md RAID, seconds | 82.85 | 85.14 | 70.73 / 65.97 (per rank) | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| GB300 mean power during the C32 run, W | 490 | 492 | 676 / 682 (each GB300) | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| GB300 highest sample over all decode runs, W | 904 | 911 | 776 | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
| GB300 mean power during the `64K` prefill run, W | 562 | 560 | 513 / 520 | [Ultra](../../logs/nemotron-ultra/ultra-one-station.log), [CC one](../../logs/nemotron-ultra/cc-one-station.log), [CC two](../../logs/nemotron-ultra/cc-two-stations.log) |
<!-- /results -->

Power is `nvidia-smi` `power.draw` sampled every second, cut to each run's window (`bench/power_windows.py`); the full
per-run table for both Stations is in each log. The GB300 limit is `1,300` W. The RTX PRO 6000s are not used by this
recipe and stayed at idle draw. The two-Station first boot spent most of its 20 minutes in a one-time FlashInfer JIT
build of the TRT-LLM MoE module (`Initial profiling/warmup run took 941.97 s`); the cache directory is mounted, so
later boots should skip it, but we did not time one.

Reading the throughput table:

- **The two one-Station columns are the same architecture, recipe and weight bytes.** C1 and prefill are
  near-identical (39.71 vs 39.78 tok/s at C1); C4 to C32 differ by more (154.92 vs 118.07 tok/s at C16) because those runs mix new `8K`-token prefills
  (about three seconds each on one Station) into decode, and different random prompts route to different experts.
  Treat one-Station C16 to C32 as a band, not a point.
- **One Station saturates at C16.** C16 and C32 are flat for the general Ultra: decode is bound by reading the
  Grace-resident experts over C2C, which does not scale with batch the way HBM-resident weights do.

### Why two Stations are 2.5x faster at C1 and 6.1x at C32

Our reading of the logs, not a profiled breakdown:

1. **Every weight is in HBM.** On one Station the checkpoint does not fit: `206.82` GiB of weights are in HBM and
   `100.0` GiB of routed experts sit in Grace, read over NVLink-C2C on every decode step that touches them. The GB300
   reads pinned Grace memory at about [352.8](../../atlas/microbench/) GB/s, against several TB/s from its own HBM. With
   TP2 + EP2 each GB300 holds `155.15` GiB, entirely in HBM, and reads half of every layer.
2. **A faster MoE kernel.** UVA offload only works with the marlin MoE backend here, and marlin runs NVFP4 as
   weight-only FP4 (W4A16: the weights are dequantised and the math runs in 16-bit; vLLM warns about this in the
   one-Station log). Two Stations use FlashInfer's TRT-LLM NVFP4 MoE, which runs FP4 natively on the tensor cores.
   Together with twice the tensor cores, this is our explanation for the prefill gap (prefill is compute-bound):
   about 2.9K vs 13.1K-13.8K tok/s.
3. **Room to batch.** One Station has `15.23` GiB left for KV cache (`1,302,528` tokens) after weights; each
   two-Station rank has `75.94` GiB (`13,066,240` tokens). At C32 the one-Station arm is also bound by the C2C expert
   reads (point 1), so it stops scaling; the two-Station arm keeps scaling to C32 (889.14 tok/s, still rising).

What two Stations pay for it: one all-reduce per layer over the 400G link (see
[`recipes/fabric-data-direct`](../fabric-data-direct/)), and both GPUs.

## Quality

<!-- results -->
| Check | Result | Receipt |
|---|---|---|
| GSM8K, first `200` test rows, thinking on, greedy, `max_tokens 16,000`, competitive-coding checkpoint on one Station, % | 92.5 | [log](../../logs/nemotron-ultra/cc-one-station.log) |
| code20 pass@1: `20` LiveCodeBench v6 AtCoder problems, executed, competitive-coding checkpoint on two Stations | 11/20 (easy 6/6, medium 5/8, hard 0/6) | [log](../../logs/nemotron-ultra/cc-two-stations.log) |
<!-- /results -->

- **GSM8K is off-domain for this checkpoint.** It is a competitive-programming specialist. All 15 misses are rows that
  hit the 16,000-token cap while still reasoning; every row that finished was answered correctly. It is a coherence
  check of the serving path (marlin W4A16 plus Grace-resident experts), not a maths claim.
- **code20 is a short-budget sanity check, NOT comparable to NVIDIA's IOI 2026 result (535.4/600).** NVIDIA runs this
  model with up to 262K context and 32K-token answers inside a generate-verify-refine loop. We gave each problem one
  sample and a 24,000-token budget: 8 of the 9 failures (all 6 hard, 2 medium) hit that cap while still reasoning,
  before emitting any code; the other was a wrong answer. Problems are from AtCoder, January to March 2025, judged on
  up to 30 public and private tests each in a no-network container (`g++ -O2` or `python3`, 6 s per test); they
  predate the model, so contamination is possible. Problem ids and per-problem results are in the log.
- No quality run used the general Ultra checkpoint, and no run compared quality across topologies. For the same
  checkpoint, the one- and two-Station arms differ in numerics only through the MoE kernel (W4A16 vs native NVFP4).

## Memory plan

| Tier | One Station | Two Stations (each rank) |
|---|---|---|
| GB300 HBM (`250.7` GiB, `--gpu-memory-utilization 0.95`) | weights `206.82` GiB, KV `15.23` GiB, activations `4.54` GiB, graphs `0.35` GiB | weights `155.15` GiB, KV `75.94` GiB, activations `5.77` GiB, graphs `0.22` GiB |
| Grace (pinned, UVA) | routed experts `100.0` GiB (host Shmem +`100.1` GiB) | none |
| RTX PRO 6000 | not used | not used |

The checkpoint is about `307` GiB without the MTP layer. With `OFFGB=100` the KV cache gets `15.23` GiB
(`1,302,528` tokens): enough for 32 sequences of the bench shape, or about 17 at the full `73,728`-token context.

## Setup

```bash
MODEL_ROOT=/models/hf
hf download nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 --revision 02462641 \
  --local-dir $MODEL_ROOT/nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
hf download nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4 --revision 3ae22c58 \
  --local-dir $MODEL_ROOT/nvidia__NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4
docker pull vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063
# tag it as the nightly name the launcher expects, or pass IMAGE=<digest ref>
```

For two Stations the weights must be on both. Copying over the 400G link took about two minutes for `352.4 GB`. The
two-Station arm also needs the fabric pieces from [`recipes/fabric-data-direct`](../fabric-data-direct/) (host RDMA
libraries for the Data Direct overlay and the NCCL tuner plugin with `tuner3.csv`); `DD=0` runs without them, untested
for this model.

## Launch

```bash
# one Station (left over from a previous run? launch/stop-nemotron-ultra.sh first)
MODEL=ultra MODE=one launch/launch-nemotron-ultra.sh 0          # MODEL=cc for the competitive-coding checkpoint
grep "Total CPU offloaded parameters" ~/bench/nemotron-ultra-*/server-rank0.log   # must say 100.0

# two Stations: rank 1 first, then rank 0
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> MODEL=cc MODE=tp launch/launch-nemotron-ultra.sh 1   # on B
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> MODEL=cc MODE=tp launch/launch-nemotron-ultra.sh 0   # on A

launch/stop-nemotron-ultra.sh                                    # rank 0 first, then rank 1
```

OpenAI API on port 8017 as `nemotron-3-ultra`. Reasoning is on by default; per request
`{"chat_template_kwargs": {"enable_thinking": false}}` turns it off.

### Flags that matter

| Flag / env | Why |
|---|---|
| `--offload-backend uva --cpu-offload-gb 100 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight` | one Station only. In this vLLM build the experts are `layers.N.mixer.experts.routed_experts.*`; the shorter `experts.w13_weight` matches nothing and offloads nothing, silently |
| `--moe-backend marlin` (one Station) | the only MoE backend we found that leaves UVA-offloaded experts in Grace; others re-allocate them on the GPU |
| `--moe-backend flashinfer_trtllm` + `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=...` (two Stations) | native NVFP4 MoE; FlashInfer's autotuner is skipped for that op, as in our other two-Station recipes |
| `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024` | exact-size pinned host buffers for the offloaded experts |
| `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching` | NVIDIA's deployment settings for the competitive-coding checkpoint |
| `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` | from NVIDIA's deployment notes for this checkpoint |
| `MTP=0` | MTP keeps `1 + MTP` mamba state blocks per sequence (about `0.4` GB each in fp32), so `MTP=3` at 32 sequences costs about `52` GiB of HBM, which one Station does not have. NVIDIA's deployment uses MTP; we did not measure it (random-token prompts would make acceptance meaningless anyway) |
| `--enable-expert-parallel --disable-custom-all-reduce --nnodes 2` | two Stations: TP2 + EP2 over vLLM's multi-node `mp` backend |

## Benchmarks

- `bench/bench-nemotron-ultra.sh OUTDIR SEEDBASE` on the rank-0 Station runs the throughput table above: a fresh seed
  per run and per concurrency, its own warm-up seed and shape, and a `/metrics` prefix-hit check per run (any run
  above `2%` is flagged CONTAMINATED). Set `PEER=<user@second station>` to sample power on both Stations; then
  `bench/power_windows.py OUTDIR` after copying `power-right.csv` next to `power-left.csv`.
- `bench/gsm8k200.py <url> nemotron-3-ultra out.jsonl 32` reproduces the GSM8K row. It reads `gsm8k_test.jsonl` next to
  it: `curl -L -o bench/gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`.
- `bench/lcb_select.py test6.jsonl bench/code20.jsonl` picks the 20 problems from LiveCodeBench
  `code_generation_lite` `test6.jsonl` (revision `0fe84c39`; not vendored); private tests are decoded with an
  unpickler that refuses every global. Then `bench/code20.py <url> nemotron-3-ultra OUTDIR 20` generates and judges
  them in throwaway `--network none` containers (needs Docker on the host).

## Not included

- MTP speculative decoding, on any topology.
- The math (IMO) checkpoints: NVIDIA ships them in BF16 only (about `1.1 TB` each), and quantizing them ourselves is a
  separate project.
- An all-in-HBM run of the general Ultra on two Stations (see "Which checkpoint is in which column" above), and a
  heterogeneous GB300 + RTX PRO 6000 arm on one Station.
