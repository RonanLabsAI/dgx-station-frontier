# Results: GLM-5.3-Flash EXL3 4bpw on one DGX Station GB300 (2026-10-10)

Hardware: NVIDIA DGX Station GB300 (GB300, compute capability 10.3, 256,703 MiB; driver 595.99.02, CUDA 13.2; DGX OS,
aarch64). Software: TensorFold v0.6.0 (`c4646171`) + Mia's GLM-5.3-Flash recipe patches (`b5808ac`, 107) + 9001-9003,
on `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13.0a0, CUDA 13.3, NCCL 2.30.7, Triton 3.7.1). CUDA 13.3 ran on the 13.2
driver through forward compatibility. Checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `078455ff`;
drafter `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e`. Every LFS file sha256-checked.

## Numbers

| | Mia's defaults, drafts | Mia's defaults, no drafts | GB300-tuned, drafts |
|---|---|---|---|
| Decode, C1, mean of 3 runs | 160.4 (155.9 / 157.4 / 167.9) | 91.7 (91.9 / 91.6 / 91.7) | **173.1** (169.1 / 164.6 / 185.7) |
| C4 aggregate | 306.0 | 182.7 | 313.8 |
| C8 aggregate (64 prompts) | — | — | **403.8** |
| C1 decode after an 8,192-token prompt | 158.6 | 88.4 | 173.5 |
| Prefill 8K / 32K | 4,473 / 4,496 | 4,551 / 4,569 | 4,500 / 4,518 |
| GSM8K-50 | 49/50 | 49/50 | 49/50 |
| Load (incl. first JIT builds) | 503 s | 491 s | 485 s |

All in tok/s.
- **Mia's defaults** = her `config.sh`, changed only where one GPU requires it: `TP=1 COMM=nccl SPLIT=0`, KV pool
  40 GiB, stream smoothing off.
- **GB300-tuned** = `CONFIG=gb300` (9002 active, `--parallel 8`, `TF_GLM_L2PF=0 TF_GLM_SPLIT_TARGET=576
  TF_GLM_SHAPE_SK=0`).
- Raw summaries: `data/`.

## Method

- **Prompts:** public only.
  - Short: ShareGPT V3 first turns, 8-398 tokens.
  - Long: CNN/DailyMail 3.0.0 test articles joined to exactly 8,192 / 32,768 tokens, plus a summarise instruction.
  - Built by `bench/build_pool_glm.py`, seed 20261010; the manifest pins every file's sha256.
- **Hygiene:**
  - Each run uses its own prompt slot and seed. Warm-ups are separate runs on their own slots.
  - No two prompts share their first 64 tokens.
  - `usage.cached_tokens` was **0 on every request** (a run with >2% is marked contaminated and not reported).
- **Settings:** greedy, thinking off for the benchmark runs, streamed with usage, outputs capped at 1,024 tokens.
- **Definitions:**
  - Decode tok/s = (completion tokens - 1) / (last delta - first delta) per request (excludes prefill).
  - Aggregate = completion tokens / wall time.
  - Prefill = prompt tokens / time to first token (C1, 16-token reply).
- **Quality:**
  - GSM8K: first 50 test rows, thinking on, greedy, `#### <n>` scoring. The miss is row 12 in every configuration.
  - Smoke prompts (3 code, 2 tool calls; `data/*/smoke.txt`) were correct by reading; tool calls were valid JSON.
- **Repeatability:**
  - A same-day rerun of Mia's defaults gave C1 160.8 / C4 306.4 (<0.5% from the first run).
  - Drafted and undrafted replies were identical on every compared request (50 GSM8K, 31 bench).
  - 9002's replies were byte-identical to the run without it.

## What did not help

| Change | C1 decode | C4 |
|---|---|---|
| `DENSE=fp8` (vs q4) | 151.8 | 259.1 |
| `TF_GLM_EXL3_LOADS=nc4` (16-byte loads 4 steps ahead) | 171.8 | 304.3 |
| `TF_GLM_EXL3_DEC_ORDER=1` | 171.6 | 303.0 |

Also: `--parallel 16` is refused (GLM's maximum is 8).

## Profile (one 512-token decode, drafts off, nsys with CUDA-graph node tracing)

`data/profile-before.csv` is Mia's defaults; `data/profile-after.csv` is GB300-tuned.

| | Before | After |
|---|---|---|
| GPU kernel time for the request | 6,333 ms | 4,734 ms (-25%) |
| Decode, drafts off, under the profiler | 86.8 tok/s | 94.0 tok/s |
| EXL3 experts (per token) | 3.07 ms (`grouped_kernel`) | 3.03 ms (`dec_kernel`: 45 us gate/up + 27 us down a layer) |
| q4 dense matmuls (per token) | ~2.8 ms | ~2.5 ms |
| L2 prefetch kernel (per token) | 2.4 ms, mostly overlapped | off |

Reading:
- **EXL3 experts are ALU-bound on the GB300.** At one decode row they move ~1.2-1.5 TB/s, ~20% of HBM bandwidth, and more loads in flight did not help. The cost is the in-register trellis decode, which a GB10's memory bandwidth hides and a GB300's does not.
- **The host gap is now visible.** After the patches, kernel time per token is below wall time per token, leaving the GPU idle ~13% of a step.
- **Next steps are kernel work:**
  - a cheaper EXL3 decode for many-SM GPUs;
  - FP8/NVFP4 expert weights on the GB300's tensor cores;
  - fewer host round trips per step.

## Round 4: host gap (2026-10-11)

The profile above left the GPU idle ~13% of each step waiting on the host. This round tries three host-side knobs that
are already in Mia's recipe code, on the same image (`tensorfold-b1:v0.6.0-c4646171-mia-b5808ac-tp1-dec-ngc`, i.e. the
base plus 9001-9003) and the same hygiene (fresh slots and seeds, `cached_tokens` 0 on every request, one boot per arm).
Every arm runs the GB300-tuned config above (`--parallel 8`, `TF_GLM_L2PF=0 TF_GLM_SPLIT_TARGET=576
TF_GLM_SHAPE_SK=0`, DFlash2 drafts on) plus:

- **H0:** nothing else except `TF_GLM_MULTI_PROFILE=64` (a per-round time breakdown every 64 rounds; the control);
- **H1:** `TF_GLM_MULTI_LONE=1` (a lone stream runs on the one-stream graphs);
- **H2:** `TF_GLM_MULTI_ASYNC=1` (async messages);
- **H3:** both, plus `TF_GLM_MULTI_SAMPLER=packed`.

| Arm | C1 decode, mean of 3 | C1 output | C4 aggregate | C8 aggregate (64 prompts) | C1 after 8K | vs H0 (C1) |
|---|---|---|---|---|---|---|
| H0 | 172.7 (168.6 / 164.3 / 185.3) | 167.9 | 313.1 | 402.9 | 172.9 | — |
| H1 | 178.7 (174.4 / 170.0 / 191.7) | 173.5 | 313.9 | 404.0 | 176.7 | +3.5% |
| H2 | 175.1 (171.0 / 166.5 / 187.9) | 170.2 | 315.9 | 404.6 | 175.2 | +1.4% |
| H3 | **181.1** (177.0 / 172.3 / 194.2) | **175.8** | **316.2** | **405.8** | **178.5** | **+4.9%** |

All in tok/s. H0 reproduces the earlier GB300-tuned figure (172.7 vs 173.1). Prefill was flat across arms (8K
4,497-4,504; 32K 4,521-4,526 tok/s).

- **Replies are identical.** Compared with the earlier GB300-tuned run, every reply matched in all four arms: C1 5/5 in
  each of the three runs, C4 16/16, C8 64/64 (`data/host-gap-ab4.out`). These are speed knobs only. GSM8K was not rerun
  for this round; the identical replies stand in for it.
- **The gain is at C1.** H3 is +4.9% over H0; C4 and C8 move +1.0% and +0.7%. `CONFIG=gb300` now sets all three.
- **Where the time goes (H0's `TF_GLM_MULTI_PROFILE` lines, `data/host-gap-H0/multi-profile.txt`).** At C1 a
  round takes ~15.5 ms wall, of which ~14 ms is the `sample` bucket, and that bucket tracks the GPU's verify time almost
  exactly, so it is where the host waits on the GPU (an inference from the matching numbers). The rest (message,
  propose, verify launch, commit, taps, emit) is ~1.6 ms a round, about 10%. The one-stream graphs are 0.3-0.4 ms
  faster per verify window than the batched graphs at the same row count, which is what H1 picks up. At C8 the host
  share is ~4.6 of 49.6 ms (~9%), consistent with C4 and C8 barely moving.
- **What is left.** The experts are still ALU-bound (see the profile above); the host gap is now ~10% at C1.

Raw summaries for each arm: `data/host-gap-H0/` to `data/host-gap-H3/`.
