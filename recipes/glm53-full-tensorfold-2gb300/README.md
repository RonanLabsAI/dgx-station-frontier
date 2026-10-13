# Full GLM-5.3 on two NVIDIA DGX Station GB300 with TensorFold

Mia's full GLM-5.3 EXL3 recipe (built for 3-4 DGX Sparks) on Ash Hart's TensorFold engine, served across **two GB300**
Stations (sm_103) at TP2 over a 400G RoCE link. Measured 2026-10-11 on two DGX Station GB300.
The model is EXL3 2.75 bpw (routed experts EXL3, the rest BF16): 272.7 GiB of weights, about 136 GiB a rank. It does
not fit one GB300, and TensorFold has no weight offload, so two Stations is the smallest layout that works here.

| | Two GB300, TP2, DSpark drafts |
|---|---|
| Decode, one request (3 runs: 71.2 / 74.1 / 74.1) | **73.2 tok/s** (mean) |
| Decode after an 8K prompt | 72.7 tok/s |
| Prefill, 8K / 32K prompt | ~2,180-2,210 / ~2,093-2,096 tok/s |
| Time to first token, 8K / 32K prompt | 3.7 s / 15.6 s |
| GSM8K-50 (thinking on, greedy) | 49/50 |
| Load (healthy after launching both ranks) | 316 s |
| GPU memory in use after load | 141,480 MiB (138.2 GiB) on each GB300 |
| Window / parallel | 163,840 tokens / 1 request |

Concurrency above one request was not measured. Details, method and logs: [results/data/](results/data/) (attempt 3
only; see "Attempts" below).

## What it took

Mia's full recipe sets `CUDA_TP = (3, 4)` in TensorFold's `glm_moe_dsa` family, because ~129 GiB a rank does not fit a
128 GB Spark. One small patch widens it; everything else is her 110 patches unchanged.

| Patch | Change |
|---|---|
| `patches/9101-glm-full-tp2.patch` | `CUDA_TP = (2, 3, 4)`: lets the engine serve two ranks (two GB300, ~136 GiB a rank) |

Settings that differ from Mia's defaults (set in `run-b3.sh`):
- `TP=2`, `COMM=nccl`, stream smoothing off.
- Context parallelism off (`CP=0`) and no disk prompt cache (`DISK_CACHE=0`). Her TP>2 defaults turn on context
  parallelism, an FP4 KV cache, a 500K window and a 64 GiB disk cache that is not mounted here. The resulting window is
  163,840 tokens.
- Drafter: `RedHatAI/GLM-5.3-speculator.dspark` @ `b374b956` (her default for this recipe), `--parallel 1`.
- Host networking so NCCL can use the 400G rail (NCCL over `mlx5_1`, RoCE, GPUDirect RDMA); only
  `/dev/infiniband/uverbs1` is passed into the containers.

## Run it

Inputs, pinned:
- TensorFold v0.6.0 `c4646171`.
- Mia's full recipe `eb6414b` ([MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)):
  the 110 files in `patches/` go to `patches/mia`, and `scripts/config.sh` to `mia-scripts/`.
- Checkpoint `Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` @ `2d747d0e` (292.8 GB) on both Stations, sha256-verified.
- Drafter `RedHatAI/GLM-5.3-speculator.dspark` @ `b374b956`.
- The bench client, smoke and GSM8K scripts and prompt pool from [../glm53flash-tensorfold-gb300/](../glm53flash-tensorfold-gb300/)
  (`prepare.sh` builds the pool), placed in `b1/` of the build context.

The image follows [../glm53flash-tensorfold-gb300/Dockerfile](../glm53flash-tensorfold-gb300/Dockerfile) (TensorFold
`c4646171` in NGC PyTorch 26.07), with Mia's full-recipe patch set (`eb6414b`) and `9101` in place of the Flash patch set.
Build it on both Stations and tag it `tensorfold-b3:v0.6.0-c4646171-miafull-eb6414b-9101-ngc` (the tag `run-b3.sh` uses).

```bash

# from a third machine with ssh to both Stations; both GB300s must be idle
LEFT_HOST=user@left RIGHT_HOST=user@right RAIL_IP_A=<rank-0 rail address> RAIL_IP_B=<rank-1 rail address> \
RUNS=/path/for/outputs ./run-b3.sh
```

`run-b3.sh` checks both GPUs are empty and both checkpoints are verified, computes Mia's env inside the image, starts
rank 1 then rank 0 (TensorFold's rule), waits for health, then runs the smoke test, GSM8K-50 and the benchmark inside
rank 0's container and pulls the logs back. The interface and HCA names in it (`enP1p3s0f1np1`, `mlx5_1`, `uverbs1`) are
the 400G port on our Stations. Expect 45-75 minutes: two loads of ~136 GiB a rank, the first-start CUDA extension
builds, and the bench. The server listens on 127.0.0.1 of the rank-0 Station, port 8898 by default (`B3_PORT`).

## Method

Same as the one-GB300 recipe: public prompts only (ShareGPT V3 first turns, CNN/DailyMail articles joined to exactly
8,192 / 32,768 tokens), each run on its own prompt slot and seed with separate warm-ups, `usage.cached_tokens` 0 on
every request (max prefix-cache hit 0.0). Greedy, thinking off for the benchmark, outputs capped at 1,024 tokens;
decode tok/s excludes prefill. GSM8K: first 50 test rows, thinking on, greedy, one request at a time, 0 runaways, 0
errors. Smoke prompts (3 code, 2 tool calls) are in `results/data/smoke.txt`.

## Attempts

This is the third attempt. The first two failed on our launch script and are not TensorFold results:
1. The container did not start. `run-b3.sh` passed `--device /dev/infiniband/rdma_cm`, which does not exist on our
   Stations (the `rdma_ucm` module is not loaded). NCCL over RoCE uses `uverbs*`, so the device was dropped.
2. Rank 0 served on port 8888, which another service already running on the host was using, so the health check
   answered for that service and TensorFold never loaded. Its smoke, GSM8K and bench numbers belong to that other
   service and are discarded. The script now uses its own port and refuses to start if the port is already bound.

The numbers above are from attempt 3 only.

## Limits

- One recipe, one drafter, one run of each prompt set. C1 decode across the three runs ranged 71.2-74.1.
- No GB300-specific tuning was done for this model (the one-GB300 Flash recipe has some; none is applied here).
- Only one request at a time was measured, and the longest prompt was 32K tokens.
- No comparison with other engines or quantizations was made in this recipe.

## Credits

Ash Hart (TensorFold), Mia / MiaAI-Lab and the recipe's contributors (the full GLM-5.3 patches, checkpoint,
calibration), Z.AI (GLM-5.3), turboderp (EXL3), Red Hat AI (the DSpark drafter). Licenses in [NOTICE](NOTICE). See also
[../glm53flash-tensorfold-gb300/](../glm53flash-tensorfold-gb300/) for the one-GB300 Flash version.
