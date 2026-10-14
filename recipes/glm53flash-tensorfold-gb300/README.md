# GLM-5.3-Flash on one NVIDIA DGX Station GB300 with TensorFold

Mia's GLM-5.3-Flash EXL3 recipe (built for 2-4 DGX Sparks) on Ash Hart's TensorFold engine, served from **one GB300**
(sm_103, 252 GB HBM3e) at TP1. Measured 2026-10-10 on a DGX Station GB300.

| | Mia's defaults (`CONFIG=mia`) | Tuned for GB300 (`CONFIG=gb300`) |
|---|---|---|
| Decode, one request (mean of 3 runs, DFlash2 drafts) | 160.4 tok/s | **181.1 tok/s** (173.1 before the host-gap settings) |
| Decode, one request, no drafts | 91.7 tok/s | — |
| 4 requests at once, aggregate | 306 tok/s | 316 tok/s |
| 8 requests at once, aggregate | — | **406 tok/s** |
| Prefill, 8K / 32K prompt | ~4,470 / ~4,500 tok/s | ~4,502 / ~4,524 tok/s |
| GSM8K-50 (thinking on, greedy) | 49/50 | 49/50 |
| Window / cache pool | 1,048,576 / 6.6M tokens | same |

The `CONFIG=gb300` preset now includes the host-gap settings (`TF_GLM_MULTI_LONE=1 TF_GLM_MULTI_ASYNC=1
TF_GLM_MULTI_SAMPLER=packed`, round 4 in the results): C1 decode 172.7 -> 181.1 tok/s (+4.9%), replies byte-identical.
GSM8K-50 (49/50) was measured before these settings; replies were identical afterwards. 173.1 is the figure without them.

Details, method and logs: [results/RESULTS.md](results/RESULTS.md).

## What it took

TensorFold's GLM engine only served 2, 3 or 4 ranks (one GPU each). Every kernel compiled for sm_103 unchanged. Three
small patches, applied after Mia's 107:

| Patch | Change | Effect |
|---|---|---|
| `9001-glm-tp1.patch` | `CUDA_TP` and `tp.SIZES` include 1; `--master` defaults to 127.0.0.1 at TP1 | serves on one GPU |
| `9002-glm-exl3-dec-tp1.patch` | `DEC_WIDTHS` += 2048 and a `dec_launch<32>`, so one rank's whole 2048-wide experts take the fused decode kernel | +2.1% decode; replies byte-identical (50/50 GSM8K, 34/34 bench replies) |
| `9003-glm-split-target-env.patch` | `TF_GLM_SPLIT_TARGET` (default 192 = 4 waves of GB10's 48 SMs) and `TF_GLM_SHAPE_SK=0`, both env knobs; defaults unchanged | 576 with the table off: +3.2% decode on GB300 |

`CONFIG=gb300` also sets the three host-gap settings and `TF_GLM_L2PF=0` (Mia's L2 prefetch overlaps the cross-Spark all-gathers; at TP1 there are
none: +2.4%) and `--parallel 8` (GLM's maximum).

## Run it

Needs: a GB300 (or another discrete NVIDIA GPU with ≥ 200 GB), docker with the NVIDIA runtime, ~200 GB disk, and
python3 with `huggingface_hub pandas tokenizers` for `prepare.sh`.

```bash
MODELS_DIR=/data/models ./prepare.sh            # pins + sha256-verifies inputs, builds the image (network: here only)
MODELS_DIR=/data/models CONFIG=gb300 ./run.sh   # serve (--network none) + smoke + GSM8K-50 + bench, then tear down
```

- The first start compiles TensorFold's CUDA extensions for your GPU (~8 min load in total on a GB300).
- `BASE=vllm VLLM_BASE_IMAGE=...` builds on a CUDA 13.0 image instead of NGC 26.07 (CUDA 13.3). Use it where the
  driver is older than 13.3 and the GPU does not support CUDA forward compatibility.
- Serve only (no bench): `./run.sh preflight env serve`. The server listens on 127.0.0.1:8888 *inside* the container
  (OpenAI and Anthropic APIs). To expose it, change `--network none` in `run.sh` to a published port.

## What limits it on GB300 (profiled)

At one decode row, the EXL3 expert kernels move ~1.2-1.5 TB/s, about 20% of HBM bandwidth. They are bound by
trellis-decode arithmetic, not memory as on a GB10. The host leaves the GPU idle ~13% of each step. More speed needs
kernel work: a cheaper EXL3 decode, or expert weights on the GB300's FP8/FP4 tensor cores. See results.

## Credits

Ash Hart (TensorFold), Mia / MiaAI-Lab and the recipe's contributors (patches, checkpoint, calibration), Z.AI
(GLM-5.3-Flash), turboderp (EXL3), incoai (DFlash2). Licenses in [NOTICE](NOTICE).
