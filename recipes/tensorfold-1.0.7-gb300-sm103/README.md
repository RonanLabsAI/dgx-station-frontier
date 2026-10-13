# TensorFold 1.0.7 native on NVIDIA GB300 (sm_103): build recipe, checks and logs

From Stephen Ronan's DGX Stations, 2026-10-11 and 2026-10-12. Nemotron 3.5 Lightning 30B-A3B.

## Hardware and software

| | |
|---|---|
| Host | NVIDIA DGX Station GB300: Grace CPU (aarch64), DGX OS (Ubuntu 24.04), driver 595.99.02 (CUDA 13.2) |
| GPUs | GB300, compute capability 10.3, 256,703 MiB; RTX PRO 6000 Blackwell Max-Q Workstation Edition, 12.0, 97,887 MiB (same host) |
| Source | `ashhart/TensorFold` @ `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` (v1.0.7), `git archive` of the tag |
| Zig | 0.17.0 (`.zig-version`), `zig-aarch64-linux-0.17.0.tar.xz`, sha256 `9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8` |
| nvcc | 13.3, from `nvcr.io/nvidia/pytorch:26.07-py3` (the image your `zig/tests/cuda/box/run.sh` uses) |
| Release binary (6000 arm) | `tensorfold-1.0.7-linux-aarch64.tar.gz`, checked against its `.sha256` |
| Checkpoints | `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` @ `d9d758fb83953437f7263256b0d96157e2a348b8`; `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `bee7596271d1495f6992ae224aefde4410e816b8` |

## Build for sm_103

All builds run offline in the NGC container (`--network none`), as your box runner does. `$TF` holds `src/` (the tag),
`toolchain/zig` and `cache/`.

```sh
# kernels + tf-cuda-test (your box runner's command, with -Dsm=103)
docker run --rm --network none -v $TF:$TF -w $TF/src \
  -e ZIG_LOCAL_CACHE_DIR=$TF/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=$TF/cache/zig-global \
  --entrypoint $TF/toolchain/zig nvcr.io/nvidia/pytorch:26.07-py3 \
  build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=103 -Doptimize=safe --prefix $OUT/zig-out -j8 fatbins install

# the CUDA server, tensorfold-native (zig/build/cuda.zig: `zig build native`)
docker run --rm --network none -v $TF:$TF -w $TF/src \
  -e ZIG_LOCAL_CACHE_DIR=$TF/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=$TF/cache/zig-global \
  --entrypoint $TF/toolchain/zig nvcr.io/nvidia/pytorch:26.07-py3 \
  build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=103 -Doptimize=fast --prefix $OUT/zig-out -j8 native
```

- **Builds:** both compile cleanly (compiler warnings only). Every one of the 30 fatbins holds an sm_103 image
  (`cuobjdump --list-elf`), and `tensorfold-native` is 26.6 MB.
- **One trip-up worth a line in the runbook:** `fatbins install` also produces a `bin/tensorfold`, which is the
  developer CLI and rejects `serve --name`. The server is `zig build native` → `zig-out/native/bin/tensorfold-native`.

## Checks on the GB300

- **`tf-cuda-test` (our sm_103 `-Doptimize=safe` build):** `info`, `smoke`, `graph`, `launch-ex`, `ptx`, `symbols`,
  `overhead 1000 20`, `overhead 10000 10`, `overhead-pdl 1000 20`, `overhead-pdl 10000 10` and `cublaslt` pass. `nccl`
  did not run, because our container had no libnccl (single-GPU runs, so not needed). Logs: `logs/native-20261011-1836/GT/`.
- **The oracle / fixture tests** (`gdn-replay`, `gdn-tree`, `triton`) were not run. They need the Python oracle's
  fixtures.
- **Same output across architectures.** The MLX 4-bit checkpoint gave identical replies on your sm_120 release (RTX PRO
  6000) and our sm_103 build (GB300): 50/50 GSM8K replies character for character, and 15/15 single-stream bench
  replies with the same token counts.
- **Same output day to day.** The GB300 repeat on 10-12 gave identical replies to 10-11 for both checkpoints (50/50
  GSM8K, 15/15 bench), with speed within 0.1%.

## Serving and results

**Serve command:** your runbook flags. Each arm ran in a container with `--network none`, one GPU by UUID, and the
weights read-only:

```sh
tensorfold-native serve /model --name nemotron --host 127.0.0.1 --port 8080 \
  --parallel 1 --context 40960 --temperature 0 --no-thinking
```

**Bench:** greedy, with the MTP head's drafts on (the default). Public prompts: ShareGPT V3 first turns for short
prompts, and CNN/DailyMail 3.0.0 test articles joined to exactly 8,192 / 32,768 tokens for long ones. Each run has a
fresh seed and a separate warm-up, and there were 0 prefix-cache hits on every request. Decode tok/s = (completion
tokens - 1) / (last delta - first delta), per request. The client is `run-native.sh` + our `bench_tf.py` (logs hold
per-request JSON).

| Arm | Binary | Checkpoint | Decode, one user (mean of 3) | 4 queued (`--parallel 1`) | Prefill 8K / 32K | GSM8K-50 (thinking off) |
|---|---|---|---|---|---|---|
| RTX PRO 6000 | your release (sm_120) | MLX 4-bit | **557.7** | 508.0 | 19,511 / 17,472 | 37/50 |
| GB300 | our sm_103 build | MLX 4-bit | **534.8** (repeat 534.7) | 487.7 | 20,404 / 18,419 | 37/50 |
| GB300 | our sm_103 build | NVIDIA NVFP4 | **411.5** (repeat 411.4) | 408.0 | 19,513 / 17,748 | 41/50 |

**Notes from Ash Hart (TensorFold's author), 2026-10-12:**
- Greedy decoding with the MTP head's drafts on: the drafted output is identical, token for token, to plain decoding.
- The GB300 numbers come from our from-source sm_103 build. GB300 is not an official or tuned TensorFold platform.
- TensorFold 1.0.8 is out; single-user speed is unchanged from 1.0.7.

**Notes:**
- GSM8K misses are the same 13 items on both GPUs for MLX 4-bit, so they are the model at thinking off, not a GPU
  difference.
- NVFP4 / MLX = 0.77 on the GB300, matching your 1.3x-bytes figure.
- The GB300 is not tuned. These are your sm_12x tile choices compiled for sm_103.
- One user on a 3B-active model is step-latency-bound, so the GB300's extra bandwidth does not show at C1. We have not
  measured concurrency beyond 4 queued.

## Files

- `run-native.sh`: our driver. Internal service names and host details are replaced with placeholders.
- `logs/native-20261011-1836/`: the RTX PRO 6000 arm (N6) and the GB300 `tf-cuda-test` (GT).
- `logs/native-20261011-1945/`: GB300 MLX 4-bit (G4) and NVFP4 (GF).
- `logs/native-20261012-1704/`: the GB300 repeat.
- Each arm folder holds the server log, smoke replies, GSM8K replies, per-request bench JSON and summaries.
