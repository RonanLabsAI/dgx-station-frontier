# TensorFold native 1.0.7 on a DGX Station GB300 (sm_103)

Ash Hart's [TensorFold](https://github.com/ashhart/TensorFold) 1.0.7 is the native engine. On NVIDIA it serves
Nemotron 3.5 Lightning 30B-A3B and is qualified on GB10, RTX 3090 and RTX PRO 6000. It ships no GB300 (sm_103)
build, so we built one from source and ran it on the GB300 of one DGX Station, next to Ash's own release binary on
the same Station's RTX PRO 6000, with the same harness. Measured 2026-10-11 PT.

<!-- results -->
| Arm | Decode, one request, tok/s | Queued requests, aggregate tok/s | Decode after a long prompt, tok/s | Prefill, long / longest prompt, tok/s | Receipt |
|---|---|---|---|---|---|
| RTX PRO 6000, Ash's release binary, MLX int4 | 557.7 | 508.0 | 539.3 | 19,511 / 17,472 | [log](../../logs/tensorfold-native-gb300/native-1.0.7.log) |
| GB300, our sm_103 build, MLX int4 | 534.8 | 487.7 | 509.7 | 20,404 / 18,419 | [log](../../logs/tensorfold-native-gb300/native-1.0.7.log) |
| GB300, our sm_103 build, NVIDIA NVFP4 | 411.5 | 408.0 | 433.4 | 19,513 / 17,748 | [log](../../logs/tensorfold-native-gb300/native-1.0.7.log) |
<!-- /results -->

One request = mean of three runs; queued = four requests against `--parallel 1`; long prompt = 8K tokens, longest = 32K tokens.
MLX int4 is the `MLX-4bit` checkpoint.

- Same model, two GPUs: the GB300 decodes a little slower than the RTX PRO 6000 and prefills a little faster. NVFP4
  decodes slower than MLX 4-bit on the GB300.
- Ash's own GPU test suite (`tf-cuda-test`) passes on the GB300 with our build: info, smoke, graph, launch-ex, ptx,
  symbols, overhead, overhead-pdl and cuBLASLt (20 PASS lines in the receipt). Its nccl check could not run: that
  container had no libnccl.
- Smoke check (3 code + 2 tool prompts): code answers complete in all three arms; tool calls 1 of 2 on MLX 4-bit
  (both GPUs), 2 of 2 on NVFP4.
- Prompts are prose with thinking off; prefix-cache hits were 0 on every request. One request at a time per GPU
  (`--parallel 1`), so the 4-request row is queued, not batched.

## Build (sm_103)

Source 1.0.7 at `e7c523eb`, zig 0.17.0 (official aarch64 tarball), in NGC PyTorch 26.07 for its CUDA toolkit. The
server is the `native` build step; the default step builds only the CLI and the test binary.

```bash
TF=$HOME/tensorfold/native-1.0.7          # src/ = TensorFold source, toolchain/zig = zig 0.17.0
docker run --rm --network none --cpus 8 --memory 16g -v $TF:$TF -w $TF/src \
  -e ZIG_LOCAL_CACHE_DIR=$TF/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=$TF/cache/zig-global \
  --entrypoint $TF/toolchain/zig nvcr.io/nvidia/pytorch:26.07-py3 \
  build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=103 -Doptimize=fast --prefix $TF/build-sm103 -j8 native
ls $TF/build-sm103/native/bin             # tensorfold-native
```

`-Doptimize=fast` is Ash's default for serving; `safe` (his test-harness mode) is slower.

## Serve (Ash's runbook command)

```bash
tensorfold-native serve /model --name nemotron --host 127.0.0.1 --port 8080 \
  --parallel 1 --context 40960 --temperature 0 --no-thinking
```

One GPU per container (`--gpus device=<uuid>`), weights mounted read-only, `--network none`. Checkpoints:
`TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` @d9d758fb and
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @bee75962.

Credits: TensorFold and its test suite are Ash Hart's; the MLX 4-bit checkpoint is TensorFold's; the NVFP4
checkpoint is NVIDIA's. No TensorFold code is copied into this repository.
