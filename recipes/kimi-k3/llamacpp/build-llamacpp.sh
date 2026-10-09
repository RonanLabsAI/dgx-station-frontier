#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# build-llamacpp.sh SRC [ARCHS] -- build llama.cpp (with patches/ggml-cuda-c2c-zerocopy.patch applied) for DGX Station.
#
# DO NOT BUILD WITH CUDA 13.2.0 / 13.2.1 (nvcc 13.2.51 or 13.2.78): their NVVM (cicc) miscompiles the IQ1_S / IQ2_S /
# IQ3_S kernels and Kimi K3 UD-IQ1_M then produces garbage (wikitext-2 perplexity about 300 instead of 2.04). This
# script builds inside the SGLang dev image, which ships CUDA 13.0.88. nvcc >= 13.2.86 is also fine. See ../README.md.
#
# ARCHS: 103-real for the GB300 only (one or two Stations); 103-real;120a-real to also run on the RTX PRO 6000 (four GPUs).
# The build runs with --network none and no GPU. Afterwards it runs the IQ guard below if a GB300 is visible.
set -euo pipefail
SRC=${1:?llama.cpp source tree (patch applied)}; ARCHS=${2:-103-real;120a-real}
IMG=${IMG:-lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d}
SRC=$(cd "$SRC" && pwd)
grep -q GGML_CUDA_C2C_ZEROCOPY "$SRC/ggml/src/ggml-cuda/ggml-cuda.cu" || echo "WARN: C2C zero-copy patch not applied in $SRC"
docker run --rm --network none -v "$SRC":/src --entrypoint bash "$IMG" -c "
  set -e; cd /src; nvcc --version | tail -2
  case \$(nvcc --version | grep -o 'V13\.2\.[0-9]*' || true) in V13.2.51|V13.2.78) echo 'REFUSING: nvcc 13.2.51/13.2.78 miscompiles IQ kernels'; exit 9;; esac
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='$ARCHS' -DGGML_NATIVE=ON \
        -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
  cmake --build build -j 32 --target llama-server ggml-rpc-server llama-perplexity test-backend-ops"
echo "built: $SRC/build/bin"
# Guard (a few seconds): the IQ*_S matmuls must pass on the GPU before this build is trusted.
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null | awk -F", " '/GB300/{print $1;exit}' || true)
if [ -n "$GB300" ]; then
  docker run --rm --device "nvidia.com/gpu=$GB300" -v "$SRC/build/bin":/llama:ro -e LD_LIBRARY_PATH=/llama:/usr/local/cuda/lib64 \
    --entrypoint /llama/test-backend-ops "$IMG" test -b CUDA0 -o MUL_MAT -p iq3_s | tail -3
fi
