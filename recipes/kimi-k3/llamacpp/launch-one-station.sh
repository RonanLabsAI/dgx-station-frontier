#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-one-station.sh -- Kimi K3 UD-IQ1_M (unsloth GGUF, 604 GiB) on ONE DGX Station GB300 with llama.cpp: attention,
# dense layers and the experts of layers 63-92 in HBM; the experts of layers 0-62 (NCPU) in pinned Grace memory, read in
# place by the GB300 over NVLink-C2C (GGML_CUDA_C2C_ZEROCOPY=1, needs patches/ggml-cuda-c2c-zerocopy.patch).
# ZC=0 runs the same build without zero-copy (stock behaviour: about 10.5 instead of 16.0 tok/s at C1).
#
# usage: [BIN=<llama.cpp build/bin>] [GGUF=<first shard>] [ZC=1] [NCPU=63] [NP=16] [PORT=8015] launch-one-station.sh [extra args]
#   The RTX PRO 6000 is not used. API: http://<host>:$PORT (llama-server; OpenAI routes under /v1), model kimi-k3.
#   Stop: docker rm -f kimi-k3
set -euo pipefail
BIN=${BIN:?BIN=<llama.cpp build/bin, built by build-llamacpp.sh>}
GGUF=${GGUF:-/models/gguf/unsloth__Kimi-K3-GGUF/UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf}
IMG=${IMG:-lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d}
ZC=${ZC:-1}; NCPU=${NCPU:-63}; NP=${NP:-16}; PORT=${PORT:-8015}
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
ENVZC=(); [ "$ZC" = 1 ] && ENVZC=(-e GGML_CUDA_C2C_ZEROCOPY=1)
docker rm -f kimi-k3 >/dev/null 2>&1 || true
docker run -d --name kimi-k3 --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1:-1 "${ENVZC[@]}" \
  -v "$(dirname "$GGUF")":/gguf:ro -v "$BIN":/llama:ro -e LD_LIBRARY_PATH=/llama:/usr/local/cuda/lib64 \
  --entrypoint /llama/llama-server "$IMG" -m "/gguf/$(basename "$GGUF")" --alias kimi-k3 --host 0.0.0.0 --port "$PORT" \
  -ngl 999 --n-cpu-moe "$NCPU" -fit off -fa on -t 64 --jinja --metrics --load-mode none \
  -c 65536 -np "$NP" -kvu -b 4096 -ub 2048 "$@" >/dev/null
echo "kimi-k3 starting (about 70 s from a warm page cache): docker logs -f kimi-k3"
