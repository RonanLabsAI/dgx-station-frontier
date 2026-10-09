#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-rpc.sh two|four -- Kimi K3 UD-IQ1_M with llama.cpp RPC, llama-server on THIS Station's GB300 (start the
# rpc-server.sh processes first). Layer split (-sm layer): each device holds a contiguous block of layers, so one token
# passes through the devices in turn.
#
#   two:  this GB300 + the second Station's GB300 (RPC). -ts 34,59 is in llama.cpp's default device order for this run,
#         RPC0 (the second Station's GB300) then CUDA0 (this GB300): 34 shares there, 59 here. The experts of the last
#         NCPU=27 layers (on this Station) stay in its Grace memory, read in place over NVLink-C2C (C2C zero-copy patch).
#   four: both GB300s + both RTX PRO 6000s, whole model in GPU memory, no Grace. Device order (pipeline order):
#         CUDA0 this GB300, RPC0 this Station's 6000 (127.0.0.1:50053), RPC1 the second Station's 6000, RPC2 its GB300.
#         -ts 36,11,11,35 puts 225.2 / 72.8 / 75.5 / 229.6 GiB of the 604.3 GiB on them. Needs a 103-real;120a-real build.
#
# usage: PEER_IP=<rail ip of the second Station> [BIN=..] [GGUF=..] [TS=..] [NCPU=27] [PORT=8015] launch-rpc.sh two|four [extra]
#   Stop: docker rm -f kimi-k3 here, then the k3-rpc-* containers.
set -euo pipefail
MODE=${1:?two|four}; shift
BIN=${BIN:?BIN=<llama.cpp build/bin>}; PEER=${PEER_IP:?PEER_IP=<rail ip of the second Station>}
GGUF=${GGUF:-/models/gguf/unsloth__Kimi-K3-GGUF/UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf}
IMG=${IMG:-lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d}
PORT=${PORT:-8015}; L=93   # Kimi K3 has 93 blocks
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
EXTRA=()
if [ "$MODE" = two ]; then
  TS=${TS:-34,59}; NCPU=${NCPU:-27}; RPC=$PEER:50052
  FIRST=$((L - NCPU)); RX=$(seq -s '|' "$FIRST" $((L - 1)))
  [ "$NCPU" -gt 0 ] && EXTRA=(-ot "blk\.($RX)\.ffn_(up|gate|down)_exps\.weight=CPU")
  DEV=()
elif [ "$MODE" = four ]; then
  TS=${TS:-36,11,11,35}; RPC=127.0.0.1:50053,$PEER:50053,$PEER:50052
  DEV=(-dev CUDA0,RPC0,RPC1,RPC2 -sm layer)
else echo "MODE must be two|four"; exit 2; fi
RDMA=(); [ -e "/dev/infiniband/${UVERBS:-uverbs1}" ] && RDMA=(--device "/dev/infiniband/${UVERBS:-uverbs1}")
docker rm -f kimi-k3 >/dev/null 2>&1 || true
docker run -d --name kimi-k3 --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1:-1 "${RDMA[@]}" \
  -e GGML_CUDA_C2C_ZEROCOPY=1 -v "$(dirname "$GGUF")":/gguf:ro -v "$BIN":/llama:ro -e LD_LIBRARY_PATH=/llama:/usr/local/cuda/lib64 \
  --entrypoint /llama/llama-server "$IMG" -m "/gguf/$(basename "$GGUF")" --alias kimi-k3 --host 0.0.0.0 --port "$PORT" \
  --rpc "$RPC" "${DEV[@]}" -ngl 999 -ts "$TS" -fit off "${EXTRA[@]}" \
  --load-mode none -fa on -t 64 --jinja --metrics -c 65536 -np 16 -kvu -b 4096 -ub 2048 "$@" >/dev/null
echo "kimi-k3 ($MODE) starting: docker logs -f kimi-k3 (loads in about 7.5 min for two, 11.5 min for four)"
