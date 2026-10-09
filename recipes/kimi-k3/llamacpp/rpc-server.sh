#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# rpc-server.sh KIND BIND_IP PORT -- one ggml-rpc-server on ONE GPU of this Station (KIND = gb300 | rtx).
#
# One GPU per process: on our Stations a CUDA process that is given BOTH the GB300 and the RTX PRO 6000 can use only one
# of them (llama.cpp: "ggml_cuda_init: failed to initialize CUDA: system not yet initialized"; torch: device_count() 2 but
# "invalid device ordinal" on cuda:1). So every GPU gets its own container and the main llama-server talks to the others
# over RPC, even on the same Station (bind 127.0.0.1 for that one).
#
# usage: [BIN=<llama.cpp build/bin>] rpc-server.sh gb300|rtx <bind ip> <port>      stop: docker rm -f k3-rpc-<kind>
#   two Stations: rpc-server.sh gb300 <rail ip of this Station> 50052 on the second Station.
#   four GPUs:    second Station: gb300 <rail ip> 50052 and rtx <rail ip> 50053; first Station: rtx 127.0.0.1 50053.
set -euo pipefail
KIND=${1:?gb300|rtx}; IP=${2:?bind ip}; PORT=${3:?port}
BIN=${BIN:?BIN=<llama.cpp build/bin (103-real;120a-real for rtx)>}
IMG=${IMG:-lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d}
PAT=$([ "$KIND" = gb300 ] && echo GB300 || echo "RTX PRO 6000"); N=k3-rpc-$KIND
G=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " -v p="$PAT" '$2 ~ p {print $1;exit}')
u=$(nvidia-smi --id="$G" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$u" -lt 1024 ] || { echo "GPU $KIND busy (${u} MiB)"; exit 4; }
RDMA=(); [ -e "/dev/infiniband/${UVERBS:-uverbs1}" ] && RDMA=(--device "/dev/infiniband/${UVERBS:-uverbs1}")   # RPC over RDMA when available
docker rm -f "$N" >/dev/null 2>&1 || true
docker run -d --name "$N" --device "nvidia.com/gpu=$G" --ipc host --network host --ulimit memlock=-1:-1 "${RDMA[@]}" \
  -v "$BIN":/llama:ro -e LD_LIBRARY_PATH=/llama:/usr/local/cuda/lib64 \
  --entrypoint /llama/ggml-rpc-server "$IMG" -H "$IP" -p "$PORT" >/dev/null
echo "$N up on $IP:$PORT"
