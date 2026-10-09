#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-k3.sh -- Kimi K3 (vellum-ai/Kimi-K3-W3A16-g64: GPTQ int3 g64 routed experts, BF16 everything else) across TWO
# DGX Station GB300s with vLLM nightly af7f9488: TP2 + EP2 over one RoCE link (vLLM mp multi-node), Humming 3-bit WNA16
# MoE backend, and OFFGB GiB per rank of routed-expert weights in Grace memory, read by the GB300 over NVLink-C2C (UVA).
#
# What it needs beyond stock vLLM (see ../README.md):
#   - preprocessor_config.json from moonshotai/Kimi-K3 copied into the vellum directory (vellum omits it; this vLLM
#     build raises "Can't load image processor" without it, even with --language-model-only);
#   - the load hook in ./hook (HOOKDIR): 3-bit repack + re-offload after Humming renames the expert params;
#   - the offload names routed_experts.{w13,w2}_weight_packed. The shorter routed_experts.w13_weight (right for MXFP4
#     checkpoints) matches nothing on a WNA16 checkpoint and silently offloads nothing; the log must say
#     "Total CPU offloaded parameters: 408.27" on both ranks.
# The fabric block (Data Direct overlay + NCCL tuner + traffic class) is the one from recipes/fabric-data-direct; DD=0
# skips it (untested for this model).
#
# usage: RANK0_IP=<rail ip of rank 0> RANK1_IP=<rail ip of rank 1> [IFN=<rail netdev>] [HCA=<rdma dev>] [UVERBS=uverbs1]
#        [MODEL=/models/hf/vellum-ai__Kimi-K3-W3A16-g64] [SHARDS=275] [OFFGB=408] [CTX=65536] [SEQS=16] [MBT=8192]
#        [UTIL=0.95] [PORT=8010] [HOOKDIR=<dir>] [LOGDIR=..] [EXTRA_ARGS=".."] launch-k3.sh RANK
#   Start RANK 1 on the second Station first, then RANK 0. Ready in about 17 minutes from a warm page cache.
#   API on rank 0: http://<host>:$PORT/v1, model kimi-k3. Teardown: stop-k3.sh on rank 0 FIRST, then rank 1.
set -euo pipefail
RANK=${1:?rank 0|1}; N=${NAME:-k3}
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}; DD_IMAGE=$IMAGE
MODEL=${MODEL:-/models/hf/vellum-ai__Kimi-K3-W3A16-g64}; SHARDS=${SHARDS:-275}
SERVED=${SERVED:-kimi-k3}; PORT=${PORT:-8010}; MPORT=${MPORT:-29614}
OFFGB=${OFFGB:-408}; MOE=${MOE:-humming}
CTX=${CTX:-65536}; SEQS=${SEQS:-16}; MBT=${MBT:-8192}; UTIL=${UTIL:-0.95}
MINAVAIL=${MINAVAIL:-420}     # GiB of MemAvailable required before launch: OFFGB pinned + a working floor
OPARAMS=${OPARAMS:-routed_experts.w13_weight_packed routed_experts.w2_weight_packed}
HOOKDIR=${HOOKDIR:-$HERE/hook}
LOGDIR=${LOGDIR:-$HOME/bench/k3-$(date +%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/k3/cache}; mkdir -p "$CACHE"/{vllm,triton,tilelang,nv,fi,dj}
HEAD=${RANK0_IP:?needs RANK0_IP (rail IP of rank 0)}
MYIP=$([ "$RANK" = 0 ] && echo "$RANK0_IP" || echo "${RANK1_IP:?needs RANK1_IP}")

# --- preflight -------------------------------------------------------------------------------------------------------------
test -s "$MODEL/model.safetensors.index.json" && [ "$(ls "$MODEL"/model-*.safetensors | wc -l | tr -d ' ')" = "$SHARDS" ] \
  || { echo "WEIGHTS MISSING/INCOMPLETE on $(hostname): $MODEL (want $SHARDS shards)"; exit 3; }
test -s "$MODEL/preprocessor_config.json" \
  || { echo "copy preprocessor_config.json from moonshotai/Kimi-K3 into $MODEL (see README)"; exit 3; }
test -s "$HOOKDIR/sitecustomize.py" || { echo "hook missing: $HOOKDIR/sitecustomize.py"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING on $(hostname): $IMAGE"; exit 3; }
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy (${USED} MiB) on $(hostname)"; exit 4; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge "$MINAVAIL" ] || { echo "MemAvailable ${AVAIL} GiB < ${MINAVAIL} GiB on $(hostname)"; exit 5; }
grep -E "MemAvailable|Shmem:|Mlocked|Unevictable|AnonPages|Cached:" /proc/meminfo > "$LOGDIR/meminfo-pre-rank$RANK.txt"

# --- fabric block (recipes/fabric-data-direct): Data Direct overlay + tuner3 + NCCL_IB_TC=106 ------------------------------
DDARGS=(); DD=${DD:-1}; TUNER_CSV=${TUNER_CSV:-tuner3.csv}
HOSTRDMA=${HOSTRDMA:-$HOME/hostrdma}; TUNERDIR=${TUNERDIR:-$HOME/nccl-tuner}
if [ "$DD" = 1 ]; then
  nm -D "$HOSTRDMA/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path \
    || { echo "$HOSTRDMA not staged (see recipes/fabric-data-direct; DD=0 to skip)"; exit 3; }
  for f in libnccl-tuner-example.so "$TUNER_CSV"; do [ -s "$TUNERDIR/$f" ] || { echo "$TUNERDIR/$f missing"; exit 3; }; done
  _C=$(docker create "$DD_IMAGE" true); _T=$(mktemp -d)
  docker cp "$_C:/usr/lib/aarch64-linux-gnu/libibverbs.so.1" "$_T/" >/dev/null; docker cp "$_C:/usr/lib/aarch64-linux-gnu/libmlx5.so.1" "$_T/" >/dev/null
  docker rm "$_C" >/dev/null; _IBV=$(readlink "$_T/libibverbs.so.1"); _MLX=$(readlink "$_T/libmlx5.so.1"); rm -rf "$_T"
  [ -n "$_IBV" ] && [ -n "$_MLX" ] || { echo "could not resolve image rdma lib names"; exit 3; }
  [ -c /dev/infiniband/rdma_cm ] && DDARGS+=(--device /dev/infiniband/rdma_cm)
  DDARGS+=(-v "$HOSTRDMA/libibverbs.so.1:/usr/lib/aarch64-linux-gnu/$_IBV:ro"
           -v "$HOSTRDMA/libmlx5.so.1:/usr/lib/aarch64-linux-gnu/$_MLX:ro"
           -v /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
           -v /etc/libibverbs.d:/etc/libibverbs.d:ro -v "$TUNERDIR:/opt/nccl-tuner:ro"
           -e NCCL_TUNER_PLUGIN=/opt/nccl-tuner/libnccl-tuner-example.so -e NCCL_TUNER_CONFIG_FILE=/opt/nccl-tuner/$TUNER_CSV
           -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1 -e NCCL_IB_TC=106)
fi

# --- engine env / args -----------------------------------------------------------------------------------------------------
IFN=${IFN:-enP1p3s0f1np1}; HCA=${HCA:-mlx5_1}   # the rail's netdev and RDMA device (port 1 of the CX-8 on our pair)
NCCL=(-e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e TP_SOCKET_IFNAME=$IFN
      -e NCCL_IB_HCA==$HCA:1 -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1
      -e NCCL_MNNVL_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET)
ENVS=(-e VLLM_HOST_IP="$MYIP" -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e VLLM_LOGGING_LEVEL=INFO
      -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e VLLM_KIMI_K3_GEMM_AR=0                                   # the fused CuTe GEMM + all-reduce is intra-node only
      -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024   # exact-size pinned host buffers for the offloaded experts
      -e PYTHONPATH=/opt/k3hook -e K3_REPACK3=1)                  # the load hook (./hook/sitecustomize.py)
[ "$MOE" = deep_gemm_mega_moe ] && { echo "MegaMoE owns its weights and does not compose with UVA offload"; exit 2; }

ARGS=(serve /model --served-model-name "$SERVED" --trust-remote-code
      --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice
      --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" --master-addr "$HEAD" --master-port "$MPORT"
      --disable-custom-all-reduce --moe-backend "$MOE"
      --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params $OPARAMS
      --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MBT" --gpu-memory-utilization "$UTIL"
      --no-enable-flashinfer-autotune
      --tensor-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel   # not PP: vLLM issue #50947
      --language-model-only)
if [ "$RANK" = 0 ]; then ARGS+=(--host 0.0.0.0 --port "$PORT"); else ARGS+=(--headless); fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

MOUNTS=(-v "$MODEL":/model:ro -v "$HOOKDIR":/opt/k3hook:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton
        -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/nv":/root/.nv -v "$CACHE/fi":/root/.cache/flashinfer -v "$CACHE/dj":/root/.dj)

docker rm -f "$N" 2>/dev/null || true
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE --device "/dev/infiniband/${UVERBS:-uverbs1}" "${DDARGS[@]}" \
  "${MOUNTS[@]}" "${ENVS[@]}" "${NCCL[@]}" --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server-rank$RANK.log" 2>&1 &
docker inspect --format '{{.Image}}' "$N" > "$LOGDIR/image-digest-rank$RANK.txt"
{ echo "launched $(date '+%F %T %Z') host=$(hostname) rank=$RANK image=$IMAGE"
  echo "model=$MODEL moe=$MOE offgb=$OFFGB ctx=$CTX seqs=$SEQS mbt=$MBT util=$UTIL dd=$DD tuner=$TUNER_CSV"
  printf 'args:'; printf ' %q' "${ARGS[@]}"; echo; } | tee "$LOGDIR/launch-rank$RANK.txt"
echo "log $LOGDIR/server-rank$RANK.log   (check: grep 'Total CPU offloaded parameters' -> 408.27 on both ranks)"
