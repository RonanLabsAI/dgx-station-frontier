#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-nemotron-ultra.sh -- NVIDIA Nemotron 3 Ultra 550B-A55B NVFP4 (or the IOI competitive-coding checkpoint, same
# architecture) on DGX Station GB300, vLLM nightly af7f9488 (nemotron_h, nemotron_h_mtp and the nemotron_v3 reasoning
# parser are all in that image).
#
# MODE=one  ONE Station, TP1 on the GB300. The checkpoint (~307 GiB without MTP) does not fit in 250.7 GiB of HBM, so
#           OFFGB GiB of routed-expert weights live in Grace memory and the GPU reads them over NVLink-C2C (UVA,
#           --offload-backend uva). MOE=marlin: the only MoE backend we found that keeps UVA-offloaded experts in Grace
#           (other backends re-allocate w13 on the GPU). Marlin runs NVFP4 as weight-only FP4 (W4A16).
# MODE=tp   TWO Stations, TP2 + EP2 over one RoCE link (vLLM mp multi-node), no offload (~155 GiB weights per rank),
#           MOE=flashinfer_trtllm (native NVFP4). Uses the fabric block of recipes/fabric-data-direct (DD=1).
#
# Expert parameter names in this vLLM build: FusedMoEFactory builds a MoERunner, so the experts are
# layers.N.mixer.experts.routed_experts.{w13,w2}_weight. "experts.w13_weight" matches NOTHING and silently offloads
# nothing; check "Total CPU offloaded parameters: 100.0" in the server log.
# Prefix caching is off (NVIDIA's competitive-coding deployment config), so benchmark cache hits are 0 by construction.
#
# usage: [MODEL=ultra|cc|/path] [MODEL_ROOT=/models/hf] [MODE=one|tp] [OFFGB=100] [MOE=..] [MTP=0|3|5]
#        [SSM=float32|float16] [CTX=73728] [SEQS=32] [MBT=16384] [UTIL=0.95] [PORT=8017] [LOGDIR=..] [EXTRA_ARGS=".."]
#        [RANK0_IP=<rail ip of rank 0> RANK1_IP=<rail ip of rank 1> IFN=<rail netdev> HCA=<rdma dev>]  (MODE=tp only)
#        launch-nemotron-ultra.sh RANK
#   MODE=one: "launch-nemotron-ultra.sh 0" on one Station.
#   MODE=tp:  RANK 1 on the second Station first, then RANK 0. API on rank 0: http://<host>:$PORT/v1, model nemotron-3-ultra.
#   Teardown: launch/stop-nemotron-ultra.sh (rank 0 first).
set -euo pipefail
RANK=${1:?rank 0|1}; N=${NAME:-nemotron-ultra}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}; DD_IMAGE=$IMAGE
MODEL_ROOT=${MODEL_ROOT:-/models/hf}
case "${MODEL:-ultra}" in
  ultra) MODEL=$MODEL_ROOT/nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4; SHARDS=${SHARDS:-113} ;;
  cc)    MODEL=$MODEL_ROOT/nvidia__NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4; SHARDS=${SHARDS:-108} ;;
  *)     SHARDS=${SHARDS:?SHARDS required for a custom MODEL path} ;;
esac
MODE=${MODE:-one}; MTP=${MTP:-0}; SSM=${SSM:-float32}; CTX=${CTX:-73728}; SEQS=${SEQS:-32}; MBT=${MBT:-16384}
UTIL=${UTIL:-0.95}; PORT=${PORT:-8017}; MPORT=${MPORT:-29617}; SERVED=${SERVED:-nemotron-3-ultra}
if [ "$MODE" = one ]; then OFFGB=${OFFGB:-100}; MOE=${MOE:-marlin}; MINAVAIL=${MINAVAIL:-200}
else OFFGB=${OFFGB:-0}; MOE=${MOE:-flashinfer_trtllm}; MINAVAIL=${MINAVAIL:-60}; fi
OPARAMS=${OPARAMS:-routed_experts.w13_weight routed_experts.w2_weight}
LOGDIR=${LOGDIR:-$HOME/bench/nemotron-ultra-$(date +%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/nemotron-ultra/cache}; mkdir -p "$CACHE"/{vllm,triton,nv,fi}
if [ "$MODE" = tp ]; then
  HEAD=${RANK0_IP:?MODE=tp needs RANK0_IP (rail IP of rank 0)}
  MYIP=$([ "$RANK" = 0 ] && echo "$RANK0_IP" || echo "${RANK1_IP:?MODE=tp needs RANK1_IP}")
else
  MYIP=${MYIP:-127.0.0.1}
fi

# --- preflight -------------------------------------------------------------------------------------------------------------
test -s "$MODEL/model.safetensors.index.json" && [ "$(ls "$MODEL"/model-*.safetensors | wc -l)" = "$SHARDS" ] \
  || { echo "WEIGHTS MISSING/INCOMPLETE on $(hostname): $MODEL (want $SHARDS shards)"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING on $(hostname): $IMAGE"; exit 3; }
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy (${USED} MiB) on $(hostname)"; exit 4; }
[ "$MODE" = one ] && [ "$OFFGB" -gt 0 ] && [ "$MOE" != marlin ] && echo "WARN: MOE=$MOE with UVA offload is unproven here; watch HBM"
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge "$MINAVAIL" ] || { echo "MemAvailable ${AVAIL} GiB < ${MINAVAIL} GiB on $(hostname)"; exit 5; }
grep -E "MemAvailable|Shmem:|Mlocked|Unevictable|AnonPages|Cached:" /proc/meminfo > "$LOGDIR/meminfo-pre-rank$RANK.txt"
nvidia-smi --query-gpu=index,name,memory.used,power.draw,power.limit,driver_version --format=csv > "$LOGDIR/gpus-pre-rank$RANK.csv"

# --- fabric block (recipes/fabric-data-direct): Data Direct overlay + tuner3 + NCCL_IB_TC=106; two-Station arm only -------
DDARGS=(); DD=${DD:-$([ "$MODE" = tp ] && echo 1 || echo 0)}; TUNER_CSV=${TUNER_CSV:-tuner3.csv}
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
      -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
# exact-size pinned host buffers for the offloaded experts (no power-of-two round-up of the pinned allocation)
[ "$OFFGB" -gt 0 ] && ENVS+=(-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024)
case "$MOE" in
  flashinfer_*) ENVS+=(-e VLLM_USE_FLASHINFER_MOE_FP4=1
                       -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,flashinfer::trtllm_fp4_block_scale_moe") ;;
esac

ARGS=(serve /model --served-model-name "$SERVED" --trust-remote-code --dtype auto
      --kv-cache-dtype fp8 --block-size 64 --mamba-ssm-cache-dtype "$SSM"
      --no-enable-prefix-caching --enable-chunked-prefill --reasoning-parser nemotron_v3
      --moe-backend "$MOE" --no-enable-flashinfer-autotune
      --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MBT" --gpu-memory-utilization "$UTIL"
      --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}')
[ "$OFFGB" -gt 0 ] && ARGS+=(--offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params $OPARAMS)
# MTP keeps 1 + MTP mamba state blocks per sequence (~0.4 GB each in fp32): MTP=3 at 32 sequences costs ~52 GiB of HBM.
[ "$MTP" -gt 0 ] && ARGS+=(--speculative-config "{\"method\":\"nemotron_h_mtp\",\"num_speculative_tokens\":$MTP}")
if [ "$MODE" = tp ]; then
  ARGS+=(--tensor-parallel-size 2 --enable-expert-parallel --disable-custom-all-reduce
         --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" --master-addr "$HEAD" --master-port "$MPORT")
  if [ "$RANK" = 0 ]; then ARGS+=(--host 0.0.0.0 --port "$PORT"); else ARGS+=(--headless); fi
elif [ "$MODE" = one ]; then
  [ "$RANK" = 0 ] || { echo "MODE=one runs rank 0 only"; exit 2; }
  ARGS+=(--tensor-parallel-size 1 --host 0.0.0.0 --port "$PORT")
else echo "MODE must be one|tp"; exit 2; fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

MOUNTS=(-v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton
        -v "$CACHE/nv":/root/.nv -v "$CACHE/fi":/root/.cache/flashinfer)
docker rm -f "$N" 2>/dev/null || true
T0=$(date +%s)
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE \
  $([ "$MODE" = tp ] && echo --device "/dev/infiniband/${UVERBS:-uverbs1}") "${DDARGS[@]}" \
  "${MOUNTS[@]}" "${ENVS[@]}" "${NCCL[@]}" --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server-rank$RANK.log" 2>&1 &
docker inspect --format '{{.Image}}' "$N" > "$LOGDIR/image-digest-rank$RANK.txt"
docker image inspect --format '{{join .RepoDigests ","}}' "$IMAGE" >> "$LOGDIR/image-digest-rank$RANK.txt"
echo "$T0" > "$LOGDIR/launch-epoch-rank$RANK.txt"
{ echo "launched $(date '+%F %T %Z') rank=$RANK image=$IMAGE"
  echo "model=$MODEL mode=$MODE moe=$MOE mtp=$MTP ssm=$SSM offgb=$OFFGB ctx=$CTX seqs=$SEQS mbt=$MBT util=$UTIL dd=$DD"
  printf 'args:'; printf ' %q' "${ARGS[@]}"; echo; } | tee "$LOGDIR/launch-rank$RANK.txt"
echo "log $LOGDIR/server-rank$RANK.log   (ready when: curl -s localhost:$PORT/v1/models returns 200)"
