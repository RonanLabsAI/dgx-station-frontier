#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-beam.sh RANK -- Reflection Beam 1.0 NVFP4 (501B / 23B active) under SGLang (lmsysorg/sglang:beam) on DGX Station GB300.
# Flags follow the SGLang cookbook "Beam 1.0, GB300, Low-Latency + NVFP4" recipe (docs.sglang.io/cookbook/autoregressive/Reflection/
# Beam-1_0, read 2026-10-16) with these deviations: TP1 on one GB300 (arms a1/b1) or TP2 over the two Stations' rail (arm a2) instead
# of TP4 on one NVLink domain; context 131072 / 16 running requests instead of 1M; --cpu-offload-gb (SGLang OffloaderV1: pinned Grace
# memory, copied to the GPU on every forward) when OFFGB > 0; the rail NCCL env of recipes/fabric-data-direct instead of the
# cookbook's MNNVL env (two Stations are not one NVLink domain).
#
# usage: [NNODES=1|2] [TP=1|2] [OFFGB=0] [MEMF=0.85] [SPEC=0|1] [BLOCK=8] [NOGRAPH=0|1] [OPTFLAGS=".."] launch-beam.sh RANK
#   NNODES=2 also needs RANK0_IP=<rail ip of rank 0> RANK1_IP=<rail ip of rank 1> [IFN=<rail netdev>] [HCA=<rdma dev>] [UVERBS=uverbs1]
#   RANK 0 serves the HTTP API on PORT; RANK 1 (NNODES=2 only) serves no HTTP. Start rank 1 first, then rank 0.
#   Measured arms (README): a1 = NNODES=1 TP=1 OFFGB=60 MEMF=0.92 NOGRAPH=1; b1 = a1 + SPEC=1;
#                           a2 = NNODES=2 TP=2 SPEC=1 (OFFGB=0 MEMF=0.85 NOGRAPH=0, the defaults).
#   All measured boots used OPTFLAGS='--enable-metrics --cuda-graph-max-bs-decode 16'.
# env: MODEL DRAFTER IMAGE NNODES TP OFFGB (GiB, 0 = no offload) MEMF (--mem-fraction-static) SPEC (1 = DFLASH with the DFlash2 drafter)
#      BLOCK (--speculative-dflash-block-size; the drafter's config.json says 16, we ran 8) NOGRAPH (1 = --disable-cuda-graph)
#      CTX (131072) SEQS (16) CHUNK (8192) PAGE (64) OPTFLAGS EXTRA_ARGS NAME (beam) SERVED (beam) PORT (8050) DPORT (29650, dist-init on
#      rank 0's rail IP) MINAVAIL (GiB of host MemAvailable required) LOGDIR CACHE DD (1 when NNODES=2: host RDMA libs + NCCL tuner, see
#      recipes/fabric-data-direct) TUNER_CSV (tuner3.csv)
# Teardown: docker rm -f beam (rank 0 first, then rank 1); the GB300 returns to about 22 MiB.
set -euo pipefail
RANK=${1:?rank 0|1}; N=${NAME:-beam}
IMAGE=${IMAGE:-lmsysorg/sglang:beam}
MODEL=${MODEL:-/models/hf/reflection__Beam-1.0-501B-A23B-NVFP4}
DRAFTER=${DRAFTER:-/models/hf/reflection__Beam-1.0-501B-A23B-DFlash2}
NNODES=${NNODES:-1}; TP=${TP:-1}; OFFGB=${OFFGB:-0}; MEMF=${MEMF:-0.85}; SPEC=${SPEC:-0}; BLOCK=${BLOCK:-8}; NOGRAPH=${NOGRAPH:-0}
CTX=${CTX:-131072}; SEQS=${SEQS:-16}; CHUNK=${CHUNK:-8192}; PAGE=${PAGE:-64}
SERVED=${SERVED:-beam}; PORT=${PORT:-8050}; DPORT=${DPORT:-29650}; MINAVAIL=${MINAVAIL:-150}
LOGDIR=${LOGDIR:-$HOME/bench/beam-$(date +%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/beamkit-cache}; mkdir -p "$CACHE"/{root,triton}
[ "$NNODES" = 1 ] && [ "$RANK" != 0 ] && { echo "NNODES=1 needs RANK 0"; exit 2; }
[ "$NNODES" = 2 ] && [ "$OFFGB" != 0 ] && { echo "a2 is the no-offload arm: OFFGB must be 0 with NNODES=2"; exit 2; }
if [ "$NNODES" = 2 ]; then
  HEAD=${RANK0_IP:?needs RANK0_IP (rail IP of rank 0)}
  MYIP=$([ "$RANK" = 0 ] && echo "$RANK0_IP" || echo "${RANK1_IP:?needs RANK1_IP}")
fi

MISS=$(cd "$MODEL" 2>/dev/null && python3 -c "import json,os;print(sum(not os.path.exists(f) for f in set(json.load(open('model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 1)
[ "$MISS" = 0 ] || { echo "WEIGHTS MISSING/INCOMPLETE on $(hostname): $MODEL (missing shards $MISS)"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING on $(hostname): $IMAGE"; exit 3; }
if [ "$SPEC" = 1 ]; then test -s "$DRAFTER/config.json" && test -s "$DRAFTER/model.safetensors" || { echo "DRAFTER MISSING on $(hostname): $DRAFTER"; exit 3; }; fi
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy/stranded (${USED} MiB) on $(hostname)"; exit 4; }
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge "$MINAVAIL" ] || { echo "MemAvailable ${AVAIL} GiB < ${MINAVAIL} GiB on $(hostname)"; exit 5; }
grep -E "MemAvailable|Shmem:|Mlocked|Unevictable|AnonPages|Cached:" /proc/meminfo > "$LOGDIR/meminfo-pre-rank$RANK.txt"

DDARGS=(); NCCL=()
if [ "$NNODES" = 2 ]; then
  if [ "${DD:-1}" = 1 ]; then
    TUNER_CSV=${TUNER_CSV:-tuner3.csv}
    nm -D "$HOME/hostrdma/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path || { echo "~/hostrdma not staged (DD=0 to skip)"; exit 3; }
    for f in libnccl-tuner-example.so "$TUNER_CSV"; do [ -s "$HOME/nccl-tuner/$f" ] || { echo "~/nccl-tuner/$f missing"; exit 3; }; done
    _C=$(docker create "$IMAGE" true); _T=$(mktemp -d)
    docker cp "$_C:/usr/lib/aarch64-linux-gnu/libibverbs.so.1" "$_T/" >/dev/null; docker cp "$_C:/usr/lib/aarch64-linux-gnu/libmlx5.so.1" "$_T/" >/dev/null
    docker rm "$_C" >/dev/null; _IBV=$(readlink "$_T/libibverbs.so.1"); _MLX=$(readlink "$_T/libmlx5.so.1"); rm -rf "$_T"
    [ -n "$_IBV" ] && [ -n "$_MLX" ] || { echo "could not resolve image rdma lib names"; exit 3; }
    [ -c /dev/infiniband/rdma_cm ] && DDARGS+=(--device /dev/infiniband/rdma_cm)
    DDARGS+=(-v "$HOME/hostrdma/libibverbs.so.1:/usr/lib/aarch64-linux-gnu/$_IBV:ro"
             -v "$HOME/hostrdma/libmlx5.so.1:/usr/lib/aarch64-linux-gnu/$_MLX:ro"
             -v /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
             -v /etc/libibverbs.d:/etc/libibverbs.d:ro
             -v "$HOME/nccl-tuner:/opt/nccl-tuner:ro"
             -e NCCL_TUNER_PLUGIN=/opt/nccl-tuner/libnccl-tuner-example.so -e NCCL_TUNER_CONFIG_FILE=/opt/nccl-tuner/$TUNER_CSV
             -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1 -e NCCL_IB_TC=106)
  fi
  IFN=${IFN:-enP1p3s0f1np1}; HCA=${HCA:-mlx5_1}   # the rail's netdev and RDMA device (port 1 of the CX-8 on our pair)
  NCCL=(-e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e TP_SOCKET_IFNAME=$IFN
        -e NCCL_IB_HCA==$HCA:1 -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1
        -e NCCL_MNNVL_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET
        -e SGLANG_HOST_IP="$MYIP" --device "/dev/infiniband/${UVERBS:-uverbs1}")
fi
ENVS=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID)

ARGS=(-m sglang.launch_server --model-path /model --served-model-name "$SERVED"
      --tp "$TP" --attention-backend trtllm_mha --page-size "$PAGE" --context-length "$CTX"
      --max-running-requests "$SEQS" --chunked-prefill-size "$CHUNK" --mem-fraction-static "$MEMF" --enable-unified-memory
      --tool-call-parser beam --reasoning-parser beam --watchdog-timeout 1800)
[ "$OFFGB" != 0 ] && ARGS+=(--cpu-offload-gb "$OFFGB")
[ "$SPEC" = 1 ] && ARGS+=(--speculative-algorithm DFLASH --speculative-draft-model-path /drafter --speculative-dflash-block-size "$BLOCK")
[ "$NOGRAPH" = 1 ] && ARGS+=(--disable-cuda-graph)
if [ "$NNODES" = 2 ]; then
  ARGS+=(--nnodes 2 --node-rank "$RANK" --dist-init-addr "$HEAD:$DPORT" --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion)
fi
[ "$RANK" = 0 ] && ARGS+=(--host 0.0.0.0 --port "$PORT")
[ -n "${OPTFLAGS:-}" ] && ARGS+=($OPTFLAGS)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

MOUNTS=(-v "$MODEL":/model:ro -v "$CACHE/root":/root/.cache -v "$CACHE/triton":/root/.triton)
[ "$SPEC" = 1 ] && MOUNTS+=(-v "$DRAFTER":/drafter:ro)
docker rm -f "$N" 2>/dev/null || true
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size 32g --cap-add IPC_LOCK --cap-add SYS_NICE "${DDARGS[@]}" "${NCCL[@]}" \
  "${MOUNTS[@]}" "${ENVS[@]}" --entrypoint python3 "$IMAGE" "${ARGS[@]}" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server-rank$RANK.log" 2>&1 &
docker inspect --format '{{.Image}}' "$N" > "$LOGDIR/image-digest-rank$RANK.txt"
{ echo "launched $(date '+%F %T %Z') host=$(hostname) rank=$RANK gpu=$GB300 image=$IMAGE"
  echo "model=$MODEL nnodes=$NNODES tp=$TP offgb=$OFFGB memf=$MEMF spec=$SPEC block=$BLOCK nograph=$NOGRAPH ctx=$CTX seqs=$SEQS chunk=$CHUNK page=$PAGE optflags='${OPTFLAGS:-}'"
  printf 'args:'; printf ' %q' "${ARGS[@]}"; echo; } | tee "$LOGDIR/launch-rank$RANK.txt"
echo "log $LOGDIR/server-rank$RANK.log"
