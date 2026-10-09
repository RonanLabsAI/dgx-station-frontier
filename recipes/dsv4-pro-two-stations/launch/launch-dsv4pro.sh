#!/usr/bin/env bash
# launch-dsv4pro.sh -- DeepSeek-V4-Pro-0813 on two DGX Station GB300s (E1 stock / E2 pin-hot / E3 6000 warm tier).
# Same file for every stage; the as-run E1/E2 launcher was this file without the E3 block.
#   E3HOOK=0 (default): E1 (stock) or, with PIN_MODE=split ROWMAP=<file>, E2.
#   E3HOOK=1: mount the repo at /e3 and put /e3/hook FIRST on PYTHONPATH; its sitecustomize installs E2's
#     pin hook (if PIN_MODE=split) and then hook/e3_hook.py in every vLLM process. The rank's sidecar
#     (launch/e3-sidecar.sh RANK) must already be serving.   E3HOOK=0: the E1/E2 container unchanged.
#   E3_CHECK=N per-layer cosine-gate budget (eager calls with T >= E3_CHECK_MIN_T); E3_COUNT=1 dumps route counts.
#   Container name defaults to dsv4pro-e3 (bench: NAME=dsv4pro-e3 tools/bench_e3.sh ...).
# Original E1 header follows.
# launch-dsv4pro.sh -- (2026-10-07): DeepSeek-V4 family (model_type deepseek_v4) across BOTH GB300 Stations,
# vLLM native multi-node (mp backend, --nnodes 2) over rail 1 (mlx5_1 / enP1p3s0f1np1), with UVA routed-expert offload into
# Grace. E0 dress rehearsal: MODEL=<a 156 GB deepseek_v4-class checkpoint> SHARDS=<n> OFFGB=40. E1: the defaults below.
#
# Pitfalls handled (our pre-run code reading of vLLM af7f9488; B-numbers are our internal labels):
#   B4  PIN=alloc (default): PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -> exact-size pinning (no pow2 round-up).
#       PIN=nopin: VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 (fallback if host Shmem/Mlocked delta >= 1.5x offload).
#   B6  never deep_gemm_mega_moe with offload (it re-allocates w13 on the GPU); default MOE=marlin.
#       MOE=flashinfer_trtllm (arm B) adds the B8 autotune skip env.
#   B7  PP has no decode microbatching -> MODE=tp (TP2 + EP2) is primary; MODE=pp (TP1 x PP2) is the comparison arm.
#   B14 --cpu-offload-params $OPARAMS (WRONG for af7f9488: see OPARAMS below).
#       Check "Total CPU offloaded parameters: X GiB" on BOTH ranks.
# Fabric: Data Direct block (Data Direct overlay + tuner3 + NCCL_IB_TC=106), GPU pinned by UUID via CDI.
#
# usage: [MODEL=..] [OFFGB=200] [MODE=tp|pp] [MOE=marlin] [PIN=alloc|nopin] [CTX=131072] [SEQS=16] [UTIL=0.95]
#        [LOGDIR=~/bench/dsv4pro-hhmm] [EXTRA_ARGS="..."] RANK0_IP=<rail ip> RANK1_IP=<rail ip> launch-dsv4pro.sh RANK
#   Start RANK 1 first, then RANK 0.  API: http://<rank-0 host>:8010/v1  model deepseek-v4-pro
# TEARDOWN (multi-node vLLM has stranded HBM before): launch/stop-dsv4pro.sh on rank 0 FIRST, then on rank 1; then check both
# GB300s are back to ~22 MiB. If HBM stays stranded: STOP and report (no reboot).
set -euo pipefail
RANK=${1:?rank 0|1}; HEAD=${RANK0_IP:?rail IP of rank 0}; N=${NAME:-dsv4pro-e3}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}; DD_IMAGE=$IMAGE
MODEL=${MODEL:-/models/hf/deepseek-ai__DeepSeek-V4-Pro-0813}; SHARDS=${SHARDS:-66}
SERVED=${SERVED:-deepseek-v4-pro}; PORT=${PORT:-8010}; MPORT=${MPORT:-29614}
OFFGB=${OFFGB:-200}; MODE=${MODE:-tp}; MOE=${MOE:-marlin}; PIN=${PIN:-alloc}
CTX=${CTX:-131072}; SEQS=${SEQS:-16}; MBT=${MBT:-8192}; UTIL=${UTIL:-0.95}; LMO=${LMO:-1}
PPPART=${PPPART:-31,30}; MINAVAIL=${MINAVAIL:-420}
# B14 correction (E0, 2026-10-07): in af7f9488 DSV4 experts live at layers.N.ffn.experts.routed_experts.* (MoERunner),
# so "experts.w13_weight" never matches a segment and NOTHING is offloaded. Use routed_experts.*.
OPARAMS=${OPARAMS:-routed_experts.w13_weight routed_experts.w2_weight}
LOGDIR=${LOGDIR:-$HOME/bench/dsv4pro-e3-$(date +%Y%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/dsv4pro/cache}; mkdir -p "$CACHE"/{vllm,triton,tilelang,nv,fi,dj} "$HOME/dsv4pro/prof"
MYIP=$([ "$RANK" = 0 ] && echo "$RANK0_IP" || echo "${RANK1_IP:?rail IP of rank 1}")

# --- preflight -----------------------------------------------------------------------------------------------------------
test -s "$MODEL/model.safetensors.index.json" && [ "$(ls "$MODEL"/model-*.safetensors | wc -l)" = "$SHARDS" ] \
  || { echo "WEIGHTS MISSING/INCOMPLETE on $(hostname): $MODEL (want $SHARDS shards)"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING on $(hostname): $IMAGE"; exit 3; }
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy/stranded (${USED} MiB) on $(hostname)"; exit 4; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge "$MINAVAIL" ] || { echo "MemAvailable ${AVAIL} GiB < ${MINAVAIL} GiB on $(hostname)"; exit 5; }
grep -E "MemAvailable|Shmem:|Mlocked|Unevictable|AnonPages|Cached:" /proc/meminfo > "$LOGDIR/meminfo-pre-rank$RANK.txt"

# --- Data Direct block (recipes/fabric-data-direct): DD overlay + tuner3 + NCCL_IB_TC=106; DD=0 = off ---
DD=${DD:-1}; DDARGS=(); TUNER_CSV=${TUNER_CSV:-tuner3.csv}   # file in ~/nccl-tuner; tuner3 = LL 2ch <=128K (fixes tuner2's C8 -3.2%)
if [ "$DD" = 1 ]; then
  nm -D "$HOME/hostrdma/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path || { echo "~/hostrdma not staged (DD=0 to skip)"; exit 3; }
  for f in libnccl-tuner-example.so "$TUNER_CSV"; do [ -s "$HOME/nccl-tuner/$f" ] || { echo "~/nccl-tuner/$f missing"; exit 3; }; done
  _C=$(docker create "$DD_IMAGE" true); _T=$(mktemp -d)
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

# --- engine env / args ---------------------------------------------------------------------------------------------------
IFN=${IFN:-enP1p3s0f1np1}; HCA=${HCA:-mlx5_1}   # the rail's netdev and RDMA device (port 1 of the CX-8 on our pair)
NCCL=(-e NCCL_SOCKET_IFNAME=$IFN -e GLOO_SOCKET_IFNAME=$IFN -e TP_SOCKET_IFNAME=$IFN
      -e NCCL_IB_HCA==$HCA:1 -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1
      -e NCCL_MNNVL_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET)
ENVS=(-e VLLM_HOST_IP="$MYIP" -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e VLLM_LOGGING_LEVEL=INFO
      -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
case "$PIN" in
  alloc) ENVS+=(-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024) ;;
  nopin) ENVS+=(-e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1) ;;
  *) echo "PIN must be alloc|nopin"; exit 2 ;;
esac
[ "$MOE" = deep_gemm_mega_moe ] && { echo "B6: never MegaMoE with UVA offload"; exit 2; }
[ "$MOE" != marlin ] && ENVS+=(-e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,flashinfer::trtllm_fp4_block_scale_moe")
[ -n "${ROUTECOUNT:-}" ] && ENVS+=(-e ROUTECOUNT="$ROUTECOUNT")

ARGS=(serve /model --served-model-name "$SERVED" --trust-remote-code
      --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice
      --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" --master-addr "$HEAD" --master-port "$MPORT"
      --disable-custom-all-reduce --moe-backend "$MOE"
      --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params $OPARAMS
      --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MBT" --gpu-memory-utilization "$UTIL"
      --no-enable-flashinfer-autotune)
if [ "$MODE" = tp ]; then
  ARGS+=(--tensor-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel)
elif [ "$MODE" = pp ]; then
  ARGS+=(--tensor-parallel-size 1 --pipeline-parallel-size 2); ENVS+=(-e VLLM_PP_LAYER_PARTITION="$PPPART")
else echo "MODE must be tp|pp"; exit 2; fi
[ "$LMO" = 1 ] && ARGS+=(--language-model-only)
if [ "$RANK" = 0 ]; then ARGS+=(--host 0.0.0.0 --port "$PORT"); else ARGS+=(--headless); fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

MOUNTS=(-v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton
        -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/nv":/root/.nv -v "$CACHE/fi":/root/.cache/flashinfer
        -v "$CACHE/dj":/root/.dj -v "$HOME/dsv4pro/prof":/prof)
# E2: pin-hot-experts hook (marlin split). PIN_MODE=split|off, ROWMAP=<file in ~/dsv4pro/prof>.
PINDIR=${PINDIR:-${E3DIR:-$HOME/dgx-station-frontier/recipes/dsv4-pro-two-stations}/hook}
if [ -n "${PIN_MODE:-}" ]; then
  [ -s "$PINDIR/dsv4pro_pin_hook.py" ] || { echo "pin hook missing"; exit 3; }
  [ "$PIN_MODE" = off ] || [ -s "$HOME/dsv4pro/prof/${ROWMAP:?ROWMAP required}" ] || { echo "rowmap missing"; exit 3; }
  MOUNTS+=(-v "$PINDIR:/opt/pinhook:ro")
  ENVS+=(-e PYTHONPATH=/opt/pinhook -e PIN_MODE="$PIN_MODE" -e PIN_ROWMAP="/prof/${ROWMAP:-none}" -e PIN_SELFTEST="${PIN_SELFTEST:-1}")
fi
[ -s "$HOME/dsv4pro/routecount/sitecustomize.py" ] && [ -n "${ROUTECOUNT:-}" ] && \
  MOUNTS+=(-v "$HOME/dsv4pro/routecount:/opt/routecount:ro" -e PYTHONPATH=/opt/routecount)

# --- E3 block ------------------------------------------------------------------------------------------------
E3HOOK=${E3HOOK:-0}; E3DIR=${E3DIR:-$HOME/dgx-station-frontier/recipes/dsv4-pro-two-stations}; E3ROWMAP=${E3ROWMAP:-rowmap-e3-gsm-v1.json}
E3LOG=${E3LOG:-$HOME/dsv4pro-e3/logs}
if [ "$E3HOOK" = 1 ]; then
  test -s "$E3DIR/hook/e3_hook.py" && test -s "$E3DIR/rowmaps/$E3ROWMAP" || { echo "E3 repo/rowmap missing under $E3DIR"; exit 3; }
  grep -q "^serving" "$E3LOG/sidecar-rank$RANK.log" 2>/dev/null && test -e /dev/shm/e3_dsv4pro_peer \
    || { echo "E3 sidecar not serving on $(hostname) (run launch/e3-sidecar.sh $RANK first)"; exit 4; }
  MOUNTS+=(-v "$E3DIR:/e3:ro" -v "$LOGDIR:/e3log")
  # last -e PYTHONPATH wins: /e3/hook first so its sitecustomize (E2 + E3) shadows /opt/pinhook's
  ENVS+=(-e PYTHONPATH=/e3/hook -e E3_HOOK=/e3/hook/e3_hook.py -e E3_ROWMAP=/e3/rowmaps/$E3ROWMAP -e E3_ENABLE=${E3_ENABLE:-1}
         -e E3_CHECK=${E3_CHECK:-0} -e E3_CHECK_MIN_T=${E3_CHECK_MIN_T:-1024} -e E3_STATS=/e3log/e3-stats-rank$RANK.json
         -e E3_SMALL_ROWS=64 -e E3_PEER_SPINS=${E3_PEER_SPINS:-20000000})
  [ -n "${PIN_MODE:-}" ] && ENVS+=(-e PIN_HOOK=/opt/pinhook/dsv4pro_pin_hook.py)
  [ "${E3_COUNT:-0}" = 1 ] && ENVS+=(-e E3_COUNT=/e3log/e3-counts-rank$RANK.json)
fi
echo "e3hook=$E3HOOK rowmap=$E3ROWMAP enable=${E3_ENABLE:-1} check=${E3_CHECK:-0} count=${E3_COUNT:-0} pin=${PIN_MODE:-} e2rowmap=${ROWMAP:-}" > "$LOGDIR/e3-rank$RANK.txt"

docker rm -f "$N" 2>/dev/null || true
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE --device /dev/infiniband/${UVERBS:-uverbs1} "${DDARGS[@]}" \
  "${MOUNTS[@]}" "${ENVS[@]}" "${NCCL[@]}" \
  --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server-rank$RANK.log" 2>&1 &
docker inspect --format '{{.Image}}' "$N" > "$LOGDIR/image-digest-rank$RANK.txt"
{ echo "launched $(date '+%F %T %Z') host=$(hostname) rank=$RANK gpu=$GB300 image=$IMAGE"
  echo "model=$MODEL mode=$MODE moe=$MOE pin=$PIN offgb=$OFFGB ctx=$CTX seqs=$SEQS mbt=$MBT util=$UTIL dd=$DD tuner=$TUNER_CSV"
  printf 'args:'; printf ' %q' "${ARGS[@]}"; echo; } | tee "$LOGDIR/launch-rank$RANK.txt"
echo "log $LOGDIR/server-rank$RANK.log"
