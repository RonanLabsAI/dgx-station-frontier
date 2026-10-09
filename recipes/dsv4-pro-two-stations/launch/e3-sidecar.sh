#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# e3-sidecar.sh RANK -- E3: start this Station's RTX PRO 6000 warm-expert sidecar for DeepSeek-V4-Pro EP rank RANK
# (rank 0 = API Station, rank 1 = headless Station) and wait for "serving". Runs in the vLLM image (torch, triton, cuda-python,
# cutlass-dsl 4.7.1 already match b12x master 2cc7f66a's pins) with b12x from a source checkout on PYTHONPATH; nothing
# is installed into the image. RTX pinned by UUID via CDI. The RTX must be empty (stop whatever normally runs on it).
#   env: E3DIR (this recipe folder, default ~/dgx-station-frontier/recipes/dsv4-pro-two-stations)  E3LOG (default ~/dsv4pro-e3/logs)  B12X (default ~/dsv4pro-e3/b12x)  E3ROWMAP  E3_MODE=a8|a16
#   stop: docker rm -f dsv4pro-e3-sidecar && rm -f /dev/shm/e3_dsv4pro_peer
set -euo pipefail
RANK=${1:?rank 0|1}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
MODEL=${MODEL:-/models/hf/deepseek-ai__DeepSeek-V4-Pro-0813}
E3DIR=${E3DIR:-$HOME/dgx-station-frontier/recipes/dsv4-pro-two-stations}; B12X=${B12X:-$HOME/dsv4pro-e3/b12x}
E3ROWMAP=${E3ROWMAP:-rowmap-e3-gsm-v1.json}; MODE=${E3_MODE:-a8}; N=dsv4pro-e3-sidecar
CACHE=${CACHE:-$HOME/dsv4pro-e3/cache}; E3LOG=${E3LOG:-$HOME/dsv4pro-e3/logs}; mkdir -p "$CACHE" "$E3LOG"
RTX=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/RTX PRO 6000/{print $1;exit}')
USED=$(nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 2048 ] || { echo "RTX busy (${USED} MiB) on $(hostname)"; exit 4; }
test -f "$B12X/b12x/moe/fused_moe/api.py" || { echo "b12x checkout missing at $B12X"; exit 3; }
test -s "$E3DIR/rowmaps/$E3ROWMAP" || { echo "rowmap missing"; exit 3; }
rm -f /dev/shm/e3_dsv4pro_peer 2>/dev/null || true
docker rm -f "$N" 2>/dev/null || true
LOG="$E3LOG/sidecar-rank$RANK.log"; : > "$LOG"
docker run -d --name "$N" --device "nvidia.com/gpu=$RTX" --ipc host --network host --cap-add SYS_NICE \
  --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "$MODEL":/model:ro -v "$E3DIR":/e3:ro -v "$B12X":/b12x:ro -v "$E3LOG":/logs -v "$CACHE":/root/.cache \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
  -e PYTHONPATH=/b12x -e E3_HOOK_DIR=/e3/hook -e CUDA_DEVICE_ORDER=PCI_BUS_ID --entrypoint bash "$IMAGE" -c \
  "python3 -c 'import b12x, b12x.moe.fused_moe as fm; print(\"b12x\", b12x.__file__)' && \
   exec numactl --membind=0 python3 /e3/sidecar/peer_server_v4.py --rowmap /e3/rowmaps/$E3ROWMAP --model /model \
     --ep-rank $RANK --ep-size 2 --mode $MODE --logdir /logs" >/dev/null
nohup docker logs -f "$N" > "$LOG" 2>&1 &
echo "$(date '+%F %T') sidecar rank $RANK on RTX $RTX ($(hostname)), rowmap $E3ROWMAP, mode $MODE; waiting for serving"
for i in $(seq 1 240); do
  grep -q "^serving" "$LOG" 2>/dev/null && { tail -4 "$LOG"; nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader; exit 0; }
  docker ps -q -f name="^$N\$" | grep -q . || { echo "sidecar died"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
echo "sidecar not serving after 20 min"; tail -20 "$LOG"; exit 1
