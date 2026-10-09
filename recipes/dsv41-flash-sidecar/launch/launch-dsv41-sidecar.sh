#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# launch-dsv41-sidecar.sh -- DeepSeek-V4.1-Flash on ONE DGX Station: GB300 hot experts (MegaMoE) + RTX PRO 6000 cold
# experts via original-el8's "M3" b12x sidecar (original-el8/dgx-station-gb300-research @7699ae54,
# deepseek-v4.1-flash/m3; Apache-2.0; NOT vendored here: clone it into $W/upstream). This launcher is ours.
# Runs on PUBLIC b12x master 2cc7f66a (el8 published results on an unpublished b12x branch; master's fused-MoE API
# matches every call the sidecar makes). First boot on a new map/image: gate with MEGA_PEER_CHECK (see README).
#
# The RTX PRO 6000 must be empty (~76 GB sidecar). For the two-Station DP2 numbers, run this on EACH Station.
# Two steps:  launch-dsv41-sidecar.sh sidecar   (waits for "serving")   then   launch-dsv41-sidecar.sh server
# Stop: launch/stop-dsv41.sh (or docker rm -f dsv41-a dsv41-a-sidecar).
# Thinking: per request {"chat_template_kwargs":{"thinking":false}} or {"chat_template_kwargs":{"reasoning_effort":50}}.
set -euo pipefail
STEP=${1:?usage: launch-dsv41-A.sh sidecar|server}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
MODEL=/models/hf/deepseek-ai__DeepSeek-V4.1-Flash
W=$HOME/dsv41/m3
HOOK=$W/upstream/hook
B12X=$HOME/dsv41/b12x            # git clone of local-inference-lab/b12x, master 2cc7f66a
ROWMAP=${ROWMAP:-rowmap-mix-v1.json}
CACHE=$HOME/dsv41/cache-a
SEQS=${SEQS:-32}   # 64 for the DP2/C64 numbers, together with the CG_SIZES below
CTX=${CTX:-131072}
KSCHED=${KSCHED:-"[[1,4,5],[5,8,3],[9,$SEQS,3]]"}
CG_SIZES=${CG_SIZES:-"1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128"}
# With SEQS=64 use CG_SIZES="1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 160 192 224 256":
# 64 seqs x (1 + 3 draft tokens) = 256 tokens per decode step; without these sizes C64 falls out of the graphs and drops ~45%.
test -f "$MODEL/config.json" || { echo "MODEL MISSING"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING: $IMAGE"; exit 3; }
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
RTX=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/RTX PRO 6000/{print $1;exit}')
mkdir -p "$CACHE"/{vllm,tilelang,triton,nv,dj,b12x} "$W/logs" "$W/prof"

if [ "$STEP" = sidecar ]; then
  USED=$(nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${USED:-99999}" -lt 2048 ] || { echo "RTX busy (${USED} MiB): stop whatever runs on it first"; exit 4; }
  test -f "$B12X/b12x/moe/fused_moe/api.py" || { echo "b12x checkout missing at $B12X"; exit 3; }
  # stale shared buffer from an earlier sidecar would be opened without O_CREAT by the GB300 side; start clean
  rm -f /dev/shm/vllm_peer_tier2 2>/dev/null || true
  docker rm -f dsv41-a-sidecar 2>/dev/null || true
  # Sidecar runs inside the same vLLM image (torch, triton, cuda-python, cutlass-dsl present) on the RTX only, with b12x
  # from source on PYTHONPATH. If `import b12x.moe.fused_moe` fails on a cutlass-dsl version pin, install b12x's pins into
  # a writable overlay: SIDECAR_PIP=1 (needs network; adds ~2 min).
  PIPCMD=""
  [ "${SIDECAR_PIP:-0}" = 1 ] && PIPCMD="pip install -q --no-deps nvidia-cutlass-dsl==4.7.1 nvidia-cutlass-dsl-libs-base==4.7.1 nvidia-cutlass-dsl-libs-core==4.7.1 nvidia-cutlass-dsl-libs-cu13==4.7.1 'apache-tvm-ffi>=0.1.6,<0.2' rich && "
  docker run -d --name dsv41-a-sidecar --gpus "device=$RTX" --ipc host --network host --cap-add SYS_NICE \
    --ulimit memlock=-1 --cap-add IPC_LOCK --security-opt label=disable \
    -v "$MODEL":/model:ro -v "$HOOK":/w:ro -v "$B12X":/b12x:ro -v "$W/local":/sc:ro -v "$W/logs":/logs \
    -v "$CACHE/b12x":/root/.cache -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
    -e PYTHONPATH=/b12x -e CUDA_DEVICE_ORDER=PCI_BUS_ID --entrypoint bash "$IMAGE" -c \
    "${PIPCMD}python3 -c 'import b12x.moe.fused_moe as fm; print(\"b12x ok\", fm.PackedSourceFormat(\"fp4_e8m0_k32\"))' && \
     exec numactl --membind=0 python3 /sc/peer_server2_local.py --rowmap /w/$ROWMAP --model /model" >/dev/null
  nohup docker logs -f dsv41-a-sidecar > "$W/logs/peer_server2.log" 2>&1 &
  echo "sidecar starting on RTX $RTX; waiting for 'serving' (published: 40 layers in 45 s, 560 graphs, 75.9 GB)"
  for i in $(seq 1 120); do
    grep -q "^serving" "$W/logs/peer_server2.log" 2>/dev/null && { tail -3 "$W/logs/peer_server2.log"; ls -la /dev/shm/vllm_peer_tier2; exit 0; }
    docker ps -q -f name=^dsv41-a-sidecar$ | grep -q . || { echo "sidecar died"; tail -20 "$W/logs/peer_server2.log"; exit 1; }
    sleep 5
  done
  echo "sidecar not serving after 10 min"; tail -20 "$W/logs/peer_server2.log"; exit 1
fi

[ "$STEP" = server ] || { echo "usage: sidecar|server"; exit 2; }
grep -q "^serving" "$W/logs/peer_server2.log" 2>/dev/null || { echo "sidecar not serving; run: $0 sidecar"; exit 4; }
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy (${USED} MiB)"; exit 4; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
O=$W/overlay-58132/tree/vllm; D=/usr/local/lib/python3.12/dist-packages/vllm
OV=()
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  test -f "$O/models/deepseek_v41/decoder_replay_layers.py" || { echo "overlay tree missing (runbook P3) or REPLAY_OVERLAY=0"; exit 3; }
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do OV+=(-v "$O/$f:$D/$f:ro"); done
fi
mkdir -p "$CACHE/fi"; OV+=(-v "$CACHE/fi":/root/.cache/flashinfer)   # FlashInfer JIT cache: first boot ~30 min, warm ~8-9 min
docker rm -f dsv41-a 2>/dev/null || true
docker run -d --name dsv41-a --gpus "device=$GB300" --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/triton":/root/.triton \
  -v "$CACHE/nv":/root/.nv -v "$CACHE/dj":/root/.dj -v "$W/prof":/prof \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro "${OV[@]}" \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e VLLM_LOGGING_LEVEL=INFO -e CUDA_LOG_FILE=stderr -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
  -e MEGA_HOOK=/w/mega_peer_hook.py -e MEGA_ROWMAP=/w/$ROWMAP -e MEGA_PEER=2 -e MEGA_PEER_MIN_TOKENS=1 \
  -e MEGA_PEER_CHECK=${MEGA_PEER_CHECK:-0} -e MEGA_PEER_CHECK_MIN_T=65 -e MEGA_COUNT=${MEGA_COUNT:-} -e MEGA_COLD_TRT=${MEGA_COLD_TRT:-0} \
  -e VLLM_EXP_PEER2_SMALL=64 -e MEGA_FUSED_SEND=2 -e MEGA_MHC_OVERLAP=1 -e MEGA_LL_GEMM=1 \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name deepseek-v4.1-flash --trust-remote-code --tensor-parallel-size 1 \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization ${GPU_UTIL:-0.95} --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --enable-prefix-caching \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED,\"draft_sample_method\":\"probabilistic\"}" \
  --cudagraph-capture-sizes $CG_SIZES --host 0.0.0.0 --port 8009 >/dev/null
nohup docker logs -f dsv41-a > "$HOME/dsv41/dsv41-a.log" 2>&1 &
echo "launched dsv41-a (peer v2, $ROWMAP, MEGA_PEER_CHECK=${MEGA_PEER_CHECK:-0}, COLD_TRT=${MEGA_COLD_TRT:-0}) log ~/dsv41/dsv41-a.log"
# Bring-up gate: MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 (keeps a 62 GiB Grace copy so the sidecar can be compared with
# TRT-LLM per layer). N must be >> 40 x the number of profile forwards: vLLM's dummy profile run consumes small budgets
# on all-zero inputs. Pass = cos >= 0.999 on real prompts. Then relaunch with the defaults (0 / 0).
