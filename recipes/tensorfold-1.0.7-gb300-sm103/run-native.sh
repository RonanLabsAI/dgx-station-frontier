#!/usr/bin/env bash
# TensorFold NATIVE 1.0.7 (e7c523eb) on the LEFT DGX Station: Nemotron 3.5 Lightning 30B-A3B.
#   N6  : RTX PRO 6000 (sm_120), Ash's release binary (linux-aarch64 archive, sha256-checked; carries sm_86/120/121), MLX 4-bit
#   G-T : GB300 (sm_103), our build (zig 0.17.0, nvcc 13.3, -Dsm=103, -Doptimize=safe): Ash's tf-cuda-test GPU checks
#   G4  : GB300, our build, Nemotron MLX 4-bit
#   GF  : GB300, our build, NVIDIA's NVFP4 checkpoint (needs sm_90+)
# Each serve arm: smoke (3 code + 2 tool calls), GSM8K-50 (thinking off, greedy), C1 x3, C4, C1 after 8K, prefill 8K/32K
# (public prompt pool, fresh seeds, cached-token guard). --parallel 1 = Ash's runbook command; C4 then queues (serial).
#
# Run on the left Station from ~/tensorfold/native-1.0.7, ONLY under a <claims tool> claim covering the left GB300 AND the
# left RTX PRO 6000. The controller stops the GB300 daily (<gb300-daily>) first; this script stops/restarts <rtx-daily>
# itself (also on error). Containers: --network none, one GPU each by UUID, weights read-only, one output dir.
# usage: ./run-native.sh [N6] [GT] [G4] [GF]     (default: all four, ~45 min)
set -euo pipefail
cd "$(dirname "$0")"
TF=$PWD
IMG=tensorfold-q27:v0.6.6-cb2ebf05-vllm          # used only as a base with python3 + our bench client and pool (/opt/b1)
REL=$TF/release/tensorfold-1.0.7-linux-aarch64      # Ash's archive
OURS=$TF/runs/build-sm103/zig-out                   # our sm_103 build
MLX=/models/hf/TensorFold__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-d9d758fb
NVF=/models/hf/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-bee75962
R=$RUNS/native-$(date +%Y%m%d-%H%M); mkdir -p "$R"
NAME=tf-native
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(N6 GT G4 GF)
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*" | tee -a "$R/progress.log"; }
uuid_of() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v n="$1" 'index($2,n){print $1; exit}'; }
used_of() { nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' -v n="$1" 'index($1,n){print $2; exit}'; }

restore_qwen() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [[ "$(docker inspect -f '{{.State.Running}}' <rtx-daily> 2>/dev/null)" != true ]]; then
    log "restore: docker start <rtx-daily>"; docker start <rtx-daily> >/dev/null
    for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8888/v1/models | grep -q qwen3.8-27b && break; sleep 10; done
  fi
  curl -s -m 5 http://127.0.0.1:8888/v1/models | grep -q qwen3.8-27b && log "<rtx-daily> healthy on :8888" || log "WARN: <rtx-daily> NOT healthy"
}
restore() { touch "$R/.stop-sampler"; restore_qwen; docker ps --format '{{.Names}} {{.Status}}' > "$R/containers-after.txt"; }
trap 'rc=$?; (( rc != 0 )) && log "error rc=$rc"; restore' EXIT

# ---------- preflight ----------
log "preflight: out $R"
(cd "$TF/release" && sha256sum -c --quiet tensorfold-1.0.7-linux-aarch64.tar.gz.sha256) || { log "ABORT: release archive sha256"; exit 4; }
[[ "$(cat $MLX/.revision)" == d9d758fb83953437f7263256b0d96157e2a348b8 && "$(cat $NVF/.revision)" == bee7596271d1495f6992ae224aefde4410e816b8 ]] || { log "ABORT: checkpoint revisions"; exit 4; }
(( $(used_of GB300) < 1024 )) || { log "ABORT: GB300 in use ($(used_of GB300) MiB): stop <gb300-daily> first, under the claim"; exit 4; }
docker ps --format '{{.Names}} {{.Status}}' > "$R/containers-before.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total,power.limit,clocks.max.sm --format=csv > "$R/gpus-before.csv"
( while [[ ! -f "$R/.stop-sampler" ]]; do nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv,noheader >> "$R/gpu-1s.csv"; sleep 1; done & )

serve_arm() {   # serve_arm ARM GPU_NAME BIN_DIR MODEL_DIR
  local arm=$1 gpu=$2 bindir=$3 model=$4 uuid; uuid=$(uuid_of "$gpu")
  mkdir -p "$R/$arm"
  log "[$arm] tensorfold native on $gpu ($uuid): $(basename "$model")"
  docker run -d --name "$NAME" --restart no --network none --gpus "\"device=$uuid\"" --shm-size 8g \
    --ulimit memlock=-1 -v "$bindir":/tf:ro -v "$model":/model:ro -v "$R/$arm":/out -e HOME=/out \
    --entrypoint /tf/bin/$(ls "$bindir/bin" | grep -E '^tensorfold(-native)?$' | head -1) "$IMG" \
    serve /model --name nemotron --host 127.0.0.1 --port 8080 --parallel 1 --context 40960 --temperature 0 --no-thinking > /dev/null
  ( docker logs -f "$NAME" > "$R/$arm-server.log" 2>&1 & )
  local t0=$SECONDS
  until docker exec "$NAME" python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/v1/models',timeout=5)" >/dev/null 2>&1; do
    [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == true ]] || { log "[$arm] FAIL: exited after $((SECONDS-t0)) s"; tail -30 "$R/$arm-server.log" | tee -a "$R/progress.log"; docker rm -f "$NAME" >/dev/null; return 6; }
    (( SECONDS - t0 > 900 )) && { log "[$arm] FAIL: not up after 900 s"; docker rm -f "$NAME" >/dev/null; return 6; }
    sleep 5
  done
  log "[$arm] up after $((SECONDS-t0)) s; $gpu used $(used_of "$gpu") MiB"
  local X="docker exec $NAME python3 /opt/b1"
  $X/smoke_tf.py http://127.0.0.1:8080 /out/smoke.json nemotron > "$R/$arm-smoke.txt" 2>&1 || log "[$arm] smoke rc=$?"
  $X/gsm8k_tf.py http://127.0.0.1:8080 /opt/b1/gsm8k_test.jsonl /out/gsm8k50.jsonl --n 50 --conc 1 --think 0 --max-tokens 4096 --model nemotron | tee -a "$R/progress.log"
  b() { log "[$arm] $1: C$3 max_tokens $4 seed $5"; $X/bench_tf.py http://127.0.0.1:8080 /opt/b1/pool/$2.jsonl /out/bench-$1.jsonl --conc $3 --max-tokens $4 --seed $5 --label $1 --model nemotron > "$R/$arm-bench-$1.summary.json" || log "[$arm] bench $1 rc=$? (3 = CONTAMINATED)"; }
  b warm-short warm-short 1 256 101
  b c1-r1 c1-r1 1 1024 111; b c1-r2 c1-r2 1 1024 112; b c1-r3 c1-r3 1 1024 113
  b c4-r1 c4-r1 4 1024 141
  b warm-long8k warm-long8k 1 64 181; b prefill8k prefill8k 1 16 182; b c1long-r1 c1long-r1 1 1024 183
  b warm-long32k warm-long32k 1 16 321; b prefill32k prefill32k 1 16 322
  python3 - "$R" "$arm" <<'EOF' | tee -a "$R/progress.log" "$R/SUMMARY-$arm.txt"
import json,glob,os,statistics,sys
R,arm=sys.argv[1],sys.argv[2]
S={os.path.basename(p)[len(arm)+7:-13]:json.load(open(p)) for p in glob.glob(f'{R}/{arm}-bench-*.summary.json')}
m=lambda l,k:(S[l].get(k) or {}).get('mean') if l in S else None
c1=[m(f'c1-r{i}','decode_tps') for i in (1,2,3)]
print(f'[{arm}] C1 decode mean of 3:', round(statistics.fmean([x for x in c1 if x]),1) if any(c1) else None, [round(x,1) if x else None for x in c1])
print(f'[{arm}] C4 (queued, --parallel 1) aggregate:', round(S['c4-r1']['aggregate_output_tps'],1) if 'c4-r1' in S else None)
print(f'[{arm}] C1 decode after 8K:', m('c1long-r1','decode_tps'), '| prefill 8K / 32K:', m('prefill8k','prefill_tps'), '/', m('prefill32k','prefill_tps'))
print(f'[{arm}] max cached fraction:', max(s['max_cached_frac'] for s in S.values()), '| contaminated:', [k for k,s in S.items() if s['contaminated']])
EOF
  docker stop -t 30 "$NAME" >/dev/null; docker logs "$NAME" > "$R/$arm-server-final.log" 2>&1; docker rm "$NAME" >/dev/null
  sleep 5; log "[$arm] $gpu used after teardown: $(used_of "$gpu") MiB"
}

for a in "${ARMS[@]}"; do
  case "$a" in
    N6)
      log "stopping <rtx-daily> for N6 (RTX PRO 6000 only)"; docker stop -t 60 <rtx-daily> >/dev/null; sleep 8
      (( $(used_of 'RTX PRO 6000') < 1024 )) || { log "ABORT: RTX PRO 6000 still in use"; exit 5; }
      serve_arm N6 'RTX PRO 6000' "$REL" "$MLX" || log "[N6] failed"
      restore_qwen ;;   # the 6000 goes back to its daily before the GB300 arms
    GT)
      U=$(uuid_of GB300); mkdir -p "$R/GT"
      log "[GT] Ash's tf-cuda-test GPU checks on GB300 (sm_103 build)"
      for t in "info" "smoke" "graph" "launch-ex" "ptx" "symbols" "overhead 1000 20" "overhead 10000 10" "overhead-pdl 1000 20" "overhead-pdl 10000 10" "cublaslt"; do
        echo "--- $t"
        # shellcheck disable=SC2086
        timeout 600 docker run --rm --network none --gpus "\"device=$U\"" -v "$OURS":/tf:ro --entrypoint /tf/bin/tf-cuda-test "$IMG" $t || echo "rc=$? for $t"
      done 2>&1 | tee "$R/GT/tests.log"
      timeout 600 docker run --rm --network none --gpus "\"device=$U\"" -e NCCL_DEBUG=WARN -v "$OURS":/tf:ro --entrypoint /tf/bin/tf-cuda-test "$IMG" nccl > "$R/GT/nccl.log" 2>&1 || echo "rc=$? for nccl" >> "$R/GT/nccl.log"
      log "[GT] FAIL/rc lines: $(grep -c -E '^FAIL|^rc=' "$R/GT/tests.log" "$R/GT/nccl.log" | tr '\n' ' ')" ;;
    G4) serve_arm G4 GB300 "$OURS" "$MLX" || log "[G4] failed" ;;
    GF) serve_arm GF GB300 "$OURS" "$NVF" || log "[GF] failed" ;;
  esac
done
touch "$R/.stop-sampler"
log "done: $R  (<rtx-daily> restored; the GB300 daily is the controller's to restore)"
