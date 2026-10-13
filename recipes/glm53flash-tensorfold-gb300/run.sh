#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 4bpw on ONE NVIDIA DGX Station GB300 (sm_103) with TensorFold v0.6.0 + Mia's GLM-5.3-Flash recipe
# patches + our 9001-9003, TP1. Run ./prepare.sh first. Never touches other containers.
#
# usage: CONFIG=gb300|mia DRAFTS=dflash2|none ./run.sh [steps...]
#   steps (default): preflight env serve smoke gsm8k bench bench16 teardown   (bench16 = C8 x 64 prompts)
#   CONFIG=mia   : Mia's recipe defaults, changed only where one GPU needs it (TP=1, COMM=nccl, SPLIT=0), PARALLEL 4
#   CONFIG=gb300 : + PARALLEL 8, TF_GLM_L2PF=0, TF_GLM_SPLIT_TARGET=576, TF_GLM_SHAPE_SK=0 (measured faster on GB300),
#                  and the host-gap settings TF_GLM_MULTI_LONE=1 TF_GLM_MULTI_ASYNC=1 TF_GLM_MULTI_SAMPLER=packed (replies unchanged)
# knobs: MODELS_DIR (where prepare.sh put the weights), OUT_ROOT (logs), EXTRA_ENV="K=V ...", IMAGE_TAG, PROFILE=1 (nsys)
#
# Isolation: --network none (loopback only; the bench client runs inside via docker exec), mounts = checkpoint (ro),
# drafter (ro), one output dir. GPU = the GB300 only, by UUID (TensorFold's engine uses device 0).
set -euo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-ngc}"; DENSE="${DENSE:-q4}"; DRAFTS="${DRAFTS:-dflash2}"; CONFIG="${CONFIG:-gb300}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"; OUT_ROOT="${OUT_ROOT:-$PWD/runs}"
IMAGE="${IMAGE_TAG:-tensorfold-glm53flash-gb300:v0.6.0-c4646171-mia-b5808ac-9001-9003-$BASE}"
EXTRA_ENV="${EXTRA_ENV:-}"; CACHE_FROM="${CACHE_FROM:-}"; PROFILE="${PROFILE:-0}"
NAME="tf-glm53flash"
CKPT="$MODELS_DIR/Mia-AiLab__GLM-5.3-Flash-EXL3-4bpw-TensorFold"
DRAFTER="$MODELS_DIR/incoai__GLM-5.3-Flash-DFlash2-bf582e4e"
RUN="${RUN:-$(date +%Y%m%d-%H%M)-$CONFIG-$DENSE-$DRAFTS}"
OUT="$OUT_ROOT/$RUN"
if [[ "$CONFIG" == gb300 ]]; then
  PARALLEL="${PARALLEL:-8}"; BENCH16_CONC="${BENCH16_CONC:-8}"
  EXTRA_ENV="TF_GLM_L2PF=0 TF_GLM_SPLIT_TARGET=576 TF_GLM_SHAPE_SK=0 TF_GLM_MULTI_LONE=1 TF_GLM_MULTI_ASYNC=1 TF_GLM_MULTI_SAMPLER=packed ${EXTRA_ENV:-}"
fi
CONTEXT="${CONTEXT:-1048576}"; PARALLEL="${PARALLEL:-4}"; MAX_TOKENS="${MAX_TOKENS:-32768}"; BENCH16_CONC="${BENCH16_CONC:-4}"
RESERVE_GIB="${RESERVE_GIB:-16}"; KV_POOL_GIB="${KV_POOL_GIB:-40}"
URL=http://127.0.0.1:8888
STEPS=("$@"); [[ ${#STEPS[@]} -gt 0 ]] || STEPS=(preflight env serve smoke gsm8k bench bench16 teardown)
mkdir -p "$OUT/cache"
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*" | tee -a "$OUT/progress.log"; }
gb300_uuid() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2=="NVIDIA GB300"{print $1}'; }
gb300_used() { nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1=="NVIDIA GB300"{print $2}'; }
ex() { docker exec "$NAME" "$@"; }

preflight() {
  log "preflight: image $IMAGE, out $OUT"
  docker image inspect "$IMAGE" --format 'image {{.Id}}' | tee -a "$OUT/progress.log"
  local used; used=$(gb300_used)
  log "GB300 memory used: ${used} MiB"
  if (( used > 1024 )); then log "ABORT: GB300 is in use (${used} MiB). Someone else's model is loaded."; exit 4; fi
  docker ps --format '{{.Names}} {{.Status}}' > "$OUT/containers-before.txt"
  [[ -f "$CKPT/.revision" && "$(cat "$CKPT/.revision")" == 078455ffe6472f9a52fbc1139f58b9db2881b25c ]] || { log "ABORT: checkpoint revision"; exit 4; }
  [[ -f "$CKPT/.verified" ]] || { log "ABORT: checkpoint not sha256-verified (run ./prepare.sh)"; exit 4; }
  [[ "$(cat "$DRAFTER/.revision")" == bf582e4eacc1810f76656d1811693ff6c6737d2a ]] || { log "ABORT: drafter revision"; exit 4; }
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total,clocks.max.sm --format=csv > "$OUT/gpus-before.csv"
  log "preflight ok"
}

envfile() {
  # Mia's defaults, computed by her config.sh INSIDE the image (not on the host), with the TP1 overrides
  docker run --rm --network none --entrypoint bash \
    -e TP=1 -e COMM=nccl -e SPLIT=0 -e DENSE="$DENSE" -e KV=fp8 -e KV_POOL_GIB="$KV_POOL_GIB" \
    -e CONTEXT="$CONTEXT" -e PARALLEL="$PARALLEL" -e MAX_TOKENS="$MAX_TOKENS" -e WORKER=none \
    -e STREAM_SMOOTH=0 "$IMAGE" -c \
    'source /opt/mia-scripts/config.sh >/dev/null 2>&1; env | grep -E "^(TF_GLM_|TENSORFOLD_)" | grep -v -E "^(TF_GLM_DISPLAY_KV_BACKEND|TENSORFOLD_MEMORY_RESERVE_GIB)=" | sort' \
    > "$OUT/tf.env"
  echo "TENSORFOLD_MEMORY_RESERVE_GIB=$RESERVE_GIB" >> "$OUT/tf.env"
  for kv in $EXTRA_ENV; do sed -i "/^${kv%%=*}=/d" "$OUT/tf.env"; echo "$kv" >> "$OUT/tf.env"; log "extra env: $kv"; done
  grep -E '^TF_GLM_(KV|DENSE|COMM|HC_SPLIT|CACHE_GIB|DISPLAY_KV_MIB|STREAM_SMOOTH)=' "$OUT/tf.env" | tee -a "$OUT/progress.log"
  grep -q '^TF_GLM_COMM=nccl$' "$OUT/tf.env" && grep -q '^TF_GLM_STREAM_SMOOTH=0$' "$OUT/tf.env" && grep -q '^TF_GLM_HC_SPLIT=0$' "$OUT/tf.env" && grep -q '^TF_GLM_DISPLAY_KV_MIB=0$' "$OUT/tf.env" \
    || { log "ABORT: env overrides did not take"; exit 5; }
}

serve() {
  local uuid; uuid=$(gb300_uuid)
  local draft=(--drafter /drafter); [[ "$DRAFTS" == none ]] && draft=(--no-drafts)
  log "serve: GB300 $uuid, context $CONTEXT, parallel $PARALLEL, dense $DENSE, drafts $DRAFTS, image $IMAGE, profile $PROFILE"
  if [[ -n "$CACHE_FROM" && -d "$CACHE_FROM/cache" ]]; then cp -a "$CACHE_FROM/cache/." "$OUT/cache/"; log "JIT cache copied from $CACHE_FROM"; fi
  local entry=(--entrypoint tensorfold "$IMAGE" serve)
  [[ "$PROFILE" == 1 ]] && entry=(--entrypoint nsys "$IMAGE" launch --session-new=tf --trace=cuda,nvtx --cuda-graph-trace=node tensorfold serve)
  docker run -d --name "$NAME" --restart no --network none --gpus "\"device=$uuid\"" \
    --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$CKPT":/model:ro -v "$DRAFTER":/drafter:ro -v "$OUT":/out \
    --env-file "$OUT/tf.env" "${entry[@]}" /model --tp 1 --rank 0 --host 127.0.0.1 --port 8888 "${draft[@]}" \
    --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" > "$OUT/container.id"
  local t0=$SECONDS
  ( docker logs -f "$NAME" > "$OUT/server.log" 2>&1 & )
  ( while docker inspect "$NAME" >/dev/null 2>&1; do nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv,noheader >> "$OUT/gpu-1s.csv"; sleep 1; done & )
  until ex python3 -c "import urllib.request;urllib.request.urlopen('$URL/health',timeout=5)" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]]; then
      log "FAIL: server exited during load after $((SECONDS - t0)) s; last log lines:"; tail -40 "$OUT/server.log" | tee -a "$OUT/progress.log"; exit 6
    fi
    (( SECONDS - t0 > 3600 )) && { log "FAIL: not healthy after 3600 s"; exit 6; }
    sleep 10
  done
  log "healthy after $((SECONDS - t0)) s; GB300 used $(gb300_used) MiB"
  ex python3 -c "import urllib.request;print(urllib.request.urlopen('$URL/metrics',timeout=5).read().decode()[:4000])" > "$OUT/metrics-ready.txt" 2>&1 || true
}

smoke() { log "smoke (3 code + 2 tool-call, read by eye)"; ex python3 /opt/b1/smoke_tf.py "$URL" /out/smoke.json | tee "$OUT/smoke.txt"; }

gsm8k() {
  log "GSM8K-50 (thinking on, greedy, C4)"
  ex python3 /opt/b1/gsm8k_tf.py "$URL" /opt/b1/gsm8k_test.jsonl /out/gsm8k50.jsonl --n 50 --conc 4 --think 1 | tee -a "$OUT/progress.log"
}

b() {  # b LABEL SLOT CONC MAXTOK SEED
  log "bench $1: slot $2, C$3, max_tokens $4, seed $5"
  ex python3 /opt/b1/bench_tf.py "$URL" "/opt/b1/pool/$2.jsonl" "/out/bench-$1.jsonl" --conc "$3" --max-tokens "$4" --seed "$5" --label "$1" \
    > "$OUT/bench-$1.summary.json" || { local rc=$?; log "bench $1 rc=$rc (3 = CONTAMINATED by cached tokens)"; }
  python3 - "$OUT/bench-$1.summary.json" <<'EOF' | tee -a "$OUT/progress.log"
import json,sys
s=json.load(open(sys.argv[1]))
f=lambda k:(s[k] or {}).get('mean')
print(f"  decode {f('decode_tps')} tok/s/req | output {f('output_tps')} | agg {s['aggregate_output_tps']:.1f} | ttft {f('ttft_s')} s | prefill {f('prefill_tps')} | cached max {s['max_cached_frac']:.3f} | finish {s['finish_reasons']}")
EOF
}

bench() {
  # separate warm-up invocations on their own slots and seeds; C1 = mean of 3 runs (thinking off, prose/short prompts)
  b warm-short warm-short 4 256 101
  b c1-r1 c1-r1 1 1024 111
  b c1-r2 c1-r2 1 1024 112
  b c1-r3 c1-r3 1 1024 113
  b c4-r1 c4-r1 4 1024 141
  b warm-long8k warm-long8k 1 64 181
  b prefill8k prefill8k 1 16 182
  b c1long-r1 c1long-r1 1 1024 183
  b warm-long32k warm-long32k 1 16 321
  b prefill32k prefill32k 1 16 322
  python3 - "$OUT" <<'EOF' | tee "$OUT/B1-summary.txt"
import json,glob,os,statistics,sys
o=sys.argv[1]; S={os.path.basename(p)[6:-13]:json.load(open(p)) for p in glob.glob(o+'/bench-*.summary.json')}
m=lambda l,k:(S[l][k] or {}).get('mean') if l in S else None
c1=[m(f'c1-r{i}','decode_tps') for i in (1,2,3)]
print('C1 decode tok/s (mean of 3 runs):', round(statistics.fmean([x for x in c1 if x]),1) if any(c1) else None, [round(x,1) if x else None for x in c1])
print('C1 output tok/s (mean of 3 runs):', round(statistics.fmean([m(f'c1-r{i}','output_tps') for i in (1,2,3) if m(f'c1-r{i}','output_tps')]),1) if 'c1-r1' in S else None)
print('C4 aggregate output tok/s:', round(S['c4-r1']['aggregate_output_tps'],1) if 'c4-r1' in S else None, '| per-request decode', m('c4-r1','decode_tps'))
print('C1 decode after 8K prompt:', m('c1long-r1','decode_tps'))
print('prefill tok/s 8K / 32K:', m('prefill8k','prefill_tps'), '/', m('prefill32k','prefill_tps'))
print('max cached fraction over all runs:', max(s['max_cached_frac'] for s in S.values()), '| contaminated:', [k for k,s in S.items() if s['contaminated']])
EOF
}

bench16() {   # BENCH16_CONC: concurrency for the 64-prompt slot (GLM's --parallel is at most 8)
  local c="$BENCH16_CONC"
  b warm-c16 warm-c16 "$c" 256 1601
  b "c${c}-r1" c16-r1 "$c" 1024 1611
}

profile() {
  # one C1 decode (512 tokens) under nsys, after its own warm-up; kernel summary to CSV
  b prof-warm prof-warm 1 512 1701
  ex nsys start --session=tf --sample=none --cpuctxsw=none --output=/out/decode-c1 --force-overwrite=true
  b prof prof 1 512 1702
  ex nsys stop --session=tf
  ex nsys stats --report cuda_gpu_kern_sum --format csv --output /out/decode-c1 /out/decode-c1.nsys-rep >/dev/null 2>&1 || true
  ls -la "$OUT"/decode-c1* | tee -a "$OUT/progress.log"
}

teardown() {
  log "teardown"
  docker stop -t 120 "$NAME" >/dev/null 2>&1 || true
  docker logs "$NAME" > "$OUT/server-final.log" 2>&1 || true
  docker rm "$NAME" >/dev/null 2>&1 || true
  sleep 7
  log "GB300 used after teardown: $(gb300_used) MiB"
  docker ps --format '{{.Names}} {{.Status}}' > "$OUT/containers-after.txt"
  diff <(cut -d' ' -f1 "$OUT/containers-before.txt" | sort) <(cut -d' ' -f1 "$OUT/containers-after.txt" | sort) \
    && log "other containers untouched (same set before/after)" || log "WARN: container set changed (see containers-*.txt)"
}

trap 'rc=$?; if (( rc != 0 )) && docker inspect "$NAME" >/dev/null 2>&1; then log "error rc=$rc: tearing down"; teardown; fi' EXIT
for s in "${STEPS[@]}"; do case "$s" in env) envfile ;; *) "$s" ;; esac; done
log "done: $OUT"
