#!/usr/bin/env bash
# B3: full GLM-5.3 (Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold @ 2d747d0e, 272.7 GiB) across BOTH DGX Station GB300s, TP2
# over the 400G rail (mlx5_1 / enP1p3s0f1np1: left = rank 0, right = rank 1).
# TensorFold v0.6.0 (c4646171) + Mia's full GLM-5.3 recipe patches (eb6414b, 110) + 9101-glm-full-tp2 (CUDA_TP += 2).
# Drafter: RedHatAI/GLM-5.3-speculator.dspark @ b374b956 (Mia's default for this recipe).
#
# Runs from a third machine and drives both Stations over ssh. Both GB300s must be idle first (this script checks
# < 1 GiB used on each and aborts otherwise; it never stops or starts other containers). Host network, so NCCL can use the
# 400G rail; the HTTP server binds 127.0.0.1 on the left Station only. Weights and drafter are mounted read-only.
# Required environment (placeholders in this copy of the script):
#   LEFT_HOST, RIGHT_HOST   user@host of the rank-0 and rank-1 Stations (ssh keys, no password)
#   RAIL_IP_A, RAIL_IP_B    the two Stations' addresses on the 400G rail (rank 0 = A, rank 1 = B)
#   RUNS                    output directory on the Stations (also holds ckpt-full-verify.log from the sha256 check)
# The interface/HCA names below (enP1p3s0f1np1, mlx5_1, uverbs1) are the 400G port on our Stations; change them for yours.
#
# 2026-10-11: CP=0 and DISK_CACHE=0 (Mia's TP>2 defaults turn on context parallelism, FP4 KV, a 500K window and a
# 64 GiB /cache/pcache disk prompt cache that is not mounted here); devices: only /dev/infiniband/uverbs1 (no rdma_cm).
# usage: DRAFTS=dspark|none ./run-b3.sh      (default dspark; ~45-75 min: two loads of ~136 GiB a rank + JIT + bench)
set -euo pipefail
L="${LEFT_HOST:?set LEFT_HOST}"; Rt="${RIGHT_HOST:?set RIGHT_HOST}"
: "${RUNS:?set RUNS}"
IMAGE=tensorfold-b3:v0.6.0-c4646171-miafull-eb6414b-9101-ngc
NAME=tf-b3
CKPT=/models/hf/Mia-AiLab__GLM-5.3-EXL3-2.75bpw-TensorFold
DSPARK=/models/hf/RedHatAI__GLM-5.3-speculator.dspark-b374b956
DRAFTS="${DRAFTS:-dspark}"
RUN="b3-$(date +%Y%m%d-%H%M)-$DRAFTS"
OUT="$RUNS/$RUN"
MASTER="${RAIL_IP_A:?set RAIL_IP_A}"
: "${RAIL_IP_B:?set RAIL_IP_B}"
LOCAL="${LOCAL:-$PWD/logs}"
mkdir -p "$LOCAL/$RUN"
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T %Z')] $*" | tee -a "$LOCAL/$RUN/progress.log"; }
on() { local h=$1; shift; ssh -o BatchMode=yes "$h" "$@" </dev/null; }
onin() { local h=$1; shift; ssh -o BatchMode=yes "$h" "$@"; }   # keeps stdin (pipes, heredocs)
gb300_used() { on "$1" "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '\$1==\"NVIDIA GB300\"{print \$2}'"; }
NCCL_ENV="-e NCCL_SOCKET_IFNAME=enP1p3s0f1np1 -e GLOO_SOCKET_IFNAME=enP1p3s0f1np1 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA==mlx5_1:1 \
 -e NCCL_IB_TC=106 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"

teardown() {
  for h in $L $Rt; do
    on "$h" "docker stop -t 60 $NAME >/dev/null 2>&1; docker logs $NAME > $OUT/server-final-\$(hostname).log 2>&1; docker rm $NAME >/dev/null 2>&1; sleep 5; touch $OUT/.stop-sampler" || true
    log "$h GB300 used after teardown: $(gb300_used "$h") MiB"
  done
  rsync -a --exclude cache "$L:$OUT/" "$LOCAL/$RUN/left/" 2>/dev/null || true
  rsync -a --exclude cache "$Rt:$OUT/" "$LOCAL/$RUN/right/" 2>/dev/null || true
  log "logs in $LOCAL/$RUN (dailies are the controller's to restore)"
}
trap 'rc=$?; (( rc != 0 )) && log "error rc=$rc"; teardown' EXIT

# ---------- preflight ----------
for h in $L $Rt; do
  used=$(gb300_used "$h"); log "$h GB300 used: $used MiB"
  (( used < 1024 )) || { log "ABORT: $h GB300 in use (stop whatever is using it first)"; exit 4; }
  on "$h" "[[ \$(cat $CKPT/.revision) == 2d747d0e30eca6e3fe37ba63c471cdf9172faa83 && \$(cat $DSPARK/.revision) == b374b95663447ea0e935151be4f3d6666e36e6d7 ]] && docker image inspect $IMAGE --format '{{.Id}}' && mkdir -p $OUT/cache && docker ps --format '{{.Names}}' > $OUT/containers-before.txt" | tee -a "$LOCAL/$RUN/progress.log"
  on "$h" "grep -q '^EXIT 0$' $RUNS/ckpt-full-verify.log" || { log "ABORT: $h full checkpoint sha256 verify not clean"; exit 4; }
done
on "$L" "ping -c 2 -W 2 -I enP1p3s0f1np1 $RAIL_IP_B >/dev/null" || { log "ABORT: 400G rail left->right not answering"; exit 4; }

# ---------- env (Mia's full-recipe defaults, computed inside the image) ----------
on "$L" "docker run --rm --network none --entrypoint bash -e TP=2 -e COMM=nccl -e CP=0 -e DISK_CACHE=0 -e WORKER=none -e STREAM_SMOOTH=0 -e DRAFTER=$DRAFTS $IMAGE -c 'source /opt/mia-scripts/config.sh >/dev/null 2>&1; env | grep -E \"^(TF_GLM_|TENSORFOLD_)\" | grep -v -E \"^TF_GLM_DISPLAY_KV_BACKEND=\" | sort' > $OUT/tf.env; echo CONTEXT=\$(docker run --rm --network none --entrypoint bash -e TP=2 -e CP=0 -e DISK_CACHE=0 -e WORKER=none $IMAGE -c 'source /opt/mia-scripts/config.sh >/dev/null 2>&1; echo \$CONTEXT') > $OUT/ctx.txt; cat $OUT/tf.env $OUT/ctx.txt" | tee "$LOCAL/$RUN/tf.env"
on "$L" "grep -q '^TF_GLM_COMM=nccl$' $OUT/tf.env && grep -q '^TF_GLM_STREAM_SMOOTH=0$' $OUT/tf.env && ! grep -q '^TF_GLM_CP=1$' $OUT/tf.env && ! grep -q '^TF_GLM_DISK_CACHE=' $OUT/tf.env" || { log "ABORT: env overrides did not take (need COMM=nccl, CP off, no disk cache)"; exit 5; }
on "$L" "cat $OUT/tf.env" | onin "$Rt" "cat > $OUT/tf.env"
on "$Rt" "wc -l < $OUT/tf.env" | tee -a "$LOCAL/$RUN/progress.log"
CONTEXT=$(on "$L" "cut -d= -f2 $OUT/ctx.txt"); CONTEXT=${CONTEXT:-163840}
DRAFT_ARGS="--drafter /drafter"; [[ "$DRAFTS" == none ]] && DRAFT_ARGS="--no-drafts"
log "context $CONTEXT, drafts $DRAFTS"
# 2026-10-11: rank 0 used :8888, a port another service on the host already used (--network host), so the health check answered for that service
# (attempt 2 in the README). Own port + refuse if anything already listens on it.
B3_PORT=${B3_PORT:-8898}
if on "$L" "ss -ltn | grep -q ':$B3_PORT '"; then log "ABORT: port $B3_PORT already bound on $L (would answer for TensorFold)"; exit 7; fi
log "B3 rank-0 port $B3_PORT (free)"

serve_rank() {   # serve_rank HOST RANK EXTRA
  local h=$1 r=$2 extra=$3
  on "$h" "U=\$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '\$2==\"NVIDIA GB300\"{print \$1}'); \
    docker run -d --name $NAME --restart no --network host --gpus \"\\\"device=\$U\\\"\" --shm-size 16g \
      --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
      --device /dev/infiniband/uverbs1 \
      -v $CKPT:/model:ro -v $DSPARK:/drafter:ro -v $OUT:/out --env-file $OUT/tf.env $NCCL_ENV \
      --entrypoint tensorfold $IMAGE serve /model --tp 2 --rank $r --master $MASTER --master-port 29551 \
      $DRAFT_ARGS --context $CONTEXT $extra >/dev/null && \
    ( nohup docker logs -f $NAME > $OUT/server.log 2>&1 & ) && \
    ( nohup bash -c 'while [ ! -f $OUT/.stop-sampler ]; do nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv,noheader >> $OUT/gpu-1s.csv; sleep 1; done' >/dev/null 2>&1 & ) && echo started rank $r on \$(hostname)"
}
serve_rank "$Rt" 1 "--parallel 1 --max-tokens 32768" | tee -a "$LOCAL/$RUN/progress.log"        # rank 0 last (TensorFold's rule)
sleep 5
serve_rank "$L" 0 "--parallel 1 --max-tokens 32768 --host 127.0.0.1 --port $B3_PORT" | tee -a "$LOCAL/$RUN/progress.log"

t0=$SECONDS
until on "$L" "docker exec $NAME python3 -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:$B3_PORT/health',timeout=5)\"" >/dev/null 2>&1; do
  for h in $L $Rt; do
    [[ "$(on "$h" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null")" == true ]] || { log "FAIL: rank on $h exited after $((SECONDS-t0)) s"; on "$h" "tail -40 $OUT/server.log" | tee -a "$LOCAL/$RUN/progress.log"; exit 6; }
  done
  (( SECONDS - t0 > 3600 )) && { log "FAIL: not healthy after 3600 s"; exit 6; }
  sleep 15
done
log "healthy after $((SECONDS-t0)) s; GB300 used left $(gb300_used "$L") / right $(gb300_used "$Rt") MiB"
on "$L" "grep -E 'NET/IB|NCCL INFO Using|Channel|loaded in|startup estimate' $OUT/server.log | head -12" | tee -a "$LOCAL/$RUN/progress.log"

# ---------- smoke, GSM8K, bench (inside rank 0's container, loopback) ----------
X="docker exec $NAME python3 /opt/b1"
on "$L" "$X/smoke_tf.py http://127.0.0.1:$B3_PORT /out/smoke.json > $OUT/smoke.txt 2>&1; cat $OUT/smoke.txt" | tee "$LOCAL/$RUN/smoke.txt" | head -30
on "$L" "$X/gsm8k_tf.py http://127.0.0.1:$B3_PORT /opt/b1/gsm8k_test.jsonl /out/gsm8k50.jsonl --n 50 --conc 1 --think 1" | tee -a "$LOCAL/$RUN/progress.log"
b() { log "bench $1: slot $2 C$3 max_tokens $4 seed $5"; on "$L" "$X/bench_tf.py http://127.0.0.1:$B3_PORT /opt/b1/pool/$2.jsonl /out/bench-$1.jsonl --conc $3 --max-tokens $4 --seed $5 --label $1 > $OUT/bench-$1.summary.json; python3 -c \"import json;s=json.load(open('$OUT/bench-$1.summary.json'));f=lambda k:(s[k] or {}).get('mean');print('  decode',f('decode_tps'),'| output',f('output_tps'),'| agg',round(s['aggregate_output_tps'],1),'| ttft',f('ttft_s'),'| prefill',f('prefill_tps'),'| cached max',s['max_cached_frac'],'| finish',s['finish_reasons'])\"" | tee -a "$LOCAL/$RUN/progress.log" || log "bench $1 rc=$? (3 = CONTAMINATED)"; }
b warm-short warm-short 1 256 101
b c1-r1 c1-r1 1 1024 111; b c1-r2 c1-r2 1 1024 112; b c1-r3 c1-r3 1 1024 113
b warm-long8k warm-long8k 1 64 181; b prefill8k prefill8k 1 16 182; b c1long-r1 c1long-r1 1 1024 183
b warm-long32k warm-long32k 1 16 321; b prefill32k prefill32k 1 16 322
onin "$L" "python3 - $OUT" <<'EOF' | tee "$LOCAL/$RUN/B3-summary.txt"
import json,glob,os,statistics,sys
o=sys.argv[1]; S={os.path.basename(p)[6:-13]:json.load(open(p)) for p in glob.glob(o+'/bench-*.summary.json')}
m=lambda l,k:(S[l][k] or {}).get('mean') if l in S else None
c1=[m(f'c1-r{i}','decode_tps') for i in (1,2,3)]
print('C1 decode tok/s (mean of 3 runs):', round(statistics.fmean([x for x in c1 if x]),1) if any(c1) else None, [round(x,1) if x else None for x in c1])
print('C1 decode after 8K prompt:', m('c1long-r1','decode_tps'))
print('prefill tok/s 8K / 32K:', m('prefill8k','prefill_tps'), '/', m('prefill32k','prefill_tps'))
print('max cached fraction:', max(s['max_cached_frac'] for s in S.values()), '| contaminated:', [k for k,s in S.items() if s['contaminated']])
EOF
log "done"
