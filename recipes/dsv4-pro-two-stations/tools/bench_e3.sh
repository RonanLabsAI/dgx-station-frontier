#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# bench_e3.sh OUTDIR SEEDBASE [REPS=3] -- run on the rank-0 Station against the rank-0 container (NAME, default dsv4pro-e3).
# catid 8K/1K random (vllm bench serve, ignore-eos) C1 (NP=3) and C16 (NP=32), REPS each, plus the held-out real-text
# bench (realbench.py, thinking off, 1,024 forced tokens) C1 (n=3) and C16 (n=32), REPS each. The real-text bench needs
# REALBENCH=<your script>: ours sends prompts from a private workload and is not published (see the recipe README).
# Hygiene (README, Methodology): every catid run gets its own seed (SEEDBASE + 10*C + rep), the warm-up uses a
# separate seed and a different shape, and each run's prefix-cache hit rate is measured from /metrics deltas
# (vllm:prefix_cache_hits / vllm:prefix_cache_queries). A run with hit > 2% is marked CONTAMINATED and must not be reported.
# Decode tok/s = C x 1000 / mean TPOT (same definition as bench-dsv4pro.sh / realbench.py).
set -uo pipefail
O=${1:?outdir}; SB=${2:?seedbase}; REPS=${3:-3}; N=${NAME:-dsv4pro-e3}; P=${PORT:-8010}; M=deepseek-v4-pro
mkdir -p "$O"; S="$O/summary.txt"
met() { curl -s http://127.0.0.1:$P/metrics | awk '/^vllm:prefix_cache_(hits|queries)_total/{split($1,a,"{"); v[a[1]]+=$NF} END{printf "%d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"]}'; }
hit() { python3 -c "h0,q0,h1,q1=map(float,'$1 $2'.split()); dq=q1-q0; r=(h1-h0)/dq if dq>0 else 0; print(f'{100*r:.2f}% ({int(h1-h0)}/{int(dq)} tok)' + (' CONTAMINATED' if r>0.02 else ''))"; }
nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used --format=csv -lms 2000 > "$O/power-left.csv" & PL=$!
PEER=${PEER_SSH:?user@rank-1 rail IP}
ssh -o BatchMode=yes $PEER "mkdir -p $O; nohup nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used --format=csv -lms 2000 > $O/power-right.csv 2>&1 & echo \$! > $O/power-right.pid"
trap 'kill $PL 2>/dev/null; ssh -o BatchMode=yes $PEER "kill \$(cat $O/power-right.pid) 2>/dev/null"' EXIT
B=(docker exec "$N" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P --model $M --tokenizer /model --trust-remote-code)
"${B[@]}" --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 4 --max-concurrency 4 --ignore-eos \
  --seed $((SB + 900)) > "$O/warmup.log" 2>&1
echo "== $(date '+%F %T') seedbase $SB" | tee -a "$S"
for C in 1 16; do
  NP=$([ $C = 1 ] && echo 3 || echo 32)
  for R in $(seq 1 "$REPS"); do
    SEED=$((SB + 10 * C + R)); m0=$(met)
    "${B[@]}" --dataset-name random --random-input-len 8192 --random-output-len 1024 --num-prompts $NP --max-concurrency $C \
      --ignore-eos --seed $SEED --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 > "$O/catid-c$C-r$R.log" 2>&1
    m1=$(met); T=$(awk '/Mean TPOT/{print $NF}' "$O/catid-c$C-r$R.log"); F=$(awk '/Failed requests/{print $NF}' "$O/catid-c$C-r$R.log")
    echo "catid C$C rep$R seed $SEED: decode_agg $(python3 -c "print(round($C*1000/float('$T'),2))") (TPOT $T ms) failed ${F:-?} hit $(hit "$m0" "$m1")" | tee -a "$S"
  done
done
[ -n "${REALBENCH:-}" ] && for C in 1 16; do
  NN=$([ $C = 1 ] && echo 3 || echo 32)
  for R in $(seq 1 "$REPS"); do
    m0=$(met)
    python3 "$REALBENCH" http://127.0.0.1:$P $M "$O/real-c$C-r$R.json" $C $NN > "$O/real-c$C-r$R.log" 2>&1
    m1=$(met)
    echo "real C$C rep$R: $(tail -1 "$O/real-c$C-r$R.log") hit $(hit "$m0" "$m1")" | tee -a "$S"
  done
done
echo "DONE $(date +%T)" | tee -a "$S"
