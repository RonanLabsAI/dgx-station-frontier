#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# bench-nemotron-ultra.sh OUTDIR SEEDBASE -- throughput bench against the running nemotron-ultra container (rank 0 host).
# Shape: random 8,192 in / 1,024 out, ignore-eos. C1 NP=3 x REPS (default 3; report the mean), C4 NP=8, C16 NP=32,
# C32 NP=64. Prefill at C1: 8K in / 1 out (NP=4) and 64K in / 1 out (NP=3); prefill tok/s = ISL / mean TTFT.
# Hygiene: every run has its own seed (SEEDBASE + 100*kind + 10*C + rep); the warm-up uses its own seed (SEEDBASE+999) and
# a different shape (512/64). The server runs --no-enable-prefix-caching, and each run's hit rate is still measured from
# /metrics deltas: hit > 2% => CONTAMINATED (do not report).
# Decode tok/s = C x 1000 / mean TPOT. Output throughput is logged too.
# Power: nvidia-smi every 1 s on all GPUs of this host, and of PEER (user@host of the second Station, optional) over ssh;
# per-run windows go to windows.tsv; summarise with power_windows.py OUTDIR (copy PEER's power-right.csv next to it).
set -uo pipefail
O=${1:?outdir}; SB=${2:?seedbase}; REPS=${REPS:-3}; N=${NAME:-nemotron-ultra}; P=${PORT:-8017}; M=${SERVED:-nemotron-3-ultra}
SKIP64=${SKIP64:-0}; CONCS=${CONCS:-"1 4 16 32"}; PEER=${PEER:-}
mkdir -p "$O"; S="$O/summary.txt"; W="$O/windows.tsv"; : > "$W"
met() { curl -s http://127.0.0.1:$P/metrics | awk '/^vllm:prefix_cache_(hits|queries)_total/{split($1,a,"{"); v[a[1]]+=$NF}
  /^vllm:spec_decode_num_accepted_tokens_total/{acc+=$NF} /^vllm:spec_decode_num_drafts_total/{dr+=$NF}
  END{printf "%d %d %d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"], acc, dr}'; }
hit() { python3 -c "
h0,q0,a0,d0,h1,q1,a1,d1=map(float,'$1 $2'.split()); dq=q1-q0; r=(h1-h0)/dq if dq>0 else 0.0
s=f'{100*r:.2f}% ({int(h1-h0)}/{int(dq)} tok)' + (' CONTAMINATED' if r>0.02 else '') + ('' if dq>0 else ' [prefix cache off]')
dd=d1-d0; s+=f' mtp_acc_len {1+(a1-a0)/dd:.2f}' if dd>0 else ''
print(s)"; }
Q="timestamp,index,name,power.draw,utilization.gpu,memory.used"
nvidia-smi --query-gpu=$Q --format=csv -lms 1000 > "$O/power-left.csv" & PL=$!
if [ -n "$PEER" ]; then
  ssh -o BatchMode=yes "$PEER" "mkdir -p $O; nohup nvidia-smi --query-gpu=$Q --format=csv -lms 1000 > $O/power-right.csv 2>&1 & echo \$! > $O/power-right.pid"
  trap 'kill $PL 2>/dev/null; ssh -o BatchMode=yes "$PEER" "kill \$(cat $O/power-right.pid) 2>/dev/null"' EXIT
else
  trap 'kill $PL 2>/dev/null' EXIT
fi
B=(docker exec "$N" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P --model $M --tokenizer /model --trust-remote-code)
win() { echo -e "$1\t$2\t$(date '+%F %T')" >> "$W"; }   # label, start|end

"${B[@]}" --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 4 --max-concurrency 4 --ignore-eos \
  --seed $((SB + 999)) > "$O/warmup.log" 2>&1
echo "== $(date '+%F %T %Z') seedbase $SB container $N image $(docker inspect --format '{{.Image}}' $N)" | tee -a "$S"

run() {  # label C NP ISL OSL SEED
  local L=$1 C=$2 NP=$3 I=$4 OL=$5 SEED=$6 m0 m1 T TT F OT
  m0=$(met); win "$L" start
  "${B[@]}" --dataset-name random --random-input-len "$I" --random-output-len "$OL" --random-range-ratio 0 --num-prompts "$NP" \
    --max-concurrency "$C" --ignore-eos --seed "$SEED" --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 \
    > "$O/$L.log" 2>&1
  win "$L" end; m1=$(met)
  T=$(awk '/Mean TPOT/{print $NF}' "$O/$L.log"); TT=$(awk '/Mean TTFT/{print $NF}' "$O/$L.log")
  F=$(awk '/Failed requests/{print $NF}' "$O/$L.log"); OT=$(awk '/Output token throughput/{print $NF}' "$O/$L.log")
  if [ "$OL" = 1 ]; then
    echo "$L seed $SEED: prefill $(python3 -c "print(round($I/(float('${TT:-0}')/1000),1))" 2>/dev/null) tok/s (mean TTFT $TT ms) failed ${F:-?} hit $(hit "$m0" "$m1")" | tee -a "$S"
  else
    echo "$L seed $SEED: decode_agg $(python3 -c "print(round($C*1000/float('${T:-0}'),2))" 2>/dev/null) (TPOT $T ms, TTFT $TT ms, out_tput $OT) failed ${F:-?} hit $(hit "$m0" "$m1")" | tee -a "$S"
  fi
}
for C in $CONCS; do
  case $C in 1) NP=3; R=$REPS ;; 4) NP=8; R=1 ;; 16) NP=32; R=1 ;; 32) NP=64; R=1 ;; *) NP=$((2*C)); R=1 ;; esac
  for r in $(seq 1 "$R"); do run "catid-c$C-r$r" "$C" "$NP" 8192 1024 $((SB + 100 + 10*C + r)); done
done
run prefill-8k-c1 1 4 8192 1 $((SB + 200 + 1))
[ "$SKIP64" = 1 ] || run prefill-64k-c1 1 3 65536 1 $((SB + 300 + 1))
python3 - "$S" <<'EOF' | tee -a "$S"
import re, sys
v = [float(m.group(1)) for m in re.finditer(r'catid-c1-r\d seed \d+: decode_agg ([\d.]+)', open(sys.argv[1]).read())]
print(f"C1 mean over {len(v)} reps: {sum(v)/len(v):.2f} (min {min(v):.2f} max {max(v):.2f})" if v else "C1 mean: n/a")
EOF
echo "DONE $(date +%T)" | tee -a "$S"
