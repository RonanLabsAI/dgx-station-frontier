#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# realtext_bench.sh OUTDIR CONTAINER PORT SERVED  -- RT real-text sidecar A/B decode bench (2026-10-09). 
# DS=long   : public real text, CNN/DailyMail test articles concatenated to 8,190-8,191 tokens (chat format, thinking off),
#             1,024 FORCED output tokens (--ignore-eos).                 Pool: ${POOL:-./pool}/long  (build_realtext_pool.py)
# DS=short  : public real text, ShareGPT V3 first human turns, 12-498 tokens, NATURAL output (EOS honoured), max 1,024.
# DS=random : control, vLLM random ids 8192 in / 1024 forced out (the random-id A/B contract of results/sidecar-ab.md section 2).
# Per C in $CS: NP from $NPMAP; REPS reps at C1. decode = C x 1000 / mean TPOT; out = vLLM output token throughput.
# Hygiene (no reuse of any measured prompt; prefix-cache hit checked on every run):
#   * every invocation has its own seed (NONCE + running index) AND, for real text, its own disjoint prompt slice
#     (pool/<DS>/<label>-try<t>[-warm].jsonl; both arms use the same slices, so A and B see identical prompts);
#   * --num-warmups 0 --ready-check-timeout-sec 0; the warm-up is a SEPARATE invocation (own slice / seed, <= 64 out);
#   * /metrics prefix_cache_{hits,queries}_total delta per measured run; > MAXHIT% (2) -> retried once on try2 slice +
#     new seed; still > MAXHIT -> CONTAMINATED (never reported).
# Extra per measured run: DSpark acceptance from /metrics spec_decode_* deltas, wall start/end epoch (timings.tsv, for
# per-concurrency power), and route-counter snapshots around each concurrency block when SNAP=1 (route_snap.py).
set -uo pipefail
O=${1:?outdir}; CT=${2:?container}; P=${3:?port}; M=${4:?served name}
DS=${DS:-long}; MAXHIT=${MAXHIT:-2}; REPS=${REPS:-1}; TOKARGS=${TOKARGS:-}; SNAP=${SNAP:-0}
NONCE=${NONCE:-$(( ($(date +%s) % 1000000) * 1000 ))}; IDX=0
mkdir -p "$O"; S="$O/summary.txt"; TM="$O/timings.tsv"
echo "== $(date '+%F %T %Z') $CT :$P $M ds=$DS nonce=$NONCE image=$(docker inspect --format '{{.Config.Image}}' "$CT")" | tee -a "$S"
nextseed() { IDX=$((IDX + 1)); SEED=$((NONCE + IDX)); }
met() { curl -s http://127.0.0.1:$P/metrics | awk '
  /^vllm:prefix_cache_(hits|queries)_total/ || /^vllm:spec_decode_num_(accepted_tokens|drafts|draft_tokens)_total[{ ]/ {
    split($1,a,"{"); v[a[1]]+=$NF }
  END{printf "%d %d %d %d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"],
      v["vllm:spec_decode_num_accepted_tokens_total"], v["vllm:spec_decode_num_drafts_total"], v["vllm:spec_decode_num_draft_tokens_total"]}'; }
stats() { python3 -c "
a=list(map(float,'$1'.split())); b=list(map(float,'$2'.split())); d=[y-x for x,y in zip(a,b)]
hit=f'{100*d[0]/d[1]:.2f}' if d[1]>0 else 'NA'
acc=f'{1+d[2]/d[3]:.3f}' if d[3]>0 else 'NA'; rate=f'{100*d[2]/d[4]:.1f}' if d[4]>0 else 'NA'
print(hit, acc, rate)"; }
docker exec "$CT" mkdir -p /tmp/rt
# the image lacks pandas (vllm[bench]) which CustomDataset needs: pandas 2.3.3 + dateutil/pytz/tzdata, pip --no-deps --target
# ${PYLIB:-./pylib} (image numpy 2.2.6 kept), copied into the container and put on PYTHONPATH for the bench client only.
docker exec "$CT" test -d /tmp/rt/pylib/pandas || docker cp ${PYLIB:-./pylib} "$CT:/tmp/rt/pylib" >/dev/null
if [ "$DS" != random ]; then
  docker exec "$CT" test -d /tmp/rt/pool/$DS || docker cp ${POOL:-./pool}/. "$CT:/tmp/rt/pool/" >/dev/null
  docker exec "$CT" test -f /tmp/rt/pool/$DS/decode-c1-r1-try1.jsonl || { echo "pool missing in container" | tee -a "$S"; exit 3; }
fi
B=(docker exec -e PYTHONPATH=/tmp/rt/pylib "$CT" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P --endpoint /v1/completions --model "$M"
   --tokenizer /model --trust-remote-code $TOKARGS --temperature 0
   --num-warmups 0 --ready-check-timeout-sec 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90
   --save-result --result-dir /tmp/rt/res-$DS)
nvidia-smi --query-gpu=timestamp,index,name,power.draw,memory.used,utilization.gpu --format=csv -lms 1000 > "$O/power-$(hostname)-$DS.csv" & PW=$!
trap 'kill $PW 2>/dev/null' EXIT
bench1() { # LABEL C NP OUTLEN FILE(real text) ; seed from nextseed
  local L=$1 C=$2 NP=$3 OL=$4 F=$5 dsargs=()
  case "$DS" in
    random) dsargs=(--dataset-name random --random-range-ratio 0 --random-input-len "${ISL:-8192}" --random-output-len "$OL" --ignore-eos) ;;
    long)   dsargs=(--dataset-name custom --dataset-path "/tmp/rt/pool/long/$F" --custom-output-len "$OL" --ignore-eos
                    --skip-chat-template --disable-shuffle --no-oversample) ;;
    short)  dsargs=(--dataset-name custom --dataset-path "/tmp/rt/pool/short/$F" --custom-output-len "$OL"
                    --skip-chat-template --disable-shuffle --no-oversample) ;;
  esac
  "${B[@]}" "${dsargs[@]}" --max-concurrency "$C" --num-prompts "$NP" --seed "$SEED" --result-filename "$L.json" > "$O/$L.log" 2>&1; }
run() { # LABEL C NP
  local L=$1 C=$2 NP=$3 OL=${OSL:-1024} try m0 m1 st H ACC AR T TT OT F TIN TOUT DUR TAG WNP t0 t1
  for try in 1 2; do
    WNP=$(( C < 2 ? 1 : (C > 4 ? 4 : C) ))
    nextseed; local WS=$SEED
    bench1 "$L-warm$try" "$C" "$WNP" $(( OL > 64 ? 64 : OL )) "$L-try$try-warm.jsonl"   # warm-up: own slice/seed, discarded
    nextseed; m0=$(met); t0=$(date +%s.%N)
    bench1 "$L" "$C" "$NP" "$OL" "$L-try$try.jsonl"
    t1=$(date +%s.%N); m1=$(met)
    read -r H ACC AR <<< "$(stats "$m0" "$m1")"; TAG=""
    [ "$H" != NA ] && awk -v h="$H" -v m="$MAXHIT" 'BEGIN{exit !(h>m)}' && TAG=" CONTAMINATED(>${MAXHIT}%)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$DS" "$L" "try$try" "$t0" "$t1" "${TAG:-ok}" >> "$TM"
    [ -z "$TAG" ] && break
    [ "$try" = 2 ] && break   # still > MAXHIT on the retry: keep $L.log, the summary line carries CONTAMINATED
    mv "$O/$L.log" "$O/$L-try$try-CONTAMINATED.log"
    echo "$L try$try seed $SEED hit $H% -> retry with a new slice + seed" | tee -a "$S"
  done
  T=$(awk '/Mean TPOT/{print $NF}' "$O/$L.log"); TT=$(awk '/Mean TTFT/{print $NF}' "$O/$L.log")
  OT=$(awk '/Output token throughput/{print $NF}' "$O/$L.log"); F=$(awk '/Failed requests/{print $NF}' "$O/$L.log")
  TIN=$(awk '/Total input tokens/{print $NF}' "$O/$L.log"); TOUT=$(awk '/Total generated tokens/{print $NF}' "$O/$L.log")
  DUR=$(awk '/Benchmark duration/{print $NF}' "$O/$L.log")
  python3 - "$DS" "$L" "$SEED" "$WS" "$C" "${T:-0}" "${OT:-?}" "${TT:-?}" "${F:-?}" "$H" "$ACC" "$AR" "${TIN:-?}" "${TOUT:-?}" "${DUR:-?}" "$try" "$TAG" <<'PY' | tee -a "$S"
import sys
ds,L,seed,ws,c,t,ot,tt,f,h,acc,ar,tin,tout,dur,tr,tag=sys.argv[1:]; t=float(t)
print(f"{ds} {L} try{tr} seed {seed} warmseed {ws}: decode_agg {int(c)*1000/t if t else 0:.2f} tok/s (C{c} x 1000/meanTPOT {t} ms) | "
      f"out_tput {ot} | meanTTFT {tt} ms | in {tin} out {tout} tok in {dur} s | dspark_acc_len {acc} draft_acc {ar}% | "
      f"failed {f} | prefix_hit {h}%{tag}")
PY
}
snap() { [ "$SNAP" = 1 ] && python3 "$(dirname "$0")/route_snap.py" "$P" "$M" "$O/snaps" "$DS-$1" 2>&1 | tee -a "$S"; }
snap start
for C in ${CS:-1 16 32}; do
  NP=$(echo "${NPMAP:-1:5 16:80 32:160}" | tr ' ' '\n' | awk -F: -v c="$C" '$1==c{print $2}'); NP=${NP:-$((5 * C))}
  R=1; [ "$C" = 1 ] && R=$REPS
  for r in $(seq 1 "$R"); do run "decode-c$C-r$r" "$C" "$NP"; done
  snap "after-c$C"
done
docker cp "$CT:/tmp/rt/res-$DS/." "$O/" >/dev/null 2>&1
echo "DONE $DS $(date '+%F %T')" | tee -a "$S"
